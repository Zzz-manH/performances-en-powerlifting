def load_oof(model, variant):
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{model}__{variant}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    return oof, test_pred


def fold_rmse(pred):
    return pd.Series({k: rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)})


def paired_table(preds, base=VARIANT_FINAL_BASELINE):
    base_folds = fold_rmse(preds[base])
    rows = []
    for name, pred in preds.items():
        delta = fold_rmse(pred) - base_folds
        rows.append({'variant': name, 'oof_rmse': rmse(y, pred), 'mean_delta': delta.mean(),
                     'folds_better': int((delta < 0).sum()), **{f'd_fold{k}': v for k, v in delta.items()}})
    table = pd.DataFrame(rows).set_index('variant')
    table['keep'] = (table['mean_delta'] <= KEEP_DELTA) & (table['folds_better'] >= 4)
    table.loc[base, 'keep'] = False
    return table.sort_values('oof_rmse')


def blend_variant(variant, models=SCREEN_MODELS):
    loaded = [load_oof(m, variant) for m in models]
    if any(item is None for item in loaded):
        return None
    P_oof = np.column_stack([o for o, _ in loaded])
    P_test = np.column_stack([t for _, t in loaded])
    w, _ = nnls(P_oof, y)
    return {'oof': P_oof @ w, 'test': P_test @ w, 'weights': [round(float(x), 3) for x in w]}


for model in SCREEN_MODELS:
    preds = {v: load_oof(model, v)[0] for v in PERFORMANCE_VARIANTS if load_oof(model, v) is not None}
    if VARIANT_FINAL_BASELINE in preds:
        print(f'\n=== {model}')
        display(paired_table(preds).round(3))
BLENDS = {v: blend_variant(v) for v in PERFORMANCE_VARIANTS}
BLENDS = {v: b for v, b in BLENDS.items() if b is not None}
require(VARIANT_FINAL_BASELINE in BLENDS, 'Les résultats de la référence avec variables finales sont incomplets')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cbrel'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\n=== Three-model NNLS blend')
display(blend_table.round(3))
blend_table.to_csv(OUTPUT_DIR / 'summary_blend_screen.csv')
KEPT4 = [v for v in PERFORMANCE_VARIANTS if v != VARIANT_FINAL_BASELINE and v in blend_table.index and bool(blend_table.loc[v, 'keep'])]
print('Retained blocks: ', KEPT4 or 'none')
if len(KEPT4) >= 2:
    PERFORMANCE_VARIANTS[VARIANT_PERFORMANCE_COMBINED] = [b for v in KEPT4 for b in PERFORMANCE_VARIANTS[v]]
    run_variants([VARIANT_PERFORMANCE_COMBINED], SCREEN_MODELS)
    BLENDS[VARIANT_PERFORMANCE_COMBINED] = blend_variant(VARIANT_PERFORMANCE_COMBINED)
    display(paired_table({v: BLENDS[v]['oof'] for v in [VARIANT_FINAL_BASELINE] + KEPT4 + [VARIANT_PERFORMANCE_COMBINED]}).round(3))
FINAL4 = VARIANT_PERFORMANCE_COMBINED if VARIANT_PERFORMANCE_COMBINED in BLENDS else (KEPT4[0] if KEPT4 else None)
if FINAL4 is None and len(blend_table) > 1:
    FINAL4 = blend_table.index[0] if blend_table.index[0] != VARIANT_FINAL_BASELINE else blend_table.index[1]
print('Variante comparée à la référence avec variables finales après post-traitement : ', FINAL4)
