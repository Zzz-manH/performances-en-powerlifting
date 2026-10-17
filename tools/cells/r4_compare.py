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


def paired_table(preds, base='R3'):
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
    preds = {v: load_oof(model, v)[0] for v in VARIANTS4 if load_oof(model, v) is not None}
    if 'R3' in preds:
        print(f'\n=== {model}')
        display(paired_table(preds).round(3))
BLENDS = {v: blend_variant(v) for v in VARIANTS4}
BLENDS = {v: b for v, b in BLENDS.items() if b is not None}
require('R3' in BLENDS, 'R3 还没有全部完成')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cbrel'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\n=== 三模型 NNLS 融合（主要判读依据）')
display(blend_table.round(3))
blend_table.to_csv(OUTPUT_DIR / 'summary_blend_screen.csv')
KEPT4 = [v for v in VARIANTS4 if v != 'R3' and v in blend_table.index and bool(blend_table.loc[v, 'keep'])]
print('达标的新块：', KEPT4 or '无')
if len(KEPT4) >= 2:
    VARIANTS4['ALL4'] = [b for v in KEPT4 for b in VARIANTS4[v]]
    run_variants(['ALL4'], SCREEN_MODELS)
    BLENDS['ALL4'] = blend_variant('ALL4')
    display(paired_table({v: BLENDS[v]['oof'] for v in ['R3'] + KEPT4 + ['ALL4']}).round(3))
FINAL4 = 'ALL4' if 'ALL4' in BLENDS else (KEPT4[0] if KEPT4 else None)
if FINAL4 is None and len(blend_table) > 1:
    FINAL4 = blend_table.index[0] if blend_table.index[0] != 'R3' else blend_table.index[1]
print('与 R3 做净收益比较的变体：', FINAL4)
