def load_oof(model, variant):
    """Return (OOF, fold-averaged test predictions) when all five folds exist, else None."""
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{model}__{variant}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{model} {variant} has incomplete OOF predictions')
    return oof, test_pred


def fold_rmse(pred):
    return pd.Series({k: rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)})


def paired_table(preds, base=VARIANT_CONTEXT_BASELINE):
    require(base in preds, f'Missing baseline {base}')
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


def blend_variant(variant):
    """NNLS à trois modèles ; réutiliser la référence avec contexte pour les modèles non modifiés."""
    loaded = []
    for model in MODELS:
        item = load_oof(model, variant if model in models_for(variant) else VARIANT_CONTEXT_BASELINE)
        if item is None:
            return None
        loaded.append(item)
    P_oof = np.column_stack([o for o, _ in loaded])
    P_test = np.column_stack([t for _, t in loaded])
    w, _ = nnls(P_oof, y)
    return {'oof': P_oof @ w, 'test': P_test @ w, 'weights': [round(float(v), 3) for v in w]}


results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))])
for model in MODELS:
    preds = {}
    for variant in HISTORY_VARIANTS:
        item = load_oof(model, variant) if model in models_for(variant) else None
        if item is not None:
            preds[variant] = item[0]
    if VARIANT_CONTEXT_BASELINE in preds:
        table = paired_table(preds)
        part = results[results['model'] == model].groupby('variant')
        table['mean_best_iter'] = part['best_iteration'].mean()
        table['minutes'] = part['seconds'].sum() / 60
        print(f'\n=== {model}')
        display(table.round(3))
        table.to_csv(OUTPUT_DIR / f'summary_single_{model}.csv')

BLENDS = {}
for variant in HISTORY_VARIANTS:
    item = blend_variant(variant)
    if item is not None:
        BLENDS[variant] = item
require(VARIANT_CONTEXT_BASELINE in BLENDS, 'Les résultats de la référence avec contexte sont incomplets pour les trois modèles')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cb'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\n=== Three-model NNLS blend')
display(blend_table.round(3))
blend_table.to_csv(OUTPUT_DIR / 'summary_blend_screen.csv')

KEPT = [v for v in [BLOCK_EXTENDED_MEET_CONTEXT, BLOCK_DIVISION_CONTEXT, BLOCK_ATHLETE_HISTORY, BLOCK_DOTS_SCALE, BLOCK_PROXY_IDENTIFIERS] if v in blend_table.index and bool(blend_table.loc[v, 'keep'])]
HISTORY_VARIANTS[VARIANT_HISTORY_COMBINED] = KEPT
MODELS_FOR[VARIANT_HISTORY_COMBINED] = [MODEL_CATBOOST_ABSOLUTE] if KEPT and all(b in CATBOOST_CATEGORY_BLOCKS for b in KEPT) else MODELS
print('Retained blocks: ', KEPT or 'none', '→ ALL2 =', [VARIANT_CONTEXT_BASELINE] + KEPT)
