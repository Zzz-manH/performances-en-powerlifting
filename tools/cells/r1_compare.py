def load_results():
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))]
    require(rows, 'No completed results')
    return pd.DataFrame(rows)


def load_oof(tag, variant):
    """Return (OOF, fold-averaged test predictions) when all five folds exist, else (None, None)."""
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{tag}__{variant}__fold{fold}.npz'
        if not path.is_file():
            return None, None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{tag} {variant} has incomplete OOF predictions')
    return oof, test_pred


def fold_rmse(pred):
    return pd.Series({k: rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)})


def paired_table(preds, base=VARIANT_WEIGHT_CLASS_BASELINE):
    """preds maps variants to OOF predictions; compare with base on paired folds."""
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


results = load_results()
SINGLE, TESTS = {}, {}
for tag in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_DEPTH8_BASELINE]:
    preds = {}
    for variant in CONTEXT_VARIANTS:
        oof, test_pred = load_oof(tag, variant)
        if oof is not None:
            preds[variant], TESTS[(tag, variant)] = oof, test_pred
    if VARIANT_WEIGHT_CLASS_BASELINE in preds:
        SINGLE[tag] = paired_table(preds)
        part = results[results['tag'] == tag].groupby('variant')
        SINGLE[tag]['mean_best_iter'] = part['best_iteration'].mean()
        SINGLE[tag]['minutes'] = part['seconds'].sum() / 60
        print(f'\n=== {tag}')
        display(SINGLE[tag].round(3))

BLENDS = {}
for variant in CONTEXT_VARIANTS:
    if all((tag, variant) in TESTS for tag in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_DEPTH8_BASELINE]):
        P_oof = np.column_stack([load_oof(tag, variant)[0] for tag in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_DEPTH8_BASELINE]])
        P_test = np.column_stack([TESTS[(tag, variant)] for tag in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_DEPTH8_BASELINE]])
        w, _ = nnls(P_oof, y)
        BLENDS[variant] = {'oof': P_oof @ w, 'test': P_test @ w, 'weights': w.round(3).tolist()}
require(VARIANT_WEIGHT_CLASS_BASELINE in BLENDS, 'F0 is incomplete for the three base models')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cb'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\n=== Three-model NNLS blend')
display(blend_table.round(3))
BEST = blend_table.index[0]
print(f'Best OOF blend: {BEST}; retained variants: {blend_table.index[blend_table["keep"]].tolist() or "none"}')

pd.concat({tag: t for tag, t in SINGLE.items()}, names=['model']).to_csv(OUTPUT_DIR / 'summary_single.csv')
blend_table.to_csv(OUTPUT_DIR / 'summary_blend.csv')
