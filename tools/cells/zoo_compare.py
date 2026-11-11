def load_oof(directory, stem):
    oof, pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = directory / f'{stem}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{stem} has incomplete OOF predictions')
    return oof, pred


def fold_rmse(pred):
    return np.array([rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)])


MODELS = {}
for tag in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_DEPTH8_BASELINE, MODEL_CATBOOST_DEPTH8_REGULARIZED, MODEL_CATBOOST_DEPTH6_REGULARIZED]:
    loaded = load_oof(CONTEXT_ABLATION_DIR, f'{tag}__{FEATURE_SET}')
    if loaded is not None:
        MODELS[f'07_{tag}'] = loaded
BASE_NAMES = list(MODELS)
for name in ZOO:
    loaded = load_oof(OUTPUT_DIR, name)
    if loaded is not None:
        MODELS[name] = loaded
require(MODELS, 'No completed models')
print('Models from experiment 07: ', BASE_NAMES or 'none (experiment 07 incomplete or different features)', '| Models from this notebook: ',
      [n for n in ZOO if n in MODELS])


def nnls_oof(names_):
    P_ = np.column_stack([MODELS[n][0] for n in names_])
    w, _ = nnls(P_, y)
    return P_ @ w


base_oof = nnls_oof(BASE_NAMES) if BASE_NAMES else None
rows = []
for name, (oof, _) in MODELS.items():
    row = {'model': name, 'oof_rmse': rmse(y, oof), **{f'fold{k}': v for k, v in enumerate(fold_rmse(oof), 1)}}
    if base_oof is not None and name not in BASE_NAMES:
        blended = nnls_oof(BASE_NAMES + [name])
        row.update({'corr_with_07_blend': float(np.corrcoef(oof, base_oof)[0, 1]),
                    'blend_with_07': rmse(y, blended),
                    'delta_vs_07_blend': rmse(y, blended) - rmse(y, base_oof),
                    'folds_better': int((fold_rmse(blended) < fold_rmse(base_oof)).sum())})
    rows.append(row)
standalone = pd.DataFrame(rows).set_index('model').sort_values('oof_rmse')
if base_oof is not None:
    print(f'Experiment 07 NNLS blend, OOF RMSE: {rmse(y, base_oof):.3f}')
display(standalone.round(3))
standalone.to_csv(OUTPUT_DIR / 'summary_models.csv')

# ---- Three stacking methods; cross-fit meta-models on five groups of stored OOF. ----
NAMES = list(MODELS)
oof_prediction_matrix = np.column_stack([MODELS[n][0] for n in NAMES])
test_prediction_matrix = np.column_stack([MODELS[n][1] for n in NAMES])
segment = (both['Sex'].astype(str) + '|' + both['Equipment'].astype(str)).to_numpy()
SEG, SEG_T = segment[:n_train], segment[n_train:]
META = pd.DataFrame({'Sex': both['Sex'].astype(str).astype('category'),
                     'Equipment': both['Equipment'].astype(str).astype('category'),
                     'BodyweightKg': both['BodyweightKg'].to_numpy(dtype='float64'),
                     'Age': both['Age'].to_numpy(dtype='float64')})
META_TR, META_TE = META.iloc[:n_train].reset_index(drop=True), META.iloc[n_train:].reset_index(drop=True)
META_PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=500, subsample=0.8,
                   subsample_freq=1, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)


def global_nnls(fit, apply_P):
    w, _ = nnls(oof_prediction_matrix[fit], y[fit])
    return apply_P @ w


def segment_nnls(fit, apply_P, apply_seg, min_rows=3000):
    out = global_nnls(fit, apply_P)
    for s in np.unique(apply_seg):
        m_fit = fit & (SEG == s)
        if m_fit.sum() >= min_rows:
            w_s, _ = nnls(oof_prediction_matrix[m_fit], y[m_fit])
            m_apply = apply_seg == s
            out[m_apply] = apply_P[m_apply] @ w_s
    return out


def residual_meta(fit, apply_P, apply_meta):
    """Fit global NNLS, then a small LightGBM residual model using sex, equipment, weight, age and model disagreement."""
    w, _ = nnls(oof_prediction_matrix[fit], y[fit])
    base_fit, base_apply = oof_prediction_matrix[fit] @ w, apply_P @ w
    X_fit = META_TR[fit].assign(blend=base_fit, spread=oof_prediction_matrix[fit].std(axis=1))
    X_apply = apply_meta.reset_index(drop=True).assign(blend=base_apply, spread=apply_P.std(axis=1))
    model = lgb.LGBMRegressor(**META_PARAMS).fit(X_fit, y[fit] - base_fit)
    return base_apply + model.predict(X_apply)


METHODS = {
    'global_nnls': lambda fit, va: global_nnls(fit, oof_prediction_matrix[va]),
    'segment_nnls': lambda fit, va: segment_nnls(fit, oof_prediction_matrix[va], SEG[va]),
    'residual_meta': lambda fit, va: residual_meta(fit, oof_prediction_matrix[va], META_TR[va]),
}
HONEST = {}
for method, fn in METHODS.items():
    pred = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        pred[va] = fn(fit, va)
    HONEST[method] = pred
stack_table = pd.DataFrame({m: {'nested_oof_rmse': rmse(y, p), **{f'fold{k}': v for k, v in enumerate(fold_rmse(p), 1)}}
                            for m, p in HONEST.items()}).T.sort_values('nested_oof_rmse')
display(stack_table.round(3))
stack_table.to_csv(OUTPUT_DIR / 'summary_stacking.csv')
BEST_METHOD = stack_table.index[0]
everything = np.ones(n_train, dtype=bool)
STACK_TEST = {'global_nnls': lambda: global_nnls(everything, test_prediction_matrix),
              'segment_nnls': lambda: segment_nnls(everything, test_prediction_matrix, SEG_T),
              'residual_meta': lambda: residual_meta(everything, test_prediction_matrix, META_TE)}[BEST_METHOD]()
STACK_OOF = HONEST[BEST_METHOD]
print(f'Selected stacking method: {BEST_METHOD}(Local OOF RMSE {rmse(y, STACK_OOF):.3f}), using {len(NAMES)} models: {NAMES}')
