LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=20, verbose=-1)
XGB_PARAMS = dict(learning_rate=0.03, max_depth=8, min_child_weight=10.0, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=1.0, tree_method='hist', device='cuda', enable_categorical=True,
                  max_cat_to_onehot=1, eval_metric='rmse')
CB_PARAMS = dict(learning_rate=0.05, depth=8, l2_leaf_reg=10.0, bootstrap_type='Bernoulli', subsample=0.8,
                 border_count=128)
TABM_PARAMS = dict(device='cuda', n_cv=1, n_refit=0, verbosity=0, num_emb_type='pwl')
TABM_FALLBACK = dict(device='cuda', n_cv=1, n_refit=0, verbosity=0)
PATIENCE = {MODEL_LIGHTGBM: 200, MODEL_LIGHTGBM_DOTS: 200, MODEL_XGBOOST: 200, MODEL_CATBOOST_ABSOLUTE: 300, MODEL_CATBOOST_DOTS: 300}
GBDT = [m for m in FINAL_MODELS if m != 'tabm']

SCALE = dots_scale(train['Sex'], train['BodyweightKg'])           # dots_scale is defined in the experiment 09 features.
SCALE_TEST = dots_scale(test['Sex'], test['BodyweightKg'])
require(min(SCALE.min(), SCALE_TEST.min()) > 0.3, 'Invalid DOTS scale')

X_NN, XT_NN, CATS_NN = FRAMES['tabm']
NUM_NN = [c for c in X_NN.columns if c not in CATS_NN]
NAN_NN = [c for c in NUM_NN if X_NN[c].isna().any() or XT_NN[c].isna().any()]


def target_and_weight(model, rows):
    if model.endswith('_rel'):
        return y[rows] / SCALE[rows], SCALE[rows] ** 2       # Relative error * scale = error in kg
    return y[rows], None


def gbdt_fit(model, seed, Xa, ya, wa, Xs=None, ys=None, ws=None, rounds=MAX_ROUNDS):
    """No early stopping when Xs is None. Return (model, best iteration)."""
    cats = FRAMES[model][2]
    if model in (MODEL_LIGHTGBM, MODEL_LIGHTGBM_DOTS):
        m = lgb.LGBMRegressor(n_estimators=rounds, random_state=seed, n_jobs=-1, **LGB_PARAMS)
        if Xs is None:
            m.fit(Xa, ya, sample_weight=wa)
            return m, rounds
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xs, ys)], eval_sample_weight=None if ws is None else [ws],
              callbacks=[lgb.early_stopping(PATIENCE[model], verbose=False)])
        return m, int(m.best_iteration_ or rounds)
    if model == MODEL_XGBOOST:
        if Xs is None:
            m = xgb.XGBRegressor(n_estimators=rounds, random_state=seed, **XGB_PARAMS)
            m.fit(Xa, ya, verbose=False)
            return m, rounds
        m = xgb.XGBRegressor(n_estimators=rounds, random_state=seed, early_stopping_rounds=PATIENCE[MODEL_XGBOOST],
                             **XGB_PARAMS)
        m.fit(Xa, ya, eval_set=[(Xs, ys)], verbose=False)
        return m, int(m.best_iteration) + 1
    common = dict(iterations=rounds, loss_function='RMSE', eval_metric='RMSE', random_seed=seed, verbose=0,
                  task_type='GPU', devices='0', allow_writing_files=False, **CB_PARAMS)
    if Xs is None:
        m = catboost.CatBoostRegressor(**common)
        m.fit(catboost.Pool(Xa, ya, cat_features=cats, weight=wa))
        return m, rounds
    m = catboost.CatBoostRegressor(od_type='Iter', od_wait=PATIENCE[model], **common)
    m.fit(catboost.Pool(Xa, ya, cat_features=cats, weight=wa),
          eval_set=catboost.Pool(Xs, ys, cat_features=cats, weight=ws), use_best_model=True)
    return m, int(m.get_best_iteration()) + 1


def gbdt_predict(model, m, best, X):
    if model in (MODEL_LIGHTGBM, MODEL_LIGHTGBM_DOTS):
        return m.predict(X, num_iteration=best)
    if model == MODEL_XGBOOST:
        return m.predict(X, iteration_range=(0, best))
    return m.predict(X)


def nn_frames(*frames):
    """Impute from the first fitting frame and add missingness flags; keep category dtype."""
    median = frames[0][NUM_NN].median()
    out = []
    for frame in frames:
        z = frame.copy()
        for c in NAN_NN:
            z[f'{c}_isna'] = z[c].isna().astype('float64')
        z[NUM_NN] = z[NUM_NN].fillna(median).fillna(0.0)
        out.append(z.reset_index(drop=True))
    return out


def tabm_fit_predict(seed, a, s, v, Xt, overrides=None):
    require(TabM_D_Regressor is not None, f'pytabkit unavailable: {PYTABKIT_ERROR}')
    params = dict(TABM_PARAMS, random_state=seed, **(overrides or {}))
    unknown = set(params) - set(inspect.signature(TabM_D_Regressor.__init__).parameters)
    require(not unknown, f'TabM does not support parameters {sorted(unknown)}')
    Za, Zs, Zv, Zt = nn_frames(X_NN.iloc[a], X_NN.iloc[s], X_NN.iloc[v], Xt)
    mu, sd = y[a].mean(), y[a].std()
    m = TabM_D_Regressor(**params)
    m.fit(Za, (y[a] - mu) / sd, Zs, (y[s] - mu) / sd, cat_col_names=[c for c in Za.columns if c in CATS_NN])
    p_val = np.asarray(m.predict(Zv), dtype='float64').reshape(-1) * sd + mu
    p_test = np.asarray(m.predict(Zt), dtype='float64').reshape(-1) * sd + mu
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test


def fit_fold(model, seed, a, s, v, test_rows=None, rounds=MAX_ROUNDS, overrides=None):
    """a / s / v: inner fitting / early-stopping / outer validation indices. Return (validation predictions, test predictions, best iteration or None)."""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale_t = SCALE_TEST if test_rows is None else SCALE_TEST[test_rows]
    if model == 'tabm':
        p_val, p_test = tabm_fit_predict(seed, a, s, v, Xt, overrides)
        return p_val, p_test, None
    ya, wa = target_and_weight(model, a)
    ys, ws = target_and_weight(model, s)
    if wa is not None:
        norm = wa.mean()
        wa, ws = wa / norm, ws / norm
    m, best = gbdt_fit(model, seed, Xtr.iloc[a], ya, wa, Xtr.iloc[s], ys, ws, rounds)
    p_val, p_test = gbdt_predict(model, m, best, Xtr.iloc[v]), gbdt_predict(model, m, best, Xt)
    if model.endswith('_rel'):
        p_val, p_test = np.asarray(p_val) * SCALE[v], np.asarray(p_test) * scale_t
    return np.asarray(p_val, dtype='float64'), np.asarray(p_test, dtype='float64'), best


def fit_full(model, seed, rounds, test_rows=None):
    """Fit all rows for a fixed number of iterations; return test predictions."""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale_t = SCALE_TEST if test_rows is None else SCALE_TEST[test_rows]
    rows = np.arange(n_train)
    ya, wa = target_and_weight(model, rows)
    if wa is not None:
        wa = wa / wa.mean()
    m, _ = gbdt_fit(model, seed, Xtr, ya, wa, rounds=rounds)
    p_test = gbdt_predict(model, m, rounds, Xt)
    if model.endswith('_rel'):
        p_test = np.asarray(p_test) * scale_t
    return np.asarray(p_test, dtype='float64')
