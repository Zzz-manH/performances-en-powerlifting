LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=20, verbose=-1)
XGB_PARAMS = dict(learning_rate=0.03, max_depth=8, min_child_weight=10.0, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=1.0, tree_method='hist', device='cuda', enable_categorical=True,
                  max_cat_to_onehot=1, eval_metric='rmse')
CB_PARAMS = dict(learning_rate=0.05, depth=8, l2_leaf_reg=10.0, bootstrap_type='Bernoulli', subsample=0.8,
                 border_count=128)                                   # profondeur 8, régularisation renforcée from experiment 07
PATIENCE = {MODEL_LIGHTGBM: 200, MODEL_XGBOOST: 200, MODEL_CATBOOST_ABSOLUTE: 300}


def fit_predict(model, Xa, ya, Xs, ys, Xv, Xt, cat_cols, rounds=MAX_ROUNDS):
    """Fit on (Xa, ya), stop on (Xs, ys); return (outer predictions, test predictions, best iteration)."""
    if model == MODEL_LIGHTGBM:
        m = lgb.LGBMRegressor(n_estimators=rounds, random_state=SEED, n_jobs=-1, **LGB_PARAMS)
        m.fit(Xa, ya, eval_set=[(Xs, ys)], callbacks=[lgb.early_stopping(PATIENCE[MODEL_LIGHTGBM], verbose=False)])
        best = int(m.best_iteration_ or rounds)
        return m.predict(Xv, num_iteration=best), m.predict(Xt, num_iteration=best), best
    if model == MODEL_XGBOOST:
        m = xgb.XGBRegressor(n_estimators=rounds, random_state=SEED, early_stopping_rounds=PATIENCE[MODEL_XGBOOST],
                             **XGB_PARAMS)
        m.fit(Xa, ya, eval_set=[(Xs, ys)], verbose=False)
        best = int(m.best_iteration) + 1
        return m.predict(Xv, iteration_range=(0, best)), m.predict(Xt, iteration_range=(0, best)), best
    if model == MODEL_CATBOOST_ABSOLUTE:
        m = catboost.CatBoostRegressor(iterations=rounds, loss_function='RMSE', eval_metric='RMSE',
                                       od_type='Iter', od_wait=PATIENCE[MODEL_CATBOOST_ABSOLUTE], random_seed=SEED,
                                       verbose=0, task_type='GPU', devices='0', allow_writing_files=False,
                                       **CB_PARAMS)
        m.fit(catboost.Pool(Xa, ya, cat_features=cat_cols), eval_set=catboost.Pool(Xs, ys, cat_features=cat_cols),
              use_best_model=True)
        best = int(m.get_best_iteration()) + 1
        return m.predict(Xv), m.predict(Xt), best
    raise ValueError(model)


rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for variant in SCREEN:
    for model in models_for(variant):
        Xtr, Xte, cat_cols = history_feature_frames(variant, model)
        try:
            p_val, p_test, _ = fit_predict(model, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                           Xte.iloc[:500], cat_cols, rounds=30)
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'Non-finite predictions')
            require(len(p_val) == len(pv) and len(p_test) == 500, 'Prediction length mismatch')
            print(f'Smoke check passed: {model:8s} {variant:3s}({Xtr.shape[1]} columns)RMSE@30 iterations {rmse(y[pv], p_val):.1f}')
        except Exception as exc:
            failed[(model, variant)] = repr(exc)
            print(f'Smoke check failed, skipping: {model} {variant} → {exc!r}')
require(not any(v == VARIANT_CONTEXT_BASELINE for (_, v) in failed), 'Échec du contrôle de la référence avec contexte ; vérifier l’environnement')
