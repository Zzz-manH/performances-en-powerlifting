def mean_best_iterations(model):
    bests = []
    for seed in PREV_SEEDS:
        for k in range(1, N_FOLDS + 1):
            info = cached(f'{model}__s{seed}__fold{k}')
            if info is not None and info.get('best_iteration'):
                bests.append(info['best_iteration'])
    return float(np.mean(bests)) if bests else None


try:                                             # Experiment 12: CatBoost sur cible DOTS at lr=0.03 used about 1.4 times as many iterations as lr=0.05.
    lr_check = pd.read_csv(EXPANDED_SEED_ENSEMBLE_DIR / 'summary_lr_check.csv')
    LR03_RATIO = float(lr_check['best_iter_0.03'].mean() / lr_check['best_iter_0.05'].mean())
except Exception:
    LR03_RATIO = 1.44
print(f'Iteration ratio for lr 0.03 / 0.05: {LR03_RATIO:.2f}')

if OK.get(MODEL_CATBOOST_IPF_GL) and mean_best_iterations(MODEL_CATBOOST_IPF_GL):
    gl_rounds = int(round(mean_best_iterations(MODEL_CATBOOST_IPF_GL) * LR03_RATIO * REFIT_FACTOR))
    for seed in GL_FULL_SEEDS:
        stem = f'{MODEL_CATBOOST_IPF_GL}__s{seed}__full'
        try:
            info = cached(stem)
            if info is None:
                started = time.perf_counter()
                p_test = with_cb(CB_PARAMS_LR03, fit_full_scaled, MODEL_CATBOOST_IPF_GL, seed, gl_rounds)
                require(np.isfinite(p_test).all(), f'{stem} Non-finite predictions')
                info = save_result(stem, {'test': p_test}, {'model': MODEL_CATBOOST_IPF_GL, 'seed': seed, 'rounds': gl_rounds,
                                                            'learning_rate': 0.03,
                                                            'seconds': time.perf_counter() - started})
            print(f'cb_gl full refit seed {seed}: iterations {info["rounds"]}, elapsed  {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! cb_gl full refit seed {seed} failed, skipping: {exc!r}')

if OK.get(MODEL_HISTOGRAM_GRADIENT_BOOSTING):
    for seed in HGB_FULL_SEEDS:
        stem = f'hgb__s{seed}__nearfull'
        try:
            info = cached(stem)
            if info is None:
                started = time.perf_counter()
                _, p_test, n_iter = fit_hgb(seed, np.arange(n_train), None, XT_BASE)
                info = save_result(stem, {'test': p_test}, {'model': MODEL_HISTOGRAM_GRADIENT_BOOSTING, 'seed': seed, 'n_iter': n_iter,
                                                            'seconds': time.perf_counter() - started})
            print(f'HGB (90% fitting, 10% early stopping)seed {seed}: iterations {info["n_iter"]}, elapsed  {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! HGB refit seed {seed} failed, skipping: {exc!r}')

if OK.get('tabm'):
    for seed in TABM_EXTRA_SEEDS:
        stem = f'tabm__s{seed}__nearfull'
        try:
            info = cached(stem)
            if info is None:
                fit_rows, stop_rows = train_test_split(np.arange(n_train), test_size=TABM_HOLDOUT, random_state=seed)
                started = time.perf_counter()
                p_stop, p_test = tabm_fit_predict(seed, fit_rows, stop_rows, stop_rows, XT_NN)
                require(np.isfinite(p_test).all(), f'{stem} Non-finite predictions')
                info = save_result(stem, {'test': p_test}, {'model': 'tabm', 'seed': seed,
                                                            'holdout_rmse': rmse(y[stop_rows], p_stop),
                                                            'seconds': time.perf_counter() - started})
            print(f'TabM 95% refit seed {seed}: elapsed  {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! TabM seed {seed} failed, skipping: {exc!r}')
