def mean_best_iterations(model):
    bests = []
    for seed in PREV_SEEDS:
        for k in range(1, N_FOLDS + 1):
            info = cached(f'{model}__s{seed}__fold{k}')
            if info is not None and info.get('best_iteration'):
                bests.append(info['best_iteration'])
    return float(np.mean(bests)) if bests else None


try:                                             # 12 中 cb_rel：学习率 0.03 的最佳轮数约为 0.05 的 1.4 倍
    lr_check = pd.read_csv(FINAL_V2 / 'summary_lr_check.csv')
    LR03_RATIO = float(lr_check['best_iter_0.03'].mean() / lr_check['best_iter_0.05'].mean())
except Exception:
    LR03_RATIO = 1.44
print(f'学习率 0.03 / 0.05 的轮数比：{LR03_RATIO:.2f}')

if OK.get('cb_gl') and mean_best_iterations('cb_gl'):
    gl_rounds = int(round(mean_best_iterations('cb_gl') * LR03_RATIO * REFIT_FACTOR))
    for seed in GL_FULL_SEEDS:
        stem = f'cb_gl__s{seed}__full'
        try:
            info = cached(stem)
            if info is None:
                started = time.perf_counter()
                p_test = with_cb(CB_PARAMS_LR03, fit_full_scaled, 'cb_gl', seed, gl_rounds)
                require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
                info = save_result(stem, {'test': p_test}, {'model': 'cb_gl', 'seed': seed, 'rounds': gl_rounds,
                                                            'learning_rate': 0.03,
                                                            'seconds': time.perf_counter() - started})
            print(f'cb_gl 全量重训 种子 {seed}：轮数 {info["rounds"]}，用时 {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! cb_gl 全量重训 种子 {seed} 出错，跳过：{exc!r}')

if OK.get('hgb'):
    for seed in HGB_FULL_SEEDS:
        stem = f'hgb__s{seed}__nearfull'
        try:
            info = cached(stem)
            if info is None:
                started = time.perf_counter()
                _, p_test, n_iter = fit_hgb(seed, np.arange(n_train), None, XT_BASE)
                info = save_result(stem, {'test': p_test}, {'model': 'hgb', 'seed': seed, 'n_iter': n_iter,
                                                            'seconds': time.perf_counter() - started})
            print(f'HGB 全部数据（内部 10% 早停）种子 {seed}：轮数 {info["n_iter"]}，用时 {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! HGB 全量 种子 {seed} 出错，跳过：{exc!r}')

if OK.get('tabm'):
    for seed in TABM_EXTRA_SEEDS:
        stem = f'tabm__s{seed}__nearfull'
        try:
            info = cached(stem)
            if info is None:
                fit_rows, stop_rows = train_test_split(np.arange(n_train), test_size=TABM_HOLDOUT, random_state=seed)
                started = time.perf_counter()
                p_stop, p_test = tabm_fit_predict(seed, fit_rows, stop_rows, stop_rows, XT_NN)
                require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
                info = save_result(stem, {'test': p_test}, {'model': 'tabm', 'seed': seed,
                                                            'holdout_rmse': rmse(y[stop_rows], p_stop),
                                                            'seconds': time.perf_counter() - started})
            print(f'TabM 95% 重训 种子 {seed}：用时 {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! TabM 种子 {seed} 出错，跳过：{exc!r}')
