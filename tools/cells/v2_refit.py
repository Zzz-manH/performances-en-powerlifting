def prev_best_iterations(model):
    return [json.loads((PREV_DIR / f'{model}__s{s}__fold{k}.json').read_text(encoding='utf-8'))['best_iteration']
            for s in PREV_SEEDS for k in range(1, N_FOLDS + 1)]


def refit_plan(model):
    """(参数, 轮数)：轮数 = CV 平均最佳轮数 × REFIT_FACTOR。"""
    if model == 'cb_rel' and USE_LR03:
        bests = [json.loads((OUTPUT_DIR / f'cb_rel_lr03__s{SEED}__fold{k}.json').read_text(encoding='utf-8'))
                 ['best_iteration'] for k in range(1, N_FOLDS + 1)]
        return CB_PARAMS_LR03, int(round(np.mean(bests) * REFIT_FACTOR))
    return CB_PARAMS_BASE, int(round(np.mean(prev_best_iterations(model)) * REFIT_FACTOR))


for seed in EXTRA_SEEDS:
    for model in EXTRA_MODELS:
        params, rounds = refit_plan(model)
        stem = f'{model}__s{seed}__full'
        spec = {'model': model, 'rounds': rounds, 'columns': list(FRAMES[model][0].columns),
                'cb_params': dict(params) if model in ('catboost', 'cb_rel') else None}
        info = load_cached(stem, spec)
        if info is None:
            started = time.perf_counter()
            p_test = with_cb(params, fit_full, model, seed, rounds)
            require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
            info = save_result(stem, {'test': p_test}, {'spec': spec, 'seed': seed,
                                                        'seconds': time.perf_counter() - started})
            how = f'{info["seconds"]:.0f}s'
        else:
            how = 'cache'
        print(f'种子 {seed:3d} 全量重训 {model:8s} 轮数 {rounds}  ({how})')

if TABM_OK:
    for seed in TABM_FULL_SEEDS:
        stem = f'tabm__s{seed}__nearfull'
        spec = {'model': 'tabm', 'params': dict(TABM_PARAMS), 'holdout': TABM_HOLDOUT,
                'columns': list(X_NN.columns)}
        info = load_cached(stem, spec)
        if info is None:
            fit_rows, stop_rows = train_test_split(np.arange(n_train), test_size=TABM_HOLDOUT, random_state=seed)
            started = time.perf_counter()
            p_stop, p_test = tabm_fit_predict(seed, fit_rows, stop_rows, stop_rows, XT_NN)
            require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
            info = save_result(stem, {'test': p_test}, {'spec': spec, 'seed': seed,
                                                        'holdout_rmse': rmse(y[stop_rows], p_stop),
                                                        'seconds': time.perf_counter() - started})
        print(f'TabM 近全量 种子 {seed}：早停集 RMSE {info["holdout_rmse"]:.3f}（仅供参考，早停用过它）')
