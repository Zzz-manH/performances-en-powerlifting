def adopt_previous(stem_prev_dir, prev_stem, new_stem):
    """Copy saved seed-42 folds here with consistent filenames."""
    for k in range(1, N_FOLDS + 1):
        src = stem_prev_dir / f'{prev_stem}__fold{k}.npz'
        if cached(f'{new_stem}__fold{k}') is None and src.is_file():
            saved = np.load(src)
            src_json = src.with_suffix('.json')
            info = json.loads(src_json.read_text(encoding='utf-8')) if src_json.is_file() else {}
            save_result(f'{new_stem}__fold{k}', {'valid_idx': saved['valid_idx'], 'val': saved['val'],
                                                 'test': saved['test']},
                        {'source': str(src), 'rmse': rmse(y[saved['valid_idx']], saved['val']),
                         'best_iteration': info.get('best_iteration')})


adopt_previous(PERFORMANCE_ABLATION_DIR, CACHE_CATBOOST_IPF_GL_BASELINE_STEM, CACHE_CATBOOST_IPF_GL_SEED42_STEM)
adopt_previous(COMPARE_DIR, 'hgb__s42', 'hgb__s42')


def run_cv(model, seed):
    ids = folds_for(seed)
    for k in range(1, N_FOLDS + 1):
        stem = f'{model}__s{seed}__fold{k}'
        info = cached(stem)
        if info is None:
            fit_idx, valid_idx = np.flatnonzero(ids != k), np.flatnonzero(ids == k)
            inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=seed)
            started = time.perf_counter()
            if model == MODEL_CATBOOST_IPF_GL:
                p_val, p_test, best = fit_matrix(MODEL_CATBOOST_IPF_GL, inner_fit, inner_stop, valid_idx, seed)
            else:
                p_val, p_test, best = fit_hgb(seed, fit_idx, valid_idx, XT_BASE)
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} Non-finite predictions')
            info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val, 'test': p_test},
                               {'model': model, 'seed': seed, 'fold': k, 'rmse': rmse(y[valid_idx], p_val),
                                'best_iteration': best, 'seconds': time.perf_counter() - started})
            how = f'{info["seconds"]:.0f}s'
        else:
            how = 'cache'
        print(f'{model:6s} seed {seed:4d} fold {k}  RMSE {info["rmse"]:.3f}  ({how})')


for model in [MODEL_HISTOGRAM_GRADIENT_BOOSTING, MODEL_CATBOOST_IPF_GL]:                       # Run the inexpensive CPU HGB first.
    if not OK.get(model):
        continue
    for seed in PREV_SEEDS:
        try:
            run_cv(model, seed)
        except Exception as exc:
            print(f'!! {model} seed {seed} failed, skipping: {exc!r}')
