def run_one(model, variant, fold):
    stem = f'{model}__{variant}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        return json.loads(json_path.read_text(encoding='utf-8')), 'cache'
    Xtr, Xte = frames_aug(variant, model, fold)
    fit_idx, valid_idx = np.flatnonzero(FOLD_IDS != fold), np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 07–13
    started = time.perf_counter()
    p_val, p_test, best = fit_matrix(model, Xtr, Xte, inner_fit, inner_stop, valid_idx)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    np.savez(npz_path, valid_idx=valid_idx, val=p_val, test=p_test)
    info = {'model': model, 'variant': variant, 'fold': fold, 'blocks': VARIANTS4[variant],
            'n_features': int(Xtr.shape[1]), 'best_iteration': best, 'rmse': rmse(y[valid_idx], p_val),
            'seconds': time.perf_counter() - started, 'gpu': GPU_NAME,
            'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info, f'{info["seconds"]:.0f}s'


def run_variants(variants, models):
    for fold in range(1, N_FOLDS + 1):
        fold_started = time.perf_counter()
        for model in models:
            for variant in variants:
                if (model, variant) in failed:
                    continue
                info, how = run_one(model, variant, fold)
                print(f'fold {fold}  {model:8s} {variant:4s}  RMSE {info["rmse"]:.3f}  '
                      f'最佳轮数 {info["best_iteration"]:5d}  ({how})')
        print(f'—— fold {fold} 完成，本次运行用时 {(time.perf_counter() - fold_started) / 60:.1f} 分钟')


run_variants(list(VARIANTS4), SCREEN_MODELS)
