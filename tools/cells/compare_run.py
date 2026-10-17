def run_new(name, fold):
    stem = f'{name}__s{SEED}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        return json.loads(json_path.read_text(encoding='utf-8')), 'cache'
    fit_idx, valid_idx = np.flatnonzero(FOLD_IDS != fold), np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 11
    started = time.perf_counter()
    p_val, p_test = fit_new(name, SEED, inner_fit, inner_stop, valid_idx, XT_BASE)
    p_val, p_test = np.asarray(p_val, dtype='float64'), np.asarray(p_test, dtype='float64')
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val, 'test': p_test},
                       {'model': name, 'fold': fold, 'rmse': rmse(y[valid_idx], p_val),
                        'seconds': time.perf_counter() - started})
    return info, f'{info["seconds"]:.0f}s'


for name in RUN_NEW:
    started = time.perf_counter()
    try:
        for fold in range(1, N_FOLDS + 1):
            info, how = run_new(name, fold)
            print(f'{name:8s} fold {fold}  RMSE {info["rmse"]:.3f}  ({how})')
        print(f'—— {name} 完成，本次运行用时 {(time.perf_counter() - started) / 60:.1f} 分钟')
    except Exception as exc:
        print(f'!! {name} 运行出错，跳过剩下的折：{exc!r}')
