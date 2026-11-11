USE_LR03 = False
if RUN_LR_CHECK:
    lr_rows = []
    for fold in range(1, N_FOLDS + 1):
        stem = f'{MODEL_CATBOOST_DOTS_SLOW_LEARNING_RATE}__s{SEED}__fold{fold}'
        spec = {'model': MODEL_CATBOOST_DOTS, 'params': dict(CB_PARAMS_LR03), 'columns': list(FRAMES[MODEL_CATBOOST_DOTS][0].columns)}
        info = load_cached(stem, spec)
        if info is None:
            fit_idx, valid_idx = np.flatnonzero(FOLD_IDS != fold), np.flatnonzero(FOLD_IDS == fold)
            inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 11
            started = time.perf_counter()
            p_val, _, best = with_cb(CB_PARAMS_LR03, fit_fold, MODEL_CATBOOST_DOTS, SEED, inner_fit, inner_stop, valid_idx,
                                     test_rows=np.arange(10))
            info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val},
                               {'spec': spec, 'fold': fold, 'best_iteration': best,
                                'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started})
        base = json.loads((PREV_DIR / f'{MODEL_CATBOOST_DOTS}__s{SEED}__fold{fold}.json').read_text(encoding='utf-8'))
        lr_rows.append({'fold': fold, 'lr_0.05': base['rmse'], 'lr_0.03': info['rmse'],
                        'delta': info['rmse'] - base['rmse'], 'best_iter_0.05': base['best_iteration'],
                        'best_iter_0.03': info['best_iteration'], 'minutes_0.03': info['seconds'] / 60})
        print(f'fold {fold}: lr 0.05 {base["rmse"]:.3f} → lr 0.03 {info["rmse"]:.3f}')
    lr_table = pd.DataFrame(lr_rows).set_index('fold')
    display(lr_table.round(3))
    lr_table.to_csv(OUTPUT_DIR / 'summary_lr_check.csv')
    USE_LR03 = bool(lr_table['delta'].mean() <= -0.02 and (lr_table['delta'] < 0).sum() >= 4)
print('cb_rel full-refit learning rate:', 0.03 if USE_LR03 else 0.05)
