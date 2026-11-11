if RUN_CLEANING_TEST:
    ratio = y / train['BodyweightKg'].to_numpy(dtype='float64')
    SUSPICIOUS = (np.where(train['Sex'].to_numpy() == 'M', ratio < 2.5, ratio < 1.8)
                  | (train['Age'].to_numpy(dtype='float64') < 5))
    print(f'Flagged rows: {int(SUSPICIOUS.sum())}({SUSPICIOUS.mean():.2%})')
    clean_rows = []
    for model in [m for m in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_ABSOLUTE] if m in RUN_MODELS]:
        if not all((OUTPUT_DIR / f'{model}__s{SEED}__fold{k}.npz').is_file() for k in range(1, N_FOLDS + 1)):
            print(f'{model}: seed {SEED} has incomplete CV; skip this comparison')
            continue
        base_folds, clean_folds, base_normal, clean_normal = [], [], [], []
        for fold in range(1, N_FOLDS + 1):
            stem, spec = f'clean_{model}__s{SEED}__fold{fold}', {**spec_of(model), 'cleaning': 'drop_suspicious'}
            saved = load_cached(stem, spec)
            ids = FOLDS_BY_SEED[SEED]
            fit_idx, valid_idx = np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
            if saved is None:
                inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)
                inner_fit, inner_stop = inner_fit[~SUSPICIOUS[inner_fit]], inner_stop[~SUSPICIOUS[inner_stop]]
                started = time.perf_counter()
                p_val, _, best = fit_fold(model, SEED, inner_fit, inner_stop, valid_idx, test_rows=np.arange(10))
                save_result(stem, {'valid_idx': valid_idx, 'val': p_val},
                            {'model': model, 'fold': fold, 'spec': spec, 'best_iteration': best,
                             'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started})
            clean_val = np.load(OUTPUT_DIR / f'{stem}.npz')['val']
            base_val = np.load(OUTPUT_DIR / f'{model}__s{SEED}__fold{fold}.npz')['val']
            normal = ~SUSPICIOUS[valid_idx]
            base_folds.append(rmse(y[valid_idx], base_val))
            clean_folds.append(rmse(y[valid_idx], clean_val))
            base_normal.append(rmse(y[valid_idx][normal], base_val[normal]))
            clean_normal.append(rmse(y[valid_idx][normal], clean_val[normal]))
        delta = np.array(clean_folds) - np.array(base_folds)
        clean_rows.append({'model': model, 'base_all_rows': np.mean(base_folds), 'clean_all_rows': np.mean(clean_folds),
                           'delta_all_rows': delta.mean(), 'folds_better': int((delta < 0).sum()),
                           'base_normal_rows': np.mean(base_normal), 'clean_normal_rows': np.mean(clean_normal)})
    clean_table = pd.DataFrame(clean_rows).set_index('model')
    display(clean_table.round(3))
    clean_table.to_csv(OUTPUT_DIR / 'summary_cleaning_test.csv')
    print('delta_all_rows < 0 favors removal on all validation rows; normal_rows excludes flagged validation rows.')
print('Done.')
