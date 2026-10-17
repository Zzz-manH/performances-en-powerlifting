CONFIG = {'purpose': 'feature round 1 (P/T/M blocks); inner 10% early stop, score early-stopped model on outer fold',
          'seed': SEED, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'variants': VARIANTS, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def spec_of(model, variant, key):
    Xtr, _ = frames_for(variant, model)
    params = {'lgb': LGB_PARAMS, 'xgb': XGB_PARAMS, 'catboost': CB_PARAMS[key]}[model]
    return {'model': model, 'variant': variant, 'params_key': key, 'columns': list(Xtr.columns),
            'params': dict(params), 'patience': PATIENCE[model]}


def run_one(model, variant, key, fold):
    stem = f'{tag_of(model, key)}__{variant}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    spec = spec_of(model, variant, key)
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{stem} 的定义已改变；请改 EXPERIMENT_ID 或变体名')
        return saved, 'cache'
    Xtr, Xte = frames_for(variant, model)
    fit_idx = np.flatnonzero(FOLD_IDS != fold)
    valid_idx = np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)
    started = time.perf_counter()
    p_val, p_test, best = fit_predict(model, key, Xtr.iloc[inner_fit], y[inner_fit], Xtr.iloc[inner_stop],
                                      y[inner_stop], Xtr.iloc[valid_idx], Xte)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    np.savez(npz_path, valid_idx=valid_idx, val=np.asarray(p_val, dtype='float64'),
             test=np.asarray(p_test, dtype='float64'))
    result = {'tag': tag_of(model, key), 'variant': variant, 'fold': fold, 'spec': spec,
              'n_features': int(Xtr.shape[1]), 'best_iteration': best,
              'hit_limit': bool(best >= MAX_ROUNDS - PATIENCE[model]),
              'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started,
              'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result, f'{result["seconds"]:.0f}s'


selected = [v for v in VARIANTS if RUN_ONLY is None or v in RUN_ONLY]
for fold in range(1, N_FOLDS + 1):
    fold_started = time.perf_counter()
    for model in MODELS:
        for variant in selected:
            if (model, variant, 'C1') in failed:
                continue
            result, how = run_one(model, variant, 'C1', fold)
            print(f'fold {fold}  {tag_of(model, "C1"):12s} {variant:4s}  RMSE {result["rmse"]:.3f}  '
                  f'最佳轮数 {result["best_iteration"]:5d}  ({how})')
    print(f'—— fold {fold} 完成，本次运行用时 {(time.perf_counter() - fold_started) / 60:.1f} 分钟')
