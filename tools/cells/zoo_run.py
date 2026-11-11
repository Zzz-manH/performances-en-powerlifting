CONFIG = {'purpose': CONFIG_PURPOSE_MODEL_COMPARISON,
          'feature_set': FEATURE_SET, 'seed': SEED, 'n_folds': N_FOLDS,
          'inner_stop_fraction': INNER_STOP_FRACTION, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} contains results from another configuration; use a new EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def spec_of(name):
    params = LGB_PARAMS if name == MODEL_LIGHTGBM_DOTS else CB_PARAMS if name == MODEL_CATBOOST_DOTS else NN_PARAMS[name]
    return {'model': name, 'variant': FEATURE_SET, 'columns': list(X_cat.columns), 'params': dict(params),
            'target': 'dots_relative_weighted' if name.endswith('_rel') else 'standardized_kg'}


def run_one(name, fold):
    json_path, npz_path = OUTPUT_DIR / f'{name}__fold{fold}.json', OUTPUT_DIR / f'{name}__fold{fold}.npz'
    spec = spec_of(name)
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{name} fold {fold} definition changed; use a new EXPERIMENT_ID')
        return saved, 'cache'
    fit_idx = np.flatnonzero(FOLD_IDS != fold)
    valid_idx = np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 07
    started = time.perf_counter()
    p_val, p_test, best = fit_model(name, inner_fit, inner_stop, valid_idx)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{name} fold {fold} Non-finite predictions')
    np.savez(npz_path, valid_idx=valid_idx, val=p_val, test=p_test)
    result = {'model': name, 'fold': fold, 'spec': spec, 'best_iteration': best,
              'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started,
              'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result, f'{result["seconds"]:.0f}s'


selected = [n for n in ZOO if (RUN_ONLY is None or n in RUN_ONLY) and n not in failed]
for fold in range(1, N_FOLDS + 1):
    fold_started = time.perf_counter()
    for name in selected:
        result, how = run_one(name, fold)
        best = result['best_iteration']
        print(f'fold {fold}  {name:8s}  RMSE {result["rmse"]:.3f}' + (f'  best iteration {best}' if best else '') + f'  ({how})')
    print(f'—— fold {fold} finished; elapsed {(time.perf_counter() - fold_started) / 60:.1f} min')
