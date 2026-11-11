CONFIG = {'purpose': CONFIG_PURPOSE_CAREER_ABLATION,
          'seed': SEED, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'r2_blocks': HISTORY_BASE_BLOCKS, 'extras': MODEL_SPECIFIC_BLOCKS, 'screen': CAREER_VARIANTS, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} contains results from another configuration; use a new EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def spec_of(model, variant):
    Xtr, _, cat_cols = career_feature_frames(variant, model)
    params = {MODEL_LIGHTGBM: LGB_PARAMS, MODEL_XGBOOST: XGB_PARAMS, MODEL_CATBOOST_ABSOLUTE: CB_PARAMS}[model]
    return {'model': model, 'variant': variant, 'blocks': feature_blocks_for_model(variant, model),
            'columns': list(Xtr.columns), 'cat_cols': list(cat_cols), 'params': dict(params),
            'patience': PATIENCE[model]}


def run_one(model, variant, fold):
    stem = f'{model}__{variant}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    spec = spec_of(model, variant)
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{stem} definition changed; use a new EXPERIMENT_ID or variant name')
        return saved, 'cache'
    Xtr, Xte, cat_cols = career_feature_frames(variant, model)
    fit_idx = np.flatnonzero(FOLD_IDS != fold)
    valid_idx = np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 07
    started = time.perf_counter()
    p_val, p_test, best = fit_predict(model, Xtr.iloc[inner_fit], y[inner_fit], Xtr.iloc[inner_stop],
                                      y[inner_stop], Xtr.iloc[valid_idx], Xte, cat_cols)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} Non-finite predictions')
    np.savez(npz_path, valid_idx=valid_idx, val=np.asarray(p_val, dtype='float64'),
             test=np.asarray(p_test, dtype='float64'))
    result = {'model': model, 'variant': variant, 'fold': fold, 'spec': spec, 'n_features': int(Xtr.shape[1]),
              'best_iteration': best, 'hit_limit': bool(best >= MAX_ROUNDS - PATIENCE[model]),
              'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started,
              'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result, f'{result["seconds"]:.0f}s'


def run_variants(variants):
    for fold in range(1, N_FOLDS + 1):
        fold_started = time.perf_counter()
        for model in MODELS:
            for variant in variants:
                if not runs(model, variant) or (model, variant) in failed:
                    continue
                result, how = run_one(model, variant, fold)
                print(f'fold {fold}  {model:8s} {variant:4s}  RMSE {result["rmse"]:.3f}  '
                      f'best iteration {result["best_iteration"]:5d}  ({how})')
        print(f'—— fold {fold} finished; elapsed {(time.perf_counter() - fold_started) / 60:.1f} min')


run_variants(SCREEN3)
