# ---- Smoke check on 3,000 rows, then 30-iteration full refit for tree models. ----
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for model in FINAL_MODELS:
    attempts = ['main', 'fallback'] if model == 'tabm' else ['main']
    for attempt in attempts:
        if attempt == 'fallback':
            TABM_PARAMS = dict(TABM_FALLBACK)
        started = time.perf_counter()
        try:
            p_val, p_test, _ = fit_fold(model, SEED, pa, ps, pv, test_rows=np.arange(500), rounds=30,
                                        overrides={'n_epochs': 3})
            require(len(p_val) == len(pv) and len(p_test) == 500, 'Prediction length mismatch')
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'Non-finite predictions')
            if model in GBDT and FULL_REFIT:
                p_full = fit_full(model, SEED, 30, test_rows=np.arange(500))
                require(np.isfinite(p_full).all() and len(p_full) == 500, 'Invalid full-refit predictions')
            failed.pop(model, None)
            print(f'Smoke check passed: {model:9s}({attempt})RMSE (short run)  {rmse(y[pv], p_val):.1f}, '
                  f'elapsed  {time.perf_counter() - started:.0f}s')
            break
        except Exception as exc:
            failed[model] = repr(exc)
            print(f'Smoke check failed: {model}({attempt})→ {exc!r}')
RUN_MODELS = [m for m in FINAL_MODELS if m not in failed]
print('Candidates: ', RUN_MODELS, '|skipped: ', failed or 'none')
require(MODEL_CATBOOST_ABSOLUTE in RUN_MODELS, 'CatBoost smoke check failed; check the environment')

# ---- Five folds per seed ----
CONFIG = {'purpose': CONFIG_PURPOSE_SIX_MODEL_ENSEMBLE,
          'seeds': SEEDS, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'refit_factor': REFIT_FACTOR, 'final_models': FINAL_MODELS, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} contains results from another configuration; use a new EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def folds_for(seed):
    ids = np.zeros(n_train, dtype=int)
    for k, (_, valid) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n_train)), start=1):
        ids[valid] = k
    return ids


FOLDS_BY_SEED = {s: folds_for(s) for s in SEEDS}
if SEED in FOLDS_BY_SEED:
    require(np.array_equal(FOLDS_BY_SEED[SEED], FOLD_IDS), 'Seed-42 folds differ from experiments 07-10')


def spec_of(model):
    params = {MODEL_LIGHTGBM: LGB_PARAMS, MODEL_LIGHTGBM_DOTS: LGB_PARAMS, MODEL_XGBOOST: XGB_PARAMS, MODEL_CATBOOST_ABSOLUTE: CB_PARAMS,
              MODEL_CATBOOST_DOTS: CB_PARAMS, 'tabm': TABM_PARAMS}[model]
    return {'model': model, 'columns': list(FRAMES[model][0].columns), 'cat_cols': list(FRAMES[model][2]),
            'params': dict(params), 'target': 'dots_relative_weighted' if model.endswith('_rel') else 'kg'}


def save_result(stem, arrays, info):
    np.savez(OUTPUT_DIR / f'{stem}.npz', **arrays)
    info = {**info, 'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUTPUT_DIR / f'{stem}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


def load_cached(stem, spec):
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{stem} definition changed; use a new EXPERIMENT_ID')
        return saved
    return None


def run_cv(model, seed, fold):
    stem, spec = f'{model}__s{seed}__fold{fold}', spec_of(model)
    saved = load_cached(stem, spec)
    if saved is not None:
        return saved, 'cache'
    ids = FOLDS_BY_SEED[seed]
    fit_idx, valid_idx = np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=seed)
    started = time.perf_counter()
    p_val, p_test, best = fit_fold(model, seed, inner_fit, inner_stop, valid_idx)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} Non-finite predictions')
    info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val, 'test': p_test},
                       {'model': model, 'seed': seed, 'fold': fold, 'spec': spec, 'best_iteration': best,
                        'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started})
    return info, f'{info["seconds"]:.0f}s'


def run_full(model, seed):
    stem, spec = f'{model}__s{seed}__full', spec_of(model)
    saved = load_cached(stem, spec)
    if saved is not None:
        return saved, 'cache'
    bests = [json.loads((OUTPUT_DIR / f'{model}__s{seed}__fold{k}.json').read_text(encoding='utf-8'))['best_iteration']
             for k in range(1, N_FOLDS + 1)]
    rounds = int(round(np.mean(bests) * REFIT_FACTOR))
    started = time.perf_counter()
    p_test = fit_full(model, seed, rounds)
    require(np.isfinite(p_test).all(), f'{stem} Non-finite predictions')
    info = save_result(stem, {'test': p_test}, {'model': model, 'seed': seed, 'spec': spec, 'rounds': rounds,
                                                'cv_best_iterations': bests,
                                                'seconds': time.perf_counter() - started})
    return info, f'{info["seconds"]:.0f}s'


for seed in SEEDS:
    seed_started = time.perf_counter()
    for fold in range(1, N_FOLDS + 1):
        for model in RUN_MODELS:
            info, how = run_cv(model, seed, fold)
            best = info['best_iteration']
            print(f'seed {seed:4d} fold {fold}  {model:9s} RMSE {info["rmse"]:.3f}'
                  + (f'  best iteration {best}' if best else '') + f'  ({how})')
    if FULL_REFIT:
        for model in RUN_MODELS:
            if model in GBDT:
                info, how = run_full(model, seed)
                print(f'seed {seed:4d} full refit {model:9s} iterations {info["rounds"]}  ({how})')
    print(f'==== seed {seed} finished; elapsed {(time.perf_counter() - seed_started) / 60:.1f} min ====')
