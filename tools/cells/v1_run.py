# ---- 预检：每个模型在 3,000 行上训练几轮；树模型再在全部行上做 30 轮的全量重训 ----
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
            require(len(p_val) == len(pv) and len(p_test) == 500, '预测长度不符')
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
            if model in GBDT and FULL_REFIT:
                p_full = fit_full(model, SEED, 30, test_rows=np.arange(500))
                require(np.isfinite(p_full).all() and len(p_full) == 500, '全量重训预测无效')
            failed.pop(model, None)
            print(f'预检通过：{model:9s}（{attempt}）RMSE@少量轮 {rmse(y[pv], p_val):.1f}，'
                  f'用时 {time.perf_counter() - started:.0f}s')
            break
        except Exception as exc:
            failed[model] = repr(exc)
            print(f'预检失败：{model}（{attempt}）→ {exc!r}')
RUN_MODELS = [m for m in FINAL_MODELS if m not in failed]
print('将运行：', RUN_MODELS, '｜跳过：', failed or '无')
require('catboost' in RUN_MODELS, 'CatBoost 预检失败，请检查环境')

# ---- 五折 × 种子 ----
CONFIG = {'purpose': 'final ensemble: ALL3 per-model features, 5-fold x seeds, inner 10% early stop, full refit',
          'seeds': SEEDS, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'refit_factor': REFIT_FACTOR, 'final_models': FINAL_MODELS, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def folds_for(seed):
    ids = np.zeros(n_train, dtype=int)
    for k, (_, valid) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n_train)), start=1):
        ids[valid] = k
    return ids


FOLDS_BY_SEED = {s: folds_for(s) for s in SEEDS}
if SEED in FOLDS_BY_SEED:
    require(np.array_equal(FOLDS_BY_SEED[SEED], FOLD_IDS), '种子 42 的五折与 07–10 不一致')


def spec_of(model):
    params = {'lgb': LGB_PARAMS, 'lgb_rel': LGB_PARAMS, 'xgb': XGB_PARAMS, 'catboost': CB_PARAMS,
              'cb_rel': CB_PARAMS, 'tabm': TABM_PARAMS}[model]
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
        require(saved['spec'] == spec, f'{stem} 的定义已改变；请改 EXPERIMENT_ID')
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
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
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
    require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
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
            print(f'种子 {seed:4d} fold {fold}  {model:9s} RMSE {info["rmse"]:.3f}'
                  + (f'  最佳轮数 {best}' if best else '') + f'  ({how})')
    if FULL_REFIT:
        for model in RUN_MODELS:
            if model in GBDT:
                info, how = run_full(model, seed)
                print(f'种子 {seed:4d} 全量重训 {model:9s} 轮数 {info["rounds"]}  ({how})')
    print(f'==== 种子 {seed} 完成，本次运行用时 {(time.perf_counter() - seed_started) / 60:.1f} 分钟 ====')
