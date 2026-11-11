CB_PARAMS_BASE = dict(CB_PARAMS)
CB_PARAMS_LR03 = dict(CB_PARAMS_BASE, learning_rate=0.03)


def with_cb(params, fn, *args, **kwargs):
    """Temporarily replace CB_PARAMS, read by gbdt_fit at call time."""
    global CB_PARAMS
    saved = CB_PARAMS
    CB_PARAMS = params
    try:
        return fn(*args, **kwargs)
    finally:
        CB_PARAMS = saved


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


# Require complete experiment 11: 3 seeds x 5 folds per model and 3 full refits per tree model.
for model in FINAL_MODELS:
    for s in PREV_SEEDS:
        for k in range(1, N_FOLDS + 1):
            require((PREV_DIR / f'{model}__s{s}__fold{k}.npz').is_file(), f'Experiment 11 is missing {model} seed {s} fold {k}')
        if model != 'tabm':
            require((PREV_DIR / f'{model}__s{s}__full.npz').is_file(), f'Experiment 11 is missing {model} seed {s} full refit')

# Smoke-check CatBoost sur cible DOTS at learning rate 0.03 and near-full TabM.
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
p_val, p_test, _ = with_cb(CB_PARAMS_LR03, fit_fold, MODEL_CATBOOST_DOTS, SEED, pa, ps, pv, test_rows=np.arange(500), rounds=30)
require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'cb_rel smoke check failed')
p_full = with_cb(CB_PARAMS_LR03, fit_full, MODEL_CATBOOST_DOTS, SEED, 30, test_rows=np.arange(500))
require(np.isfinite(p_full).all(), 'cb_rel full-refit smoke check failed')
TABM_OK = True
try:
    p_val, p_test = tabm_fit_predict(SEED, pa, ps, pv, XT_NN.iloc[:500], overrides={'n_epochs': 3})
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'Invalid TabM predictions')
except Exception as exc:
    try:                                               # As in experiment 11, fall back if piecewise-linear embeddings are unavailable.
        TABM_PARAMS = dict(TABM_FALLBACK)
        p_val, p_test = tabm_fit_predict(SEED, pa, ps, pv, XT_NN.iloc[:500], overrides={'n_epochs': 3})
        print('TabM fallback without embeddings: ', repr(exc))
    except Exception as exc2:
        TABM_OK = False
        print('TabM smoke check failed; skip near-full refit: ', repr(exc2))
print('Smoke check complete; near-full TabM: ', 'available' if TABM_OK else 'skipped')
