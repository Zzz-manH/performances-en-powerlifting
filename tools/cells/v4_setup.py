from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


def gl_scale(sex_values, equipment_values, bw_values):
    """IPF GL denominator / 100: total = GL score * gl_scale. Raw/Wraps use unequipped coefficients; others use equipped coefficients."""
    sex = np.asarray(sex_values).astype(str)
    equipped = ~np.isin(np.asarray(equipment_values).astype(str), ['Raw', 'Wraps'])
    bw = np.maximum(np.asarray(bw_values, dtype='float64'), 30.0)
    coef = {('M', False): (1199.72839, 1025.18162, 0.00921), ('M', True): (1236.25115, 1449.21864, 0.01644),
            ('F', False): (610.32796, 1045.59282, 0.03048), ('F', True): (758.63878, 949.31382, 0.02435)}

    def curve(s, e):
        a, b, c = coef[(s, e)]
        return a - b * np.exp(-c * bw)

    male = np.where(equipped, curve('M', True), curve('M', False))
    female = np.where(equipped, curve('F', True), curve('F', False))
    return np.where(sex == 'M', male, np.where(sex == 'F', female, 0.5 * (male + female))) / 100.0


GL = gl_scale(train['Sex'], train['Equipment'], train['BodyweightKg'])
GL_TEST = gl_scale(test['Sex'], test['Equipment'], test['BodyweightKg'])
require(min(GL.min(), GL_TEST.min()) > 0, 'Invalid GL scale')
SCALE_OF = {MODEL_CATBOOST_DOTS: (SCALE, SCALE_TEST), MODEL_CATBOOST_IPF_GL: (GL, GL_TEST)}
FRAMES[MODEL_CATBOOST_IPF_GL] = FRAMES[MODEL_CATBOOST_DOTS]                    # As in experiment 14: CatBoost Variables finales par modèle features.
PATIENCE[MODEL_CATBOOST_IPF_GL] = PATIENCE[MODEL_CATBOOST_DOTS]
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


def fit_matrix(model, a, s, v, seed, rounds=MAX_ROUNDS, test_rows=None):
    """As in experiment 14: relative target weighted by scale squared; return (outer predictions, test predictions, best iteration)."""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale, scale_test = SCALE_OF[model]
    scale_t = scale_test if test_rows is None else scale_test[test_rows]
    norm = (scale[a] ** 2).mean()
    m, best = gbdt_fit(model, seed, Xtr.iloc[a], y[a] / scale[a], scale[a] ** 2 / norm,
                       Xtr.iloc[s], y[s] / scale[s], scale[s] ** 2 / norm, rounds)
    p_val = np.asarray(gbdt_predict(model, m, best, Xtr.iloc[v]), dtype='float64') * scale[v]
    p_test = np.asarray(gbdt_predict(model, m, best, Xt), dtype='float64') * scale_t
    return p_val, p_test, best


def fit_full_scaled(model, seed, rounds, test_rows=None):
    """Fit the relative target on all training rows for fixed iterations; return test predictions."""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale, scale_test = SCALE_OF[model]
    scale_t = scale_test if test_rows is None else scale_test[test_rows]
    m, _ = gbdt_fit(model, seed, Xtr, y / scale, scale ** 2 / (scale ** 2).mean(), rounds=rounds)
    return np.asarray(gbdt_predict(model, m, rounds, Xt), dtype='float64') * scale_t


X_BASE, XT_BASE, CATS = FRAMES[MODEL_LIGHTGBM]                # As in experiment 13: LightGBM / TabM Variables finales par modèle features.
NUMS = [c for c in X_BASE.columns if c not in CATS]


def numeric_with_codes(enc, X):
    return np.hstack([X[NUMS].to_numpy(dtype='float64'), enc.transform(X[CATS])])


def fit_hgb(seed, rows, v, Xt, small=False):
    """As in experiment 13: fit rows with 10% internal early stopping; return (predictions for v or None, test predictions, iterations)."""
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan,
                         max_categories=250).fit(X_BASE.iloc[rows][CATS])
    model = HistGradientBoostingRegressor(
        categorical_features=[False] * len(NUMS) + [True] * len(CATS), learning_rate=0.05,
        max_iter=20 if small else 3000, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=100, random_state=seed)
    model.fit(numeric_with_codes(enc, X_BASE.iloc[rows]), y[rows])
    p_val = None if v is None else np.asarray(model.predict(numeric_with_codes(enc, X_BASE.iloc[v])), dtype='float64')
    return p_val, np.asarray(model.predict(numeric_with_codes(enc, Xt)), dtype='float64'), int(model.n_iter_)


def folds_for(seed):
    ids = np.zeros(n_train, dtype=int)
    for k, (_, valid) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n_train)), start=1):
        ids[valid] = k
    return ids


def save_result(stem, arrays, info):
    np.savez(OUTPUT_DIR / f'{stem}.npz', **arrays)
    info = {**info, 'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUTPUT_DIR / f'{stem}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


def cached(stem):
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        return json.loads(json_path.read_text(encoding='utf-8'))
    return None


require(np.array_equal(folds_for(SEED), FOLD_IDS), 'Seed-42 folds differ from the saved split')
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
OK = {}
try:
    p_val, p_test, _ = fit_matrix(MODEL_CATBOOST_IPF_GL, pa, ps, pv, SEED, rounds=30, test_rows=np.arange(300))
    p_full = with_cb(CB_PARAMS_LR03, fit_full_scaled, MODEL_CATBOOST_IPF_GL, SEED, 30, test_rows=np.arange(300))
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all() and np.isfinite(p_full).all(), 'Invalid predictions')
    OK[MODEL_CATBOOST_IPF_GL] = True
except Exception as exc:
    OK[MODEL_CATBOOST_IPF_GL] = False
    print('cb_gl smoke check failed; reuse only seed 42 from experiment 14: ', repr(exc))
try:
    p_val, p_test, _ = fit_hgb(SEED, np.concatenate([pa, ps]), pv, XT_BASE.iloc[:300], small=True)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'Invalid predictions')
    OK[MODEL_HISTOGRAM_GRADIENT_BOOSTING] = True
except Exception as exc:
    OK[MODEL_HISTOGRAM_GRADIENT_BOOSTING] = False
    print('HGB smoke check failed; reuse only seed 42 from experiment 13: ', repr(exc))
try:
    p_val, p_test = tabm_fit_predict(SEED, pa, ps, pv, XT_NN.iloc[:300], overrides={'n_epochs': 3})
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'Invalid predictions')
    OK['tabm'] = True
except Exception as exc:
    OK['tabm'] = False
    print('TabM smoke check failed; reuse the three 95% refits from experiment 12: ', repr(exc))
print('Smoke check: ', OK)
