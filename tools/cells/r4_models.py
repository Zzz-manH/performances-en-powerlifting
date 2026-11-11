def gl_scale(sex_values, equipment_values, bw_values):
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


ALLO = (train['BodyweightKg'].to_numpy(dtype='float64') / 80.0) ** (2 / 3)
ALLO_TEST = (test['BodyweightKg'].to_numpy(dtype='float64') / 80.0) ** (2 / 3)
GL = gl_scale(train['Sex'], train['Equipment'], train['BodyweightKg'])
GL_TEST = gl_scale(test['Sex'], test['Equipment'], test['BodyweightKg'])
require(min(GL.min(), GL_TEST.min()) > 0 and min(ALLO.min(), ALLO_TEST.min()) > 0, 'Invalid performance scale')
SCALE_OF = {MODEL_CATBOOST_DOTS: (SCALE, SCALE_TEST), MODEL_CATBOOST_ALLOMETRIC: (ALLO, ALLO_TEST), MODEL_CATBOOST_IPF_GL: (GL, GL_TEST)}
for extra in TARGET_MODELS:
    FRAMES[extra] = FRAMES[MODEL_CATBOOST_DOTS]
    PATIENCE[extra] = PATIENCE[MODEL_CATBOOST_DOTS]


def fit_matrix(model, Xtr, Xte, a, s, v, seed=SEED, rounds=MAX_ROUNDS, test_rows=None):
    """Fit the supplied features on a, stop on s; return (outer predictions, test predictions, best iteration)."""
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    if model in SCALE_OF:
        scale, scale_test = SCALE_OF[model]
        scale_t = scale_test if test_rows is None else scale_test[test_rows]
        ya, ys = y[a] / scale[a], y[s] / scale[s]
        norm = (scale[a] ** 2).mean()
        wa, ws = scale[a] ** 2 / norm, scale[s] ** 2 / norm
    else:
        ya, ys, wa, ws = y[a], y[s], None, None
    m, best = gbdt_fit(model, seed, Xtr.iloc[a], ya, wa, Xtr.iloc[s], ys, ws, rounds)
    p_val = np.asarray(gbdt_predict(model, m, best, Xtr.iloc[v]), dtype='float64')
    p_test = np.asarray(gbdt_predict(model, m, best, Xt), dtype='float64')
    if model in SCALE_OF:
        p_val, p_test = p_val * scale[v], p_test * scale_t
    return p_val, p_test, best


rng = np.random.default_rng(SEED)
probe = rng.choice(np.flatnonzero(FOLD_IDS != 1), 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
checks = [(m, v) for v in PERFORMANCE_VARIANTS for m in SCREEN_MODELS] + [(m, VARIANT_FINAL_BASELINE) for m in TARGET_MODELS]
for model, variant in checks:
    try:
        Xtr, Xte = performance_feature_frames(variant, model, 1)
        p_val, p_test, _ = fit_matrix(model, Xtr, Xte, pa, ps, pv, rounds=30, test_rows=np.arange(300))
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'Non-finite predictions')
        print(f'Smoke check passed: {model:8s} {variant:3s}({Xtr.shape[1]} columns)RMSE@30 iterations {rmse(y[pv], p_val):.1f}')
    except Exception as exc:
        failed[(model, variant)] = repr(exc)
        print(f'Smoke check failed, skipping: {model} {variant} → {exc!r}')
require(not any(v == VARIANT_FINAL_BASELINE and m in SCREEN_MODELS for (m, v) in failed), 'Échec du contrôle des variables finales ; vérifier l’environnement')
