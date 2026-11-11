from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

X_BASE, XT_BASE, CATS = FRAMES[MODEL_LIGHTGBM]            # Référence avec historique + Carrière approchée du sportif + Expérience du plateau; category dtype, as for TabM
NUMS = [c for c in X_BASE.columns if c not in CATS]
PYTABKIT_CLASSES = {'mlp_plr': 'MLP_PLR_D_Regressor', 'resnet': 'Resnet_RTDL_D_Regressor',
                    'ftt': 'FTT_D_Regressor', 'tabr': 'TabR_S_D_Regressor'}


def numeric_with_codes(enc, X, fill=None):
    num = X[NUMS].to_numpy(dtype='float64')
    if fill is not None:
        num = np.where(np.isnan(num), fill, num)
    return np.hstack([num, enc.transform(X[CATS])])


def fit_ridge(seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), NUMS),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), CATS)])
    model = make_pipeline(pre, Ridge(alpha=10.0)).fit(X_BASE.iloc[rows], y[rows])
    return model.predict(X_BASE.iloc[v]), model.predict(Xt)


def fit_hgb(seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan,
                         max_categories=250).fit(X_BASE.iloc[rows][CATS])
    mask = [False] * len(NUMS) + [True] * len(CATS)
    model = HistGradientBoostingRegressor(
        categorical_features=mask, learning_rate=0.05, max_iter=20 if small else 3000, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=100, random_state=seed)
    model.fit(numeric_with_codes(enc, X_BASE.iloc[rows]), y[rows])           # 10% internal early-stopping split
    return model.predict(numeric_with_codes(enc, X_BASE.iloc[v])), model.predict(numeric_with_codes(enc, Xt))


def fit_forest(kind, seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_BASE.iloc[rows][CATS])
    common = dict(n_estimators=10 if small else 200, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    model = (RandomForestRegressor(max_features=0.3, max_samples=0.5, **common) if kind == MODEL_RANDOM_FOREST
             else ExtraTreesRegressor(max_features=0.5, **common))
    model.fit(numeric_with_codes(enc, X_BASE.iloc[rows], fill=-999.0), y[rows])
    return (model.predict(numeric_with_codes(enc, X_BASE.iloc[v], fill=-999.0)),
            model.predict(numeric_with_codes(enc, Xt, fill=-999.0)))


def fit_ebm(seed, a, s, v, Xt, small=False):
    from interpret.glassbox import ExplainableBoostingRegressor
    rows = np.concatenate([a, s])
    model = ExplainableBoostingRegressor(interactions=0 if small else 5, outer_bags=1 if small else 2,
                                         max_rounds=50 if small else 5000, n_jobs=-1, random_state=seed)
    model.fit(X_BASE.iloc[rows], y[rows])                                   # Internal validation split for early stopping
    return np.asarray(model.predict(X_BASE.iloc[v])), np.asarray(model.predict(Xt))


def make_tabpfn():
    from tabpfn import TabPFNRegressor
    if TABPFN_TOKEN:
        return TabPFNRegressor(device='cuda', ignore_pretraining_limits=True)
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2)


def fit_tabpfn(seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    context = np.random.default_rng(seed).choice(rows, size=min(500 if small else TABPFN_CONTEXT_ROWS, len(rows)),
                                                 replace=False)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                         encoded_missing_value=np.nan).fit(X_BASE.iloc[context][CATS])
    model = make_tabpfn()
    model.fit(numeric_with_codes(enc, X_BASE.iloc[context]), y[context])
    p_val = np.asarray(model.predict(numeric_with_codes(enc, X_BASE.iloc[v])), dtype='float64')
    p_test = np.asarray(model.predict(numeric_with_codes(enc, Xt)), dtype='float64')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test


def fit_pytabkit(name, seed, a, s, v, Xt, small=False):
    if name == 'tabr':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'faiss-cpu'], check=False)
    import pytabkit
    cls = getattr(pytabkit, PYTABKIT_CLASSES[name])
    params = dict(device='cuda', random_state=seed, n_cv=1, n_refit=0, verbosity=0)
    if small:
        params['max_epochs'] = 2
    unknown = set(params) - set(inspect.signature(cls.__init__).parameters)
    require(not unknown, f'{name} does not support parameters {sorted(unknown)}')
    Za, Zs, Zv, Zt = nn_frames(X_NN.iloc[a], X_NN.iloc[s], X_NN.iloc[v], Xt)    # Same imputation as TabM
    mu, sd = y[a].mean(), y[a].std()
    m = cls(**params)
    m.fit(Za, (y[a] - mu) / sd, Zs, (y[s] - mu) / sd, cat_col_names=[c for c in Za.columns if c in CATS_NN])
    p_val = np.asarray(m.predict(Zv), dtype='float64').reshape(-1) * sd + mu
    p_test = np.asarray(m.predict(Zt), dtype='float64').reshape(-1) * sd + mu
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test


def fit_new(name, seed, a, s, v, Xt, small=False):
    if name == 'ridge':
        return fit_ridge(seed, a, s, v, Xt, small)
    if name == MODEL_HISTOGRAM_GRADIENT_BOOSTING:
        return fit_hgb(seed, a, s, v, Xt, small)
    if name in (MODEL_RANDOM_FOREST, MODEL_EXTRA_TREES):
        return fit_forest(name, seed, a, s, v, Xt, small)
    if name == 'ebm':
        return fit_ebm(seed, a, s, v, Xt, small)
    if name == 'tabpfn':
        return fit_tabpfn(seed, a, s, v, Xt, small)
    return fit_pytabkit(name, seed, a, s, v, Xt, small)


def save_result(stem, arrays, info):
    np.savez(OUTPUT_DIR / f'{stem}.npz', **arrays)
    info = {**info, 'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUTPUT_DIR / f'{stem}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


# Smoke check on 3,000 rows with a small budget; skip failed candidates.
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for name in COMPARE + OPTIONAL:
    started = time.perf_counter()
    try:
        p_val, p_test = fit_new(name, SEED, pa, ps, pv, XT_BASE.iloc[:300], small=True)
        p_val, p_test = np.asarray(p_val, dtype='float64'), np.asarray(p_test, dtype='float64')
        require(len(p_val) == len(pv) and len(p_test) == 300, 'Prediction length mismatch')
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'Non-finite predictions')
        print(f'Smoke check passed: {name:8s} RMSE (small budget) {rmse(y[pv], p_val):.1f}, elapsed  {time.perf_counter() - started:.0f}s')
    except Exception as exc:
        failed[name] = repr(exc)
        print(f'Smoke check failed, skipping: {name} → {exc!r}')
RUN_NEW = [n for n in COMPARE + OPTIONAL if n not in failed]
print('Candidates: ', RUN_NEW)
