require(FEATURE_SET in VARIANTS, f'未知特征变体：{FEATURE_SET}；可选 {list(VARIANTS)}')
X_cat, Xt_cat = frames_for(FEATURE_SET, 'lgb')        # 类别为 category：LightGBM、神经网络
X_str, Xt_str = frames_for(FEATURE_SET, 'catboost')   # 类别为字符串：CatBoost
NUM_COLS = [c for c in X_cat.columns if c not in CAT]
NAN_COLS = [c for c in NUM_COLS if X_cat[c].isna().any() or Xt_cat[c].isna().any()]
print(f'特征变体 {FEATURE_SET}：{X_cat.shape[1]} 列 = {len(NUM_COLS)} 数值（{len(NAN_COLS)} 列有缺失）+ {len(CAT)} 类别')

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=20, verbose=-1)                   # = v2 / 07
CB_PARAMS = dict(learning_rate=0.05, depth=8, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0,
                 border_count=128)                                                    # = v2 / 07 的 C1
NN_COMMON = dict(device='cuda', random_state=SEED, n_cv=1, n_refit=0, verbosity=0)
NN_PARAMS = {'realmlp': dict(NN_COMMON), 'tabm': dict(NN_COMMON, num_emb_type='pwl')}
NN_FALLBACK = {'tabm': dict(NN_COMMON)}         # 预检失败时去掉分段线性数值嵌入


def dots_scale(frame):
    """DOTS 系数的分母 / 500：TotalKg = DOTS 分 × scale。体重截断到公式的有效范围；Mx 取男女平均。"""
    sex = frame['Sex'].astype(str).to_numpy()
    bw = frame['BodyweightKg'].to_numpy(dtype='float64')
    m, f = np.clip(bw, 40, 210), np.clip(bw, 40, 150)
    poly_m = -1.0930e-6 * m**4 + 7.391293e-4 * m**3 - 0.1918759221 * m**2 + 24.0900756 * m - 307.75076
    poly_f = -1.0706e-6 * f**4 + 5.158568e-4 * f**3 - 0.1126655495 * f**2 + 13.6175032 * f - 57.96288
    return np.where(sex == 'M', poly_m, np.where(sex == 'F', poly_f, 0.5 * (poly_m + poly_f))) / 500.0


SCALE, SCALE_TEST = dots_scale(train), dots_scale(test)
require(np.isfinite(SCALE).all() and np.isfinite(SCALE_TEST).all() and min(SCALE.min(), SCALE_TEST.min()) > 0.3,
        'DOTS 系数异常')
relative = y / SCALE
print(f'变异系数：kg 目标 {y.std() / y.mean():.3f} → 相对力量 {relative.std() / relative.mean():.3f}')


def nn_frames(*frames):
    """神经网络输入：数值缺失用第一个表（外层训练部分）的中位数填补并加缺失标记；类别保持 category。"""
    median = frames[0][NUM_COLS].median()
    out = []
    for frame in frames:
        z = frame.copy()
        for c in NAN_COLS:
            z[f'{c}_isna'] = z[c].isna().astype('float64')
        z[NUM_COLS] = z[NUM_COLS].fillna(median).fillna(0.0)
        out.append(z.reset_index(drop=True))
    return out


def fit_model(name, a, s, v, test_rows=None, rounds=MAX_ROUNDS, nn_overrides=None):
    """a / s / v：内层拟合 / 内层早停 / 外层验证 的训练行号。返回 (验证预测, 测试预测, 最佳轮数或 None)。"""
    Xt_c = Xt_cat if test_rows is None else Xt_cat.iloc[test_rows]
    Xt_s = Xt_str if test_rows is None else Xt_str.iloc[test_rows]
    scale_t = SCALE_TEST if test_rows is None else SCALE_TEST[test_rows]
    if name in ('lgb_rel', 'cb_rel'):
        wa, ws = SCALE[a] ** 2, SCALE[s] ** 2        # 相对力量误差 × scale = kg 误差，所以按 scale² 加权
        norm = wa.mean()
        wa, ws = wa / norm, ws / norm
        ra, rs = y[a] / SCALE[a], y[s] / SCALE[s]
        if name == 'lgb_rel':
            m = lgb.LGBMRegressor(n_estimators=rounds, random_state=SEED, n_jobs=-1, **LGB_PARAMS)
            m.fit(X_cat.iloc[a], ra, sample_weight=wa, eval_set=[(X_cat.iloc[s], rs)], eval_sample_weight=[ws],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            best = int(m.best_iteration_ or rounds)
            p_val, p_test = m.predict(X_cat.iloc[v], num_iteration=best), m.predict(Xt_c, num_iteration=best)
        else:
            cats = [c for c in X_str.columns if c in CAT]
            m = catboost.CatBoostRegressor(iterations=rounds, loss_function='RMSE', eval_metric='RMSE',
                                           od_type='Iter', od_wait=300, random_seed=SEED, verbose=0,
                                           task_type='GPU', devices='0', allow_writing_files=False, **CB_PARAMS)
            m.fit(catboost.Pool(X_str.iloc[a], ra, cat_features=cats, weight=wa),
                  eval_set=catboost.Pool(X_str.iloc[s], rs, cat_features=cats, weight=ws), use_best_model=True)
            best = int(m.get_best_iteration()) + 1
            p_val, p_test = m.predict(X_str.iloc[v]), m.predict(Xt_s)
        return np.asarray(p_val, dtype='float64') * SCALE[v], np.asarray(p_test, dtype='float64') * scale_t, best
    cls = {'realmlp': RealMLP_TD_Regressor, 'tabm': TabM_D_Regressor}[name]
    require(cls is not None, f'pytabkit 不可用：{PYTABKIT_ERROR}')
    params = dict(NN_PARAMS[name], **(nn_overrides or {}))
    unknown = set(params) - set(inspect.signature(cls.__init__).parameters)
    require(not unknown, f'{name} 不支持参数 {sorted(unknown)}')
    Za, Zs, Zv, Zt = nn_frames(X_cat.iloc[a], X_cat.iloc[s], X_cat.iloc[v], Xt_c)
    mu, sd = y[a].mean(), y[a].std()
    m = cls(**params)
    m.fit(Za, (y[a] - mu) / sd, Zs, (y[s] - mu) / sd, cat_col_names=[c for c in Za.columns if c in CAT])
    p_val = np.asarray(m.predict(Zv), dtype='float64').reshape(-1) * sd + mu
    p_test = np.asarray(m.predict(Zt), dtype='float64').reshape(-1) * sd + mu
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test, None


rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for name in ZOO:
    for attempt in (['main', 'fallback'] if name in NN_FALLBACK else ['main']):
        if attempt == 'fallback':
            NN_PARAMS[name] = dict(NN_FALLBACK[name])
        started = time.perf_counter()
        try:
            p_val, p_test, _ = fit_model(name, pa, ps, pv, test_rows=np.arange(500), rounds=30,
                                         nn_overrides={'n_epochs': 3})
            require(len(p_val) == len(pv) and len(p_test) == 500, '预测长度不符')
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
            failed.pop(name, None)
            print(f'预检通过：{name:8s}（{attempt}）RMSE {rmse(y[pv], p_val):.1f}，用时 {time.perf_counter() - started:.0f}s')
            break
        except Exception as exc:
            failed[name] = repr(exc)
            print(f'预检失败：{name}（{attempt}）→ {exc!r}')
print('将运行：', [n for n in ZOO if n not in failed], '；跳过：', failed or '无')
