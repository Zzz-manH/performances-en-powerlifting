base_oof, base_test = BLENDS[FINAL]['oof'], BLENDS[FINAL]['test']
TR, TE = train.reset_index(drop=True), test.reset_index(drop=True)


def text_key(frame, columns):
    out = frame[columns[0]].astype(str)
    for c in columns[1:]:
        out = out + '|' + frame[c].astype(str)
    return out


BY_TR = np.floor(TR['MeetYear'] - TR['Age'] + 1e-9).astype(int)
BY_TE = np.floor(TE['MeetYear'] - TE['Age'] + 1e-9).astype(int)
YR_TR, YR_TE = TR['MeetYear'].astype(int), TE['MeetYear'].astype(int)
PREFIX = {'A0': ['Sex', 'Country', 'State'], 'A1': ['Sex', 'Country', 'State', 'Federation'],
          'A2': ['Sex', 'Country', 'State', 'Federation', 'Equipment'],
          'A3': ['Sex', 'Country', 'State', 'Federation', 'Equipment', 'WeightClassKg']}
SHRINK = {'A0': 5.0, 'A1': 5.0, 'A2': 3.0, 'A3': 2.0}
PFX_TR = {k: text_key(TR, c) for k, c in PREFIX.items()}
PFX_TE = {k: text_key(TE, c) for k, c in PREFIX.items()}
MEET_COLS = ['Federation', 'MeetYear', 'MeetCountry', 'MeetState', 'Sex', 'Equipment']
MEET_TR, MEET_TE = text_key(TR, MEET_COLS), text_key(TE, MEET_COLS)
VALID_TR = (TR['Country'] != 'Unknown').to_numpy(dtype='float64')
VALID_TE = (TE['Country'] != 'Unknown').to_numpy(dtype='float64')


def sums(keys, values):
    table = pd.DataFrame({'k': keys.to_numpy(), 'v': values}).groupby('k')['v'].agg(['sum', 'count'])
    return table['sum'], table['count']


def look(table, keys):
    s, n = table
    return keys.map(s).fillna(0).to_numpy(dtype='float64'), keys.map(n).fillna(0).to_numpy(dtype='float64')


def f_exact(k, r):
    ktr, kte = PFX_TR[k] + '|' + BY_TR.astype(str), PFX_TE[k] + '|' + BY_TE.astype(str)
    table = sums(ktr, r)
    s, n = look(table, ktr)
    st, nt = look(table, kte)
    return (s - r) / (n - 1 + SHRINK[k]), st / (nt + SHRINK[k])


def f_fuzzy(k, r):
    table = sums(PFX_TR[k] + '|' + BY_TR.astype(str), r)
    s, n, st, nt = np.zeros(n_train), np.zeros(n_train), np.zeros(len(TE)), np.zeros(len(TE))
    for d in (-1, 0, 1):
        a, b = look(table, PFX_TR[k] + '|' + (BY_TR + d).astype(str))
        s, n = s + a, n + b
        a, b = look(table, PFX_TE[k] + '|' + (BY_TE + d).astype(str))
        st, nt = st + a, nt + b
    return (s - r) / (n - 1 + SHRINK[k]), st / (nt + SHRINK[k])


def f_decay(k, r, tau=3.0, max_gap=8):
    base_tr = PFX_TR[k] + '|' + BY_TR.astype(str) + '|'
    base_te = PFX_TE[k] + '|' + BY_TE.astype(str) + '|'
    table = sums(base_tr + YR_TR.astype(str), r)
    s, n, st, nt = np.zeros(n_train), np.zeros(n_train), np.zeros(len(TE)), np.zeros(len(TE))
    for d in range(-max_gap, max_gap + 1):
        w = float(np.exp(-abs(d) / tau))
        a, b = look(table, base_tr + (YR_TR + d).astype(str))
        s, n = s + w * a, n + w * b
        a, b = look(table, base_te + (YR_TE + d).astype(str))
        st, nt = st + w * a, nt + w * b
    return (s - r) / (n - 1 + SHRINK[k]), st / (nt + SHRINK[k])     # d = 0 时自己权重为 1，已扣除


def f_meet(r, lam=20.0):
    table = sums(MEET_TR, r)
    s, n = look(table, MEET_TR)
    st, nt = look(table, MEET_TE)
    return (s - r) / (n - 1 + lam), st / (nt + lam)


def stage2_features(r, kind):
    builders = {'exact': f_exact, 'fuzzy': f_fuzzy, 'decay': f_decay}
    cols_tr, cols_te, names = [], [], []
    for kd in (['exact', 'fuzzy', 'decay'] if kind == 'all' else [kind]):
        for k in PREFIX:
            a, b = builders[kd](k, r)
            cols_tr.append(a * VALID_TR)
            cols_te.append(b * VALID_TE)
            names.append(f'{kd}_{k}')
    a, b = f_meet(r)
    cols_tr.append(a)
    cols_te.append(b)
    names.append('meet')
    return np.column_stack(cols_tr), np.column_stack(cols_te), names


alphas = {lv: fit_alpha(LEVEL == lv, base_oof, MATE) for lv in ['exact', 'core']}
p1 = apply_twins(base_oof, MATE, LEVEL, alphas)
p1_cf = base_oof.copy()
for h in (0, 1):
    a_h = {lv: fit_alpha((HALF != h) & (LEVEL == lv), base_oof, MATE) for lv in ['exact', 'core']}
    m = HALF == h
    p1_cf[m] = apply_twins(base_oof[m], MATE[m], LEVEL[m], a_h)
t1 = apply_twins(base_test, T_MATE, T_LEVEL, alphas)
r1 = y - p1
FEATS = {kind: stage2_features(r1, kind) for kind in ['exact', 'fuzzy', 'decay', 'all']}
GBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8,
                  subsample_freq=1, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)


def linear_fit_predict(F_fit, r_fit, F_apply):
    beta = np.linalg.lstsq(F_fit, r_fit, rcond=None)[0]
    return F_apply @ beta


def gbm_fit_predict(F_fit, r_fit, F_apply):
    return lgb.LGBMRegressor(**GBM_PARAMS).fit(F_fit, r_fit).predict(F_apply)


S2_VARIANTS = {'S0_exact': ('exact', linear_fit_predict), 'S1_fuzzy': ('fuzzy', linear_fit_predict),
               'S2_decay': ('decay', linear_fit_predict), 'S3_all_linear': ('all', linear_fit_predict),
               'S4_all_gbm': ('all', gbm_fit_predict)}
rows, S2_OOF, S2_TEST = [], {}, {}
for name, (kind, fn) in S2_VARIANTS.items():
    F, FT, names = FEATS[kind]
    pred_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        pred_cf[m] = p1_cf[m] + fn(F[fit], r1[fit], F[m])
    S2_OOF[name], S2_TEST[name] = pred_cf, t1 + fn(F, r1, FT)
    rows.append({'variant': name, 'n_features': len(names), 'crossfit_rmse': rmse(y, pred_cf),
                 'gain_vs_twins': rmse(y, p1_cf) - rmse(y, pred_cf),
                 **{f'fold{k}': v for k, v in fold_rmse(pred_cf).items()}})
s2_table = pd.DataFrame(rows).set_index('variant').sort_values('crossfit_rmse')
print(f'{FINAL} 融合：OOF {rmse(y, base_oof):.3f} → 加双胞胎（交叉拟合）{rmse(y, p1_cf):.3f}')
display(s2_table.round(3))
s2_table.to_csv(OUTPUT_DIR / 'summary_stage2_variants.csv')
BEST_S2 = s2_table.index[0]
for name in dict.fromkeys(['S0_exact', BEST_S2]):
    path = OUTPUT_DIR / f'submission_{FINAL}_blend3_twins_{name}.csv'
    pd.DataFrame({'id': test['id'], TARGET: np.clip(S2_TEST[name], 0, None)}).to_csv(path, index=False)
    print('写出', path.name)
print(f'交叉拟合最好的第二层：{BEST_S2}')
