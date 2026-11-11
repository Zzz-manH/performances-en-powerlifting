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
RESIDUAL_MATCH_COLUMNS = {MATCH_RESIDENCE: ['Sex', 'Country', 'State'], MATCH_FEDERATION: ['Sex', 'Country', 'State', 'Federation'],
          MATCH_EQUIPMENT: ['Sex', 'Country', 'State', 'Federation', 'Equipment'],
          MATCH_WEIGHT_CLASS: ['Sex', 'Country', 'State', 'Federation', 'Equipment', 'WeightClassKg']}
RESIDUAL_SHRINKAGE = {MATCH_RESIDENCE: 5.0, MATCH_FEDERATION: 5.0, MATCH_EQUIPMENT: 3.0, MATCH_WEIGHT_CLASS: 2.0}
PFX_TR = {k: text_key(TR, c) for k, c in RESIDUAL_MATCH_COLUMNS.items()}
PFX_TE = {k: text_key(TE, c) for k, c in RESIDUAL_MATCH_COLUMNS.items()}
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


def exact_birthyear_residuals(k, r):
    ktr, kte = PFX_TR[k] + '|' + BY_TR.astype(str), PFX_TE[k] + '|' + BY_TE.astype(str)
    table = sums(ktr, r)
    s, n = look(table, ktr)
    st, nt = look(table, kte)
    return (s - r) / (n - 1 + RESIDUAL_SHRINKAGE[k]), st / (nt + RESIDUAL_SHRINKAGE[k])


def nearby_birthyear_residuals(k, r):
    table = sums(PFX_TR[k] + '|' + BY_TR.astype(str), r)
    s, n, st, nt = np.zeros(n_train), np.zeros(n_train), np.zeros(len(TE)), np.zeros(len(TE))
    for d in (-1, 0, 1):
        a, b = look(table, PFX_TR[k] + '|' + (BY_TR + d).astype(str))
        s, n = s + a, n + b
        a, b = look(table, PFX_TE[k] + '|' + (BY_TE + d).astype(str))
        st, nt = st + a, nt + b
    return (s - r) / (n - 1 + RESIDUAL_SHRINKAGE[k]), st / (nt + RESIDUAL_SHRINKAGE[k])


def temporal_residuals(k, r, tau=3.0, max_gap=8):
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
    return (s - r) / (n - 1 + RESIDUAL_SHRINKAGE[k]), st / (nt + RESIDUAL_SHRINKAGE[k])     # At d = 0 the current row has weight 1 and is subtracted.


def meet_residuals(r, lam=20.0):
    table = sums(MEET_TR, r)
    s, n = look(table, MEET_TR)
    st, nt = look(table, MEET_TE)
    return (s - r) / (n - 1 + lam), st / (nt + lam)


def residual_smoothing_variant_features(r, kind):
    builders = {'exact': exact_birthyear_residuals, 'fuzzy': nearby_birthyear_residuals, 'decay': temporal_residuals}
    cols_tr, cols_te, names = [], [], []
    for kd in (['exact', 'fuzzy', 'decay'] if kind == 'all' else [kind]):
        for k in RESIDUAL_MATCH_COLUMNS:
            a, b = builders[kd](k, r)
            cols_tr.append(a * VALID_TR)
            cols_te.append(b * VALID_TE)
            names.append(f'{kd}_{k}')
    a, b = meet_residuals(r)
    cols_tr.append(a)
    cols_te.append(b)
    names.append('meet')
    return np.column_stack(cols_tr), np.column_stack(cols_te), names


alphas = {lv: fit_alpha(LEVEL == lv, base_oof, MATE) for lv in ['exact', 'core']}
p1 = apply_matching_correction(base_oof, MATE, LEVEL, alphas)
p1_cf = base_oof.copy()
for h in (0, 1):
    a_h = {lv: fit_alpha((HALF != h) & (LEVEL == lv), base_oof, MATE) for lv in ['exact', 'core']}
    m = HALF == h
    p1_cf[m] = apply_matching_correction(base_oof[m], MATE[m], LEVEL[m], a_h)
t1 = apply_matching_correction(base_test, T_MATE, T_LEVEL, alphas)
corrected_residuals = y - p1
FEATS = {kind: residual_smoothing_variant_features(corrected_residuals, kind) for kind in ['exact', 'fuzzy', 'decay', 'all']}
GBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8,
                  subsample_freq=1, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)


def linear_fit_predict(F_fit, r_fit, F_apply):
    beta = np.linalg.lstsq(F_fit, r_fit, rcond=None)[0]
    return F_apply @ beta


def gbm_fit_predict(F_fit, r_fit, F_apply):
    return lgb.LGBMRegressor(**GBM_PARAMS).fit(F_fit, r_fit).predict(F_apply)


RESIDUAL_SMOOTHING_VARIANTS = {SMOOTHING_EXACT_BIRTHYEAR: ('exact', linear_fit_predict), SMOOTHING_NEARBY_BIRTHYEAR: ('fuzzy', linear_fit_predict),
               SMOOTHING_TEMPORAL: ('decay', linear_fit_predict), SMOOTHING_COMBINED_LINEAR: ('all', linear_fit_predict),
               SMOOTHING_COMBINED_BOOSTING: ('all', gbm_fit_predict)}
rows, SMOOTHING_OOF, SMOOTHING_TEST = [], {}, {}
for name, (kind, fn) in RESIDUAL_SMOOTHING_VARIANTS.items():
    F, FT, names = FEATS[kind]
    pred_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        pred_cf[m] = p1_cf[m] + fn(F[fit], corrected_residuals[fit], F[m])
    SMOOTHING_OOF[name], SMOOTHING_TEST[name] = pred_cf, t1 + fn(F, corrected_residuals, FT)
    rows.append({'variant': name, 'n_features': len(names), 'crossfit_rmse': rmse(y, pred_cf),
                 'gain_vs_twins': rmse(y, p1_cf) - rmse(y, pred_cf),
                 **{f'fold{k}': v for k, v in fold_rmse(pred_cf).items()}})
smoothing_table = pd.DataFrame(rows).set_index('variant').sort_values('crossfit_rmse')
print(f'{FINAL} blend: OOF {rmse(y, base_oof):.3f} -> matched records (cross-fitted coefficients){rmse(y, p1_cf):.3f}')
display(smoothing_table.round(3))
smoothing_table.to_csv(OUTPUT_DIR / 'summary_stage2_variants.csv')
BEST_SMOOTHING = smoothing_table.index[0]
for name in dict.fromkeys([SMOOTHING_EXACT_BIRTHYEAR, BEST_SMOOTHING]):
    path = OUTPUT_DIR / f'submission_{FINAL}_blend3_twins_{name}.csv'
    pd.DataFrame({'id': test['id'], TARGET: np.clip(SMOOTHING_TEST[name], 0, None)}).to_csv(path, index=False)
    print('Saved', path.name)
print(f'Best local post-processing result: {BEST_SMOOTHING}')
