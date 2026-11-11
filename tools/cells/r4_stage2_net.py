CORE = CORE9


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def matching_group_statistics(train_keys, query_keys=None):
    table = pd.DataFrame({'k': train_keys, 'y': y}).groupby('k')['y'].agg(['sum', 'count'])
    if query_keys is None:                      # Training rows: leave one out
        keys = pd.Series(train_keys)
        n = keys.map(table['count']).to_numpy() - 1
        s = keys.map(table['sum']).to_numpy() - y
    else:                                       # Test rows: use all training rows
        keys = pd.Series(query_keys)
        n = keys.map(table['count']).fillna(0).to_numpy()
        s = keys.map(table['sum']).fillna(0).to_numpy()
    return n, np.where(n > 0, s / np.maximum(n, 1), np.nan)


def matching_record_statistics(query_exact=None, query_core=None):
    n_ex, m_ex = matching_group_statistics(row_keys(train, RAW_FEATURES), query_exact)
    n_co, m_co = matching_group_statistics(row_keys(train, CORE), query_core)
    level = np.select([n_ex > 0, n_co > 0], ['exact', 'core'], default='none')
    return level, np.where(level == 'exact', m_ex, m_co)


def fit_alpha(mask, pred, mate):
    d = mate[mask] - pred[mask]
    denom = float(np.sum(d * d))
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / denom, 0, 1)) if denom > 0 else 0.0


def apply_matching_correction(pred, mate, level, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


LEVEL, MATE = matching_record_statistics()
T_LEVEL, T_MATE = matching_record_statistics(row_keys(test, RAW_FEATURES), row_keys(test, CORE))
HALF = (pd.util.hash_pandas_object(train[['Sex', 'Country', 'State']], index=False).to_numpy() % 2).astype(int)
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


def residual_neighborhood_features(r):
    cols_tr, cols_te = [], []
    for builder in (exact_birthyear_residuals, nearby_birthyear_residuals, temporal_residuals):
        for k in RESIDUAL_MATCH_COLUMNS:
            a, b = builder(k, r)
            cols_tr.append(a * VALID_TR)
            cols_te.append(b * VALID_TE)
    a, b = meet_residuals(r)
    cols_tr.append(a)
    cols_te.append(b)
    return np.column_stack(cols_tr), np.column_stack(cols_te)




def stage2(oof, test_pred):
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    p1 = apply_matching_correction(oof, MATE, LEVEL, alphas)
    p1_cf = oof.copy()
    for h in (0, 1):
        a_h = {lv: fit_alpha((HALF != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        m = HALF == h
        p1_cf[m] = apply_matching_correction(oof[m], MATE[m], LEVEL[m], a_h)
    corrected_residuals = y - p1
    F, FT = residual_neighborhood_features(corrected_residuals)
    p2_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        p2_cf[m] = p1_cf[m] + F[m] @ np.linalg.lstsq(F[fit], corrected_residuals[fit], rcond=None)[0]
    beta = np.linalg.lstsq(F, corrected_residuals, rcond=None)[0]
    test_final = apply_matching_correction(test_pred, T_MATE, T_LEVEL, alphas) + FT @ beta
    return {'oof': rmse(y, oof), '+matched': rmse(y, p1_cf), SUMMARY_RESIDUAL_SMOOTHING: rmse(y, p2_cf),
            'fold_final': fold_rmse(p2_cf), 'test_final': test_final}


net_rows, NET = [], {}
for name in [v for v in [VARIANT_FINAL_BASELINE, FINAL4] if v is not None]:
    NET[name] = stage2(BLENDS[name]['oof'], BLENDS[name]['test'])
    net_rows.append({'variant': name, **{k: NET[name][k] for k in ['oof', '+matched', SUMMARY_RESIDUAL_SMOOTHING]}})
    path = OUTPUT_DIR / f'submission_{name}_blend3_stage2.csv'
    pd.DataFrame({'id': test['id'], TARGET: np.clip(NET[name]['test_final'], 0, None)}).to_csv(path, index=False)
    print('Saved', path.name)
net_table = pd.DataFrame(net_rows).set_index('variant')
display(net_table.round(3))
if FINAL4 is not None:
    delta = NET[FINAL4]['fold_final'] - NET[VARIANT_FINAL_BASELINE]['fold_final']
    print(f'{FINAL4} par rapport à la référence avec variables finales après lissage : moyenne {delta.mean():+.3f}, {int((delta < 0).sum())}/5 folds improved')
net_table.to_csv(OUTPUT_DIR / 'summary_net_after_stage2.csv')
