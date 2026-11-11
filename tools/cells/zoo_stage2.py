CORE = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']
ATHLETE_KEYS = {
    MATCH_RESIDENCE: ['Sex', 'Country', 'State', 'birth_year'],
    MATCH_FEDERATION: ['Sex', 'Country', 'State', 'Federation', 'birth_year'],
    MATCH_EQUIPMENT: ['Sex', 'Country', 'State', 'Federation', 'birth_year', 'Equipment'],
    MATCH_WEIGHT_CLASS: ['Sex', 'Country', 'State', 'Federation', 'birth_year', 'Equipment', 'WeightClassKg'],
    MATCH_MEET: ['Federation', 'MeetYear', 'MeetCountry', 'MeetState', 'Sex', 'Equipment'],
}
RESIDUAL_SHRINKAGE = {MATCH_RESIDENCE: 5.0, MATCH_FEDERATION: 5.0, MATCH_EQUIPMENT: 3.0, MATCH_WEIGHT_CLASS: 2.0, MATCH_MEET: 20.0}


def keyed(frame):
    out = frame[RAW_FEATURES].copy()
    out['birth_year'] = np.floor(out['MeetYear'] - out['Age'] + 1e-9)
    return out


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


KT, KQ = keyed(train), keyed(test)
EX_TR, EX_TE = row_keys(KT, RAW_FEATURES), row_keys(KQ, RAW_FEATURES)
CO_TR, CO_TE = row_keys(KT, CORE), row_keys(KQ, CORE)
AK_TR = {k: row_keys(KT, cols) for k, cols in ATHLETE_KEYS.items()}
AK_TE = {k: row_keys(KQ, cols) for k, cols in ATHLETE_KEYS.items()}
VALID_TR = (train['Country'] != 'Unknown').to_numpy(dtype='float64')
VALID_TE = (test['Country'] != 'Unknown').to_numpy(dtype='float64')
HALF = (AK_TR[MATCH_RESIDENCE] % 2).astype(int)


def group_sum(keys_tr, values, keys_query=None):
    """Training rows: sums/counts excluding the current row; query rows: sums/counts over all matching training rows."""
    table = pd.DataFrame({'k': keys_tr, 'v': values}).groupby('k')['v'].agg(['sum', 'count'])
    if keys_query is None:
        k = pd.Series(keys_tr)
        return k.map(table['sum']).to_numpy() - values, k.map(table['count']).to_numpy() - 1
    k = pd.Series(keys_query)
    return k.map(table['sum']).fillna(0).to_numpy(), k.map(table['count']).fillna(0).to_numpy()


def matching_record_values(query=False):
    s_ex, n_ex = group_sum(EX_TR, y, EX_TE if query else None)
    s_co, n_co = group_sum(CO_TR, y, CO_TE if query else None)
    level = np.select([n_ex > 0, n_co > 0], ['exact', 'core'], default='none')
    mate = np.where(level == 'exact', s_ex / np.maximum(n_ex, 1), s_co / np.maximum(n_co, 1))
    return level, mate


TW_LEVEL, TW_MATE = matching_record_values()
TW_LEVEL_T, TW_MATE_T = matching_record_values(query=True)


def fit_alpha(mask, pred):
    d = TW_MATE[mask] - pred[mask]
    denom = float(np.sum(d * d))
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / denom, 0, 1)) if denom > 0 else 0.0


def apply_matching_correction(pred, level, mate, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


def athlete_features(residual, query=False):
    cols = []
    for k in ATHLETE_KEYS:
        s, n = group_sum(AK_TR[k], residual, AK_TE[k] if query else None)
        f = s / (n + RESIDUAL_SHRINKAGE[k])
        if k != MATCH_MEET:
            f = f * (VALID_TE if query else VALID_TR)
        cols.append(f)
    return np.column_stack(cols)


def stage2(oof, test_pred, label):
    alphas = {lv: fit_alpha(TW_LEVEL == lv, oof) for lv in ['exact', 'core']}
    p1 = apply_matching_correction(oof, TW_LEVEL, TW_MATE, alphas)
    p1_cf = oof.copy()
    for h in (0, 1):
        a_h = {lv: fit_alpha((HALF != h) & (TW_LEVEL == lv), oof) for lv in ['exact', 'core']}
        m = HALF == h
        p1_cf[m] = apply_matching_correction(oof[m], TW_LEVEL[m], TW_MATE[m], a_h)
    t1 = apply_matching_correction(test_pred, TW_LEVEL_T, TW_MATE_T, alphas)

    corrected_residuals = y - p1
    F, FT = athlete_features(corrected_residuals), athlete_features(corrected_residuals, query=True)
    p2_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        beta_h = np.linalg.lstsq(F[fit], corrected_residuals[fit], rcond=None)[0]
        p2_cf[m] = p1_cf[m] + F[m] @ beta_h
    beta = np.linalg.lstsq(F, corrected_residuals, rcond=None)[0]
    gain = rmse(y, p1_cf) - rmse(y, p2_cf)
    use_athlete = bool(gain >= STAGE2_MIN_GAIN)
    t2 = t1 + FT @ beta if use_athlete else t1.copy()
    row = {'base': label, 'oof_rmse': rmse(y, oof), 'twins_crossfit': rmse(y, p1_cf),
           'twins_athlete_crossfit': rmse(y, p2_cf), 'athlete_gain': gain, 'athlete_used': use_athlete,
           'alpha_exact': alphas['exact'], 'alpha_core': alphas['core'],
           **{f'beta_{k}': float(b) for k, b in zip(ATHLETE_KEYS, beta)},
           'test_rows_twins': int((TW_LEVEL_T != 'none').sum()),
           'test_mean_abs_athlete_adj': float(np.abs(t2 - t1).mean())}
    return row, {'base': test_pred, 'twins': t1, 'final': t2}


rows, exports = [], {}
row, preds = stage2(STACK_OOF, STACK_TEST, f'Experiment 08, five-fold stacking({BEST_METHOD})')
rows.append(row)
exports['zoo5f'] = preds

TEN_FOLD_ENSEMBLE_SEEDS = {MODEL_LIGHTGBM: [42], MODEL_XGBOOST: [42, 7, 2024], MODEL_CATBOOST_ABSOLUTE: [42, 7, 2024], 'mlp': [42, 7, 2024]}
try:
    oofs, tests = [], []
    for model, seeds in TEN_FOLD_ENSEMBLE_SEEDS.items():
        oof, pred = np.zeros(n_train), np.zeros(len(test))
        for seed in seeds:
            for i, (_, valid) in enumerate(KFold(10, shuffle=True, random_state=seed).split(np.arange(n_train))):
                saved = np.load(TEN_FOLD_ENSEMBLE_DIR / f'{model}_s{seed}_f{i}.npz')
                require(len(saved['val']) == len(valid) and len(saved['test']) == len(test), 'v2 file shape mismatch')
                oof[valid] += saved['val'] / len(seeds)
                pred += saved['test'] / (10 * len(seeds))
        oofs.append(oof)
        tests.append(pred)
    w2, _ = nnls(np.column_stack(oofs), y)
    row, preds = stage2(np.column_stack(oofs) @ w2, np.column_stack(tests) @ w2, 'v2 (10 folds per seed, NNLS)')
    rows.append(row)
    exports[ENSEMBLE_TEN_FOLD] = preds
except FileNotFoundError as exc:
    print('v2 fold files not found; skipping v2: ', exc)

stage2_table = pd.DataFrame(rows).set_index('base')
display(stage2_table.round(3).T)
stage2_table.to_csv(OUTPUT_DIR / 'summary_stage2.csv')
for name, preds in exports.items():
    for kind, pred in preds.items():
        path = OUTPUT_DIR / f'submission_{name}_{kind}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('Saved', path.name)
print('Done.')
