CORE = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def matching_group_statistics(train_keys, query_keys=None):
    table = pd.DataFrame({'k': train_keys, 'y': y}).groupby('k')['y'].agg(['sum', 'count'])
    if query_keys is None:                       # Training rows: leave one out
        keys = pd.Series(train_keys)
        n = keys.map(table['count']).to_numpy() - 1
        s = keys.map(table['sum']).to_numpy() - y
    else:                                        # Test rows: use all training rows
        keys = pd.Series(query_keys)
        n = keys.map(table['count']).fillna(0).to_numpy()
        s = keys.map(table['sum']).fillna(0).to_numpy()
    return n, np.where(n > 0, s / np.maximum(n, 1), np.nan)


def matching_record_statistics(train_keys_exact, train_keys_core, query_exact=None, query_core=None):
    n_ex, m_ex = matching_group_statistics(train_keys_exact, query_exact)
    n_co, m_co = matching_group_statistics(train_keys_core, query_core)
    level = np.select([n_ex > 0, n_co > 0], ['exact', 'core'], default='none')
    return level, np.where(level == 'exact', m_ex, m_co)


def fit_alpha(mask, pred, mate):
    d = mate[mask] - pred[mask]
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / np.sum(d * d), 0, 1)) if mask.any() else 0.0


def apply_matching_correction(pred, mate, level, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


ex_tr, co_tr = row_keys(train, RAW_FEATURES), row_keys(train, CORE)
ex_te, co_te = row_keys(test, RAW_FEATURES), row_keys(test, CORE)
LEVEL, MATE = matching_record_statistics(ex_tr, co_tr)
T_LEVEL, T_MATE = matching_record_statistics(ex_tr, co_tr, ex_te, co_te)
half = (co_tr % 2).astype(int)

twin_rows, exports = [], {'F0_blend3': BLENDS[VARIANT_WEIGHT_CLASS_BASELINE], FINAL['name']: FINAL}
for name, item in exports.items():
    oof = item['oof']
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    crossfit = oof.copy()
    for h in (0, 1):
        a = {lv: fit_alpha((half != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        crossfit[half == h] = apply_matching_correction(oof[half == h], MATE[half == h], LEVEL[half == h], a)
    test_fixed = apply_matching_correction(item['test'], T_MATE, T_LEVEL, alphas)
    twin_rows.append({'name': name, 'oof_rmse': rmse(y, oof), 'oof_rmse_twinfix_crossfit': rmse(y, crossfit),
                      'alpha_exact': alphas['exact'], 'alpha_core': alphas['core'],
                      'test_rows_exact': int((T_LEVEL == 'exact').sum()), 'test_rows_core': int((T_LEVEL == 'core').sum())})
    for suffix, pred in [('', item['test']), ('_twinfix', test_fixed)]:
        path = OUTPUT_DIR / f'submission_{name}{suffix}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('Saved', path.name)
twin_table = pd.DataFrame(twin_rows).set_index('name')
display(twin_table.round(3))
twin_table.to_csv(OUTPUT_DIR / 'summary_twins.csv')
np.save(OUTPUT_DIR / 'oof_F0_blend3.npy', BLENDS[VARIANT_WEIGHT_CLASS_BASELINE]['oof'])
np.save(OUTPUT_DIR / f'oof_{FINAL["name"]}.npy', FINAL['oof'])
print('Done.')
