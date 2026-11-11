if KEPT:
    for model in models_for(VARIANT_HISTORY_COMBINED):
        Xtr, Xte, cat_cols = history_feature_frames(VARIANT_HISTORY_COMBINED, model)
        p_val, p_test, _ = fit_predict(model, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                       Xte.iloc[:500], cat_cols, rounds=30)          # Contexte et historique du sportif smoke check
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'ALL2 {model} Smoke check failed')
    run_variants([VARIANT_HISTORY_COMBINED])
    BLENDS[VARIANT_HISTORY_COMBINED] = blend_variant(VARIANT_HISTORY_COMBINED)
    final_table = paired_table({v: BLENDS[v]['oof'] for v in [VARIANT_CONTEXT_BASELINE] + KEPT + [VARIANT_HISTORY_COMBINED]})
    display(final_table.round(3))
    final_table.to_csv(OUTPUT_DIR / 'summary_blend_all2.csv')
    FINAL = VARIANT_HISTORY_COMBINED
else:
    print('Aucun nouveau bloc ne satisfait la règle de sélection ; conserver la référence avec contexte.')
    FINAL = VARIANT_CONTEXT_BASELINE

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
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / np.sum(d * d), 0, 1)) if mask.any() else 0.0


def apply_matching_correction(pred, mate, level, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


LEVEL, MATE = matching_record_statistics()
T_LEVEL, T_MATE = matching_record_statistics(row_keys(test, RAW_FEATURES), row_keys(test, CORE))
half = (row_keys(train, CORE) % 2).astype(int)
twin_rows = []
for name in [VARIANT_CONTEXT_BASELINE, FINAL] if FINAL != VARIANT_CONTEXT_BASELINE else [VARIANT_CONTEXT_BASELINE]:
    oof, test_pred = BLENDS[name]['oof'], BLENDS[name]['test']
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    crossfit = oof.copy()
    for h in (0, 1):
        a = {lv: fit_alpha((half != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        crossfit[half == h] = apply_matching_correction(oof[half == h], MATE[half == h], LEVEL[half == h], a)
    fixed = apply_matching_correction(test_pred, T_MATE, T_LEVEL, alphas)
    twin_rows.append({'name': name, 'oof_rmse': rmse(y, oof), 'oof_rmse_twinfix_crossfit': rmse(y, crossfit),
                      'alpha_exact': alphas['exact'], 'alpha_core': alphas['core']})
    for suffix, pred in [('', test_pred), ('_twinfix', fixed)]:
        path = OUTPUT_DIR / f'submission_{name}_blend3{suffix}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('Saved', path.name)
    np.save(OUTPUT_DIR / f'oof_{name}_blend3.npy', oof)
twin_table = pd.DataFrame(twin_rows).set_index('name')
display(twin_table.round(3))
twin_table.to_csv(OUTPUT_DIR / 'summary_twins.csv')
