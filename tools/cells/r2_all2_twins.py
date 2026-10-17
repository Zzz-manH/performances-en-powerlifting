if KEPT:
    for model in models_for('ALL2'):
        Xtr, Xte, cat_cols = frames2('ALL2', model)
        p_val, p_test, _ = fit_predict(model, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                       Xte.iloc[:500], cat_cols, rounds=30)          # ALL2 的预检
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'ALL2 {model} 预检失败')
    run_variants(['ALL2'])
    BLENDS['ALL2'] = blend_variant('ALL2')
    final_table = paired_table({v: BLENDS[v]['oof'] for v in ['R1'] + KEPT + ['ALL2']})
    display(final_table.round(3))
    final_table.to_csv(OUTPUT_DIR / 'summary_blend_all2.csv')
    FINAL = 'ALL2'
else:
    print('没有达标的新块，保持 R1。')
    FINAL = 'R1'

CORE = CORE9


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def mate_stats(train_keys, query_keys=None):
    table = pd.DataFrame({'k': train_keys, 'y': y}).groupby('k')['y'].agg(['sum', 'count'])
    if query_keys is None:                      # 训练行：留一
        keys = pd.Series(train_keys)
        n = keys.map(table['count']).to_numpy() - 1
        s = keys.map(table['sum']).to_numpy() - y
    else:                                       # 测试行：全部训练行
        keys = pd.Series(query_keys)
        n = keys.map(table['count']).fillna(0).to_numpy()
        s = keys.map(table['sum']).fillna(0).to_numpy()
    return n, np.where(n > 0, s / np.maximum(n, 1), np.nan)


def twins(query_exact=None, query_core=None):
    n_ex, m_ex = mate_stats(row_keys(train, RAW_FEATURES), query_exact)
    n_co, m_co = mate_stats(row_keys(train, CORE), query_core)
    level = np.select([n_ex > 0, n_co > 0], ['exact', 'core'], default='none')
    return level, np.where(level == 'exact', m_ex, m_co)


def fit_alpha(mask, pred, mate):
    d = mate[mask] - pred[mask]
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / np.sum(d * d), 0, 1)) if mask.any() else 0.0


def apply_twins(pred, mate, level, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


LEVEL, MATE = twins()
T_LEVEL, T_MATE = twins(row_keys(test, RAW_FEATURES), row_keys(test, CORE))
half = (row_keys(train, CORE) % 2).astype(int)
twin_rows = []
for name in ['R1', FINAL] if FINAL != 'R1' else ['R1']:
    oof, test_pred = BLENDS[name]['oof'], BLENDS[name]['test']
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    crossfit = oof.copy()
    for h in (0, 1):
        a = {lv: fit_alpha((half != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        crossfit[half == h] = apply_twins(oof[half == h], MATE[half == h], LEVEL[half == h], a)
    fixed = apply_twins(test_pred, T_MATE, T_LEVEL, alphas)
    twin_rows.append({'name': name, 'oof_rmse': rmse(y, oof), 'oof_rmse_twinfix_crossfit': rmse(y, crossfit),
                      'alpha_exact': alphas['exact'], 'alpha_core': alphas['core']})
    for suffix, pred in [('', test_pred), ('_twinfix', fixed)]:
        path = OUTPUT_DIR / f'submission_{name}_blend3{suffix}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('写出', path.name)
    np.save(OUTPUT_DIR / f'oof_{name}_blend3.npy', oof)
twin_table = pd.DataFrame(twin_rows).set_index('name')
display(twin_table.round(3))
twin_table.to_csv(OUTPUT_DIR / 'summary_twins.csv')
