if BLEND_TEST is not None:
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
        denom = float(np.sum(d * d))
        return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / denom, 0, 1)) if denom > 0 else 0.0


    def apply_twins(pred, mate, level, alphas):
        out = pred.copy()
        for lv, a in alphas.items():
            m = level == lv
            out[m] = pred[m] + a * (mate[m] - pred[m])
        return out


    LEVEL, MATE = twins()
    T_LEVEL, T_MATE = twins(row_keys(test, RAW_FEATURES), row_keys(test, CORE))
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


    def s3_features(r):
        cols_tr, cols_te = [], []
        for builder in (f_exact, f_fuzzy, f_decay):
            for k in PREFIX:
                a, b = builder(k, r)
                cols_tr.append(a * VALID_TR)
                cols_te.append(b * VALID_TE)
        a, b = f_meet(r)
        cols_tr.append(a)
        cols_te.append(b)
        return np.column_stack(cols_tr), np.column_stack(cols_te)


    # 第二层只依赖训练行的 OOF（嵌套融合），对三种测试预测使用同一组 alpha / 系数
    alphas = {lv: fit_alpha(LEVEL == lv, NESTED, MATE) for lv in ['exact', 'core']}
    p1 = apply_twins(NESTED, MATE, LEVEL, alphas)
    p1_cf = NESTED.copy()
    for h in (0, 1):
        a_h = {lv: fit_alpha((HALF != h) & (LEVEL == lv), NESTED, MATE) for lv in ['exact', 'core']}
        m = HALF == h
        p1_cf[m] = apply_twins(NESTED[m], MATE[m], LEVEL[m], a_h)
    r1 = y - p1
    F, FT = s3_features(r1)
    p2_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        beta_h = np.linalg.lstsq(F[fit], r1[fit], rcond=None)[0]
        p2_cf[m] = p1_cf[m] + F[m] @ beta_h
    BETA = np.linalg.lstsq(F, r1, rcond=None)[0]
    STAGE2_ADJ = FT @ BETA
    stage_table = pd.DataFrame([{'step': '融合（嵌套）', 'oof_rmse': rmse(y, NESTED)},
                                {'step': '+ 双胞胎（交叉拟合）', 'oof_rmse': rmse(y, p1_cf)},
                                {'step': '+ S3 平滑（交叉拟合）', 'oof_rmse': rmse(y, p2_cf)}]).set_index('step')
    display(stage_table.round(3))
    stage_table.to_csv(OUTPUT_DIR / 'summary_stage2.csv')
    print('alpha：', {k: round(v, 3) for k, v in alphas.items()}, '｜测试中有双胞胎的行：', int((T_LEVEL != 'none').sum()),
          '｜S3 平均调整幅度：', round(float(np.abs(STAGE2_ADJ).mean()), 2), 'kg')

    for kind, pred in BLEND_TEST.items():
        final = apply_twins(pred, T_MATE, T_LEVEL, alphas) + STAGE2_ADJ
        for stage, values in [('base', pred), ('final', final)]:
            path = OUTPUT_DIR / f'submission_final_{kind}_{stage}.csv'
            pd.DataFrame({'id': test['id'], TARGET: np.clip(values, 0, None)}).to_csv(path, index=False)
            print('写出', path.name)
    np.save(OUTPUT_DIR / 'oof_nested_blend.npy', NESTED)
    np.save(OUTPUT_DIR / 'oof_final_crossfit.npy', p2_cf)
print('完成。')
