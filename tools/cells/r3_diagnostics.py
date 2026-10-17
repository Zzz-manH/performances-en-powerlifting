def bucket(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(str)


groups = {
    'age_half': P['age_half'].iloc[:n_train].to_numpy(),
    'ath_n': bucket(H['ath_n'].iloc[:n_train].fillna(-1), [-1, 0, 2, 3, 6, 1e9], ['未知', '1', '2', '3-5', '6+']),
    'ath_years_since_first': bucket(H['ath_years_since_first'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 100],
                                    ['未知', '0', '1', '2-4', '5+']),
    'ath_ever_abroad': H2['ath_ever_abroad'].iloc[:n_train].fillna(-1).to_numpy(),
    'ath_max_meet_states': bucket(H2['ath_max_meet_states'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 10, 1e9],
                                  ['未知', '0', '1', '2-4', '5-9', '10+']),
    'meet_n_states': bucket(M2['meet_n_states'].iloc[:n_train], [0, 1, 2, 5, 10, 1e9], ['0', '1', '2-4', '5-9', '10+']),
    'dup_exact_n>0': (D['dup_exact_n'].iloc[:n_train] > 0).to_numpy(),
    'Sex×Equipment': (train['Sex'] + '|' + train['Equipment']).to_numpy(),
    'Tested': train['Tested'].to_numpy(),
    'MeetYear 段': bucket(train['MeetYear'], [1900, 2000, 2010, 2015, 2020, 2030],
                          ['<2000', '2000s', '2010-14', '2015-19', '2020+']),
}
columns = {'R2': BLENDS['R2']['oof'], FINAL: BLENDS[FINAL]['oof'], f'{FINAL}+双胞胎+{BEST_S2}': S2_OOF[BEST_S2]}
frames = []
for label, pred in columns.items():
    res = y - pred
    parts = []
    for g, key in groups.items():
        part = pd.DataFrame({'k': key, 'r': res}).groupby('k')['r'].agg(['size', 'mean'])
        part.index = pd.MultiIndex.from_product([[g], part.index.astype(str)], names=['signal', 'level'])
        parts.append(part)
    frames.append(pd.concat(parts).rename(columns={'mean': label}))
diag = pd.concat(frames, axis=1)
diag = diag.loc[:, ~diag.columns.duplicated()]
display(diag.round(2))
diag.to_csv(OUTPUT_DIR / 'diagnostics_residuals.csv')
print('完成。')
