def bucket(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(str)


groups = {
    'age_half': P['age_half'].iloc[:n_train].to_numpy(),
    'bw_precision': P['bw_precision'].iloc[:n_train].to_numpy(),
    'meet_n 八分位': pd.qcut(M['meet_n'].iloc[:n_train], 8, labels=False, duplicates='drop').to_numpy(),
    'meet_n_states': bucket(M2['meet_n_states'].iloc[:n_train], [0, 1, 2, 5, 10, 1000], ['0', '1', '2-4', '5-9', '10+']),
    'ath_n': bucket(H['ath_n'].iloc[:n_train].fillna(-1), [-1, 0, 2, 3, 6, 1e9], ['未知', '1', '2', '3-5', '6+']),
    'ath_years_since_first': bucket(H['ath_years_since_first'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 100],
                                    ['未知', '0', '1', '2-4', '5+']),
    'dup_exact_n>0': (D['dup_exact_n'].iloc[:n_train] > 0).to_numpy(),
    'div_n': bucket(D['div_n'].iloc[:n_train], [1, 2, 4, 8, 1e9], ['1', '2-3', '4-7', '8+']),
    'Sex×Equipment': (train['Sex'] + '|' + train['Equipment']).to_numpy(),
    'Tested': train['Tested'].to_numpy(),
    'MeetYear 段': bucket(train['MeetYear'], [1900, 2000, 2010, 2015, 2020, 2030],
                          ['<2000', '2000s', '2010-14', '2015-19', '2020+']),
}
frames = []
for name in ['R1', FINAL] if FINAL != 'R1' else ['R1']:
    res = y - BLENDS[name]['oof']
    rows = []
    for g, key in groups.items():
        part = pd.DataFrame({'k': key, 'r': res}).groupby('k')['r'].agg(['size', 'mean'])
        part.index = pd.MultiIndex.from_product([[g], part.index.astype(str)], names=['signal', 'level'])
        rows.append(part)
    frames.append(pd.concat(rows).rename(columns={'mean': name}))
diag = pd.concat(frames, axis=1)
diag = diag.loc[:, ~diag.columns.duplicated()]
display(diag.round(2))
diag.to_csv(OUTPUT_DIR / 'diagnostics_residuals.csv')
print('完成。')
