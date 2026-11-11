def bucket(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(str)


groups = {
    'age_half': RECORDING_PRECISION_FEATURES['age_half'].iloc[:n_train].to_numpy(),
    'ath_n': bucket(ATHLETE_HISTORY_FEATURES['ath_n'].iloc[:n_train].fillna(-1), [-1, 0, 2, 3, 6, 1e9], ['unknown', '1', '2', '3-5', '6+']),
    'ath_years_since_first': bucket(ATHLETE_HISTORY_FEATURES['ath_years_since_first'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 100],
                                    ['unknown', '0', '1', '2-4', '5+']),
    'ath_ever_abroad': ATHLETE_CAREER_FEATURES['ath_ever_abroad'].iloc[:n_train].fillna(-1).to_numpy(),
    'ath_max_meet_states': bucket(ATHLETE_CAREER_FEATURES['ath_max_meet_states'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 10, 1e9],
                                  ['unknown', '0', '1', '2-4', '5-9', '10+']),
    'meet_n_states': bucket(EXTENDED_MEET_CONTEXT_FEATURES['meet_n_states'].iloc[:n_train], [0, 1, 2, 5, 10, 1e9], ['0', '1', '2-4', '5-9', '10+']),
    'dup_exact_n>0': (DIVISION_FEATURES['dup_exact_n'].iloc[:n_train] > 0).to_numpy(),
    'Sex×Equipment': (train['Sex'] + '|' + train['Equipment']).to_numpy(),
    'Tested': train['Tested'].to_numpy(),
    'MeetYear interval': bucket(train['MeetYear'], [1900, 2000, 2010, 2015, 2020, 2030],
                          ['<2000', '2000s', '2010-14', '2015-19', '2020+']),
}
columns = {VARIANT_HISTORY_BASELINE: BLENDS[VARIANT_HISTORY_BASELINE]['oof'], FINAL: BLENDS[FINAL]['oof'], f'{FINAL}+matched+{BEST_SMOOTHING}': SMOOTHING_OOF[BEST_SMOOTHING]}
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
print('Done.')
