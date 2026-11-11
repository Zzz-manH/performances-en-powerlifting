def residual_table(pred):
    res = y - pred
    groups = {
        'age_half': RECORDING_PRECISION_FEATURES['age_half'].iloc[:n_train].to_numpy(),
        'bw_precision': RECORDING_PRECISION_FEATURES['bw_precision'].iloc[:n_train].to_numpy(),
        'other_state(-1=unknown)': TRAVEL_FEATURES['other_state'].iloc[:n_train].fillna(-1).to_numpy(),
        'abroad(-1=unknown)': TRAVEL_FEATURES['abroad'].iloc[:n_train].fillna(-1).to_numpy(),
        'meet_n octiles': pd.qcut(MEET_CONTEXT_FEATURES['meet_n'].iloc[:n_train], 8, labels=False, duplicates='drop').to_numpy(),
    }
    frames = []
    for name, key in groups.items():
        g = pd.DataFrame({'k': key, 'r': res}).groupby('k')['r'].agg(['size', 'mean'])
        g.index = pd.MultiIndex.from_product([[name], g.index], names=['signal', 'level'])
        frames.append(g)
    return pd.concat(frames)


check = residual_table(BLENDS[VARIANT_WEIGHT_CLASS_BASELINE]['oof']).rename(columns={'mean': VARIANT_WEIGHT_CLASS_BASELINE})
check[BEST] = residual_table(BLENDS[BEST]['oof'])['mean']
display(check.round(2))
