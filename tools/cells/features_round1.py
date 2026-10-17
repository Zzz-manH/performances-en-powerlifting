both = pd.concat([train[RAW_FEATURES], test[RAW_FEATURES]], ignore_index=True)

# F0：v2 的体重级别特征
weight_class = pd.to_numeric(both['WeightClassKg'].str.replace('+', '', regex=False), errors='coerce')
F0_EXTRA = pd.DataFrame({
    'WeightClass_num': weight_class,
    'Is_SHW': both['WeightClassKg'].str.endswith('+').astype('float64'),
    'BW_minus_Class': both['BodyweightKg'] - weight_class,
})

# P：记录精度（树按阈值切数值，学不到「是不是 .5」「是不是整数」）
age2 = (both['Age'] * 2).round()
require(np.allclose(both['Age'] * 2, age2), 'Age 不是 0.5 的整数倍')
bw = both['BodyweightKg']
is_int = np.isclose(bw, bw.round())
is_1dec = np.isclose(bw * 10, (bw * 10).round())
is_half = is_1dec & ~is_int & np.isclose((bw * 10).round() % 10, 5)
pounds = bw / 0.45359237
P = pd.DataFrame({
    'age_half': (age2 % 2).astype('float64'),
    'bw_precision': np.select([is_int, is_half, is_1dec], [0.0, 1.0, 2.0], default=3.0),
    'bw_lb_int': (np.abs(pounds - pounds.round()) < 0.015).astype('float64'),
})

# T：跨地参赛（居住地未知时为 NaN）
abroad = (both['Country'] != both['MeetCountry']).astype('float64').where(both['Country'] != 'Unknown')
state_known = ((both['State'] != 'Unknown') & (both['MeetState'] != 'Unknown')
               & (both['Country'] == both['MeetCountry']))
other_state = (both['State'] != both['MeetState']).astype('float64').where(state_known)
T = pd.DataFrame({'abroad': abroad, 'other_state': other_state})

# M：比赛背景；「比赛」≈ 联盟 × 年份 × 比赛国 × 比赛州（train+test 无标签统计）
meet = (both['Federation'] + '|' + both['MeetYear'].astype(str) + '|'
        + both['MeetCountry'] + '|' + both['MeetState'])


def by_meet(values):
    return values.groupby(meet).transform('mean')


M = pd.DataFrame({
    'meet_n': meet.map(meet.value_counts()).astype('float64'),
    'meet_sexeq_n': both.groupby([meet, both['Sex'], both['Equipment']])['Age'].transform('size').astype('float64'),
    'fed_year_n': both.groupby(['Federation', 'MeetYear'])['Age'].transform('size').astype('float64'),
    'meet_share_other_state': by_meet(other_state),
    'meet_share_abroad': by_meet(abroad),
    'meet_share_age_half': by_meet(P['age_half']),
    'meet_share_equipped': by_meet((both['Equipment'] != 'Raw').astype('float64')),
    'meet_share_male': by_meet((both['Sex'] == 'M').astype('float64')),
    'meet_mean_age': by_meet(both['Age']),
})

BLOCKS = {'P': P, 'T': T, 'M': M}
VARIANTS = {'F0': [], 'P': ['P'], 'T': ['T'], 'M': ['M'], 'PT': ['P', 'T'], 'ALL': ['P', 'T', 'M']}
for name, frame in {'F0': F0_EXTRA, **BLOCKS}.items():
    require(len(frame) == len(both) and frame.index.equals(both.index), f'{name} 行数或索引不符')
    require(not np.isinf(frame.to_numpy(dtype='float64')).any(), f'{name} 含无穷大')

_FRAMES = {}


def frames_for(variant, model):
    """(train, test) 特征表。CatBoost 的类别为字符串；LGB / XGB 为 category（词表来自 train+test，无标签）。"""
    kind = 'str' if model == 'catboost' else 'category'
    if (variant, kind) not in _FRAMES:
        frame = pd.concat([both[RAW_NUM], F0_EXTRA] + [BLOCKS[b] for b in VARIANTS[variant]] + [both[CAT]], axis=1)
        require(frame.columns.is_unique, f'{variant} 列名重复')
        frame[CAT] = frame[CAT].astype(str).astype(object) if kind == 'str' else frame[CAT].astype('category')
        _FRAMES[(variant, kind)] = (frame.iloc[:n_train].reset_index(drop=True),
                                    frame.iloc[n_train:].reset_index(drop=True))
    return _FRAMES[(variant, kind)]


print({v: frames_for(v, 'lgb')[0].shape[1] for v in VARIANTS})
summary_new = pd.concat([P, T, M], axis=1)
display(pd.DataFrame({'train_mean': summary_new.iloc[:n_train].mean(), 'test_mean': summary_new.iloc[n_train:].mean(),
                      'nan_share': summary_new.isna().mean()}).round(4))
