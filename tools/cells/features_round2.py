_FRAMES.clear()                     # 07 的变体表不再需要，释放内存
CORE9 = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']
sex, equipment = both['Sex'], both['Equipment']
state = both['State'].where(both['State'] != 'Unknown')
country = both['Country'].where(both['Country'] != 'Unknown')
national = both['Federation'] + '|' + both['MeetYear'].astype(str) + '|' + both['MeetCountry']


def counts(key):
    return key.map(key.value_counts()).astype('float64')


def n_distinct(values, key):
    return values.groupby(key).transform('nunique').astype('float64')


# M2：比赛背景加强（meet 与 by_meet 来自 07 的特征单元）
M2 = pd.DataFrame({
    'meet_n_states': n_distinct(state, meet),
    'meet_n_countries': n_distinct(country, meet),
    'meet_n_wclass': n_distinct(both['WeightClassKg'], meet),
    'meet_share_tested': by_meet((both['Tested'] == 'Yes').astype('float64')),
    'meet_share_raw': by_meet((equipment == 'Raw').astype('float64')),
    'meet_share_multi': by_meet((equipment == 'Multi-ply').astype('float64')),
    'meet_share_bw2dec': by_meet((P['bw_precision'] == 3).astype('float64')),
    'meet_share_junior': by_meet((both['Age'] < 24).astype('float64')),
    'meet_share_master': by_meet((both['Age'] >= 40).astype('float64')),
    'meet_age_std': both['Age'].groupby(meet).transform('std'),
    'meet_sex_bw_mean': both['BodyweightKg'].groupby([meet, sex]).transform('mean'),
    'nat_n': counts(national),
    'nat_n_meetstates': n_distinct(both['MeetState'], national),
    'nat_n_states': n_distinct(state, national),
})

# D：组别背景
division = meet + '|' + sex + '|' + equipment + '|' + both['WeightClassKg']
age_division = meet + '|' + sex + '|' + equipment + '|' + both['AgeClass']
exact_key = pd.Series(pd.util.hash_pandas_object(both[RAW_FEATURES], index=False).to_numpy())
core_key = pd.Series(pd.util.hash_pandas_object(both[CORE9], index=False).to_numpy())
D = pd.DataFrame({
    'div_n': counts(division),
    'div_bw_rank': both['BodyweightKg'].groupby(division).rank(pct=True),
    'div_age_rank': both['Age'].groupby(division).rank(pct=True),
    'agediv_n': counts(age_division),
    'dup_exact_n': counts(exact_key) - 1,
    'dup_core_n': counts(core_key) - 1,
})

# H：近似同一运动员的历史（出生年 = floor(MeetYear − Age)：x.5 年龄时恰为出生年，整数年龄可能差 1）
valid = (both['Country'] != 'Unknown').to_numpy()
birth = np.floor(both['MeetYear'] - both['Age'] + 1e-9).astype(int).astype(str)
athlete = sex + '|' + both['Country'] + '|' + both['State'] + '|' + both['Federation'] + '|' + birth
athlete0 = sex + '|' + both['Country'] + '|' + both['State'] + '|' + birth
year = both['MeetYear'].astype('float64')
H = pd.DataFrame({
    'ath_n': counts(athlete),
    'ath_n_years': n_distinct(year, athlete),
    'ath_years_since_first': year - year.groupby(athlete).transform('min'),
    'ath_years_to_last': year.groupby(athlete).transform('max') - year,
    'ath_n_equipment': n_distinct(equipment, athlete),
    'ath_bw_delta': both['BodyweightKg'] - both['BodyweightKg'].groupby(athlete).transform('mean'),
    'ath0_n': counts(athlete0),
    'ath0_n_feds': n_distinct(both['Federation'], athlete0),
})
H.loc[~valid] = np.nan


# S：DOTS 系数（TotalKg = DOTS 分 × dots_scale）
def dots_scale(sex_values, bw_values):
    s = np.asarray(sex_values).astype(str)
    bw = np.asarray(bw_values, dtype='float64')
    m, f = np.clip(bw, 40, 210), np.clip(bw, 40, 150)
    poly_m = -1.0930e-6 * m**4 + 7.391293e-4 * m**3 - 0.1918759221 * m**2 + 24.0900756 * m - 307.75076
    poly_f = -1.0706e-6 * f**4 + 5.158568e-4 * f**3 - 0.1126655495 * f**2 + 13.6175032 * f - 57.96288
    return np.where(s == 'M', poly_m, np.where(s == 'F', poly_f, 0.5 * (poly_m + poly_f))) / 500.0


S = pd.DataFrame({'dots_scale': dots_scale(sex, both['BodyweightKg'])})

# C：只给 CatBoost 的高基数 ID（有序目标统计编码）
C = pd.DataFrame({'meet_id': meet, 'athlete_id': (athlete + '|' + equipment).where(valid, 'unknown')})

BLOCKS2 = {'M2': M2, 'D': D, 'H': H, 'S': S}
CB_ONLY = {'C': C}
for name, frame in {**BLOCKS2, **CB_ONLY}.items():
    require(len(frame) == len(both) and frame.index.equals(both.index), f'{name} 行数或索引不符')
for name, frame in BLOCKS2.items():
    require(not np.isinf(frame.to_numpy(dtype='float64')).any(), f'{name} 含无穷大')
require(S['dots_scale'].min() > 0.3, 'DOTS 系数异常')

R1_BLOCKS = ['P', 'T', 'M']
VARIANT_BLOCKS = {'R1': [], 'M2': ['M2'], 'D': ['D'], 'H': ['H'], 'S': ['S'], 'C': ['C']}
MODELS_FOR = {'C': ['catboost']}      # C 只改变 CatBoost；融合时 LightGBM / XGBoost 用 R1
SCREEN = [v for v in VARIANT_BLOCKS if RUN_ONLY is None or v in RUN_ONLY or v == 'R1']
_FRAMES2 = {}


def models_for(variant):
    return MODELS_FOR.get(variant, MODELS)


def frames2(variant, model):
    """(train, test, 类别列)。R1 = F0 + P + T + M；C 块只加给 CatBoost。"""
    blocks = VARIANT_BLOCKS[variant]
    kind = 'str' if model == 'catboost' else 'category'
    key = (tuple(blocks), kind)
    if key not in _FRAMES2:
        nums = ([both[RAW_NUM], F0_EXTRA] + [BLOCKS[b] for b in R1_BLOCKS]
                + [BLOCKS2[b] for b in blocks if b in BLOCKS2])
        cats = [both[CAT]] + ([CB_ONLY[b] for b in blocks if b in CB_ONLY] if kind == 'str' else [])
        frame = pd.concat(nums + cats, axis=1)
        require(frame.columns.is_unique, f'{variant} 列名重复')
        cat_cols = [c for part in cats for c in part.columns]
        if kind == 'str':
            frame[cat_cols] = frame[cat_cols].astype(str).astype(object)
        else:
            frame[cat_cols] = frame[cat_cols].astype('category')
        _FRAMES2[key] = (frame.iloc[:n_train].reset_index(drop=True),
                         frame.iloc[n_train:].reset_index(drop=True), cat_cols)
    return _FRAMES2[key]


new_cols = pd.concat([M2, D, H, S], axis=1)
display(pd.DataFrame({'train_mean': new_cols.iloc[:n_train].mean(), 'test_mean': new_cols.iloc[n_train:].mean(),
                      'nan_share': new_cols.isna().mean()}).round(4))
print('C 块类别数：', {c: int(C[c].nunique()) for c in C.columns})
print('各变体列数（LightGBM / XGBoost）：', {v: frames2(v, 'lgb')[0].shape[1] for v in VARIANT_BLOCKS if v != 'C'},
      '｜CatBoost C：', frames2('C', 'catboost')[0].shape[1])
