_FRAMES2.clear()                    # 09 的变体表不再需要，释放内存
birth_num = np.floor(both['MeetYear'] - both['Age'] + 1e-9).astype(int)


def fuzzy_count(prefix_key):
    """同前缀、出生年为 b−1、b、b+1 的行数（整数年龄推算的出生年可能差 1）。"""
    exact = prefix_key + '|' + birth_num.astype(str)
    table = exact.value_counts()
    total = exact.map(table).astype('float64')
    for d in (-1, 1):
        total = total + (prefix_key + '|' + (birth_num + d).astype(str)).map(table).fillna(0)
    return total


# H2：运动员生涯（athlete、valid 来自 09 的特征单元；M、T 来自 07；M2 来自 09）
bw = both['BodyweightKg']
H2 = pd.DataFrame({
    'ath_max_meet_states': M2['meet_n_states'].groupby(athlete).transform('max'),
    'ath_mean_meet_states': M2['meet_n_states'].groupby(athlete).transform('mean'),
    'ath_max_meet_n': M['meet_n'].groupby(athlete).transform('max'),
    'ath_max_nat_states': M2['nat_n_states'].groupby(athlete).transform('max'),
    'ath_ever_abroad': T['abroad'].groupby(athlete).transform('max'),
    'ath_n_wclass': n_distinct(both['WeightClassKg'], athlete),
    'ath_bw_range': bw.groupby(athlete).transform('max') - bw.groupby(athlete).transform('min'),
    'ath_first_age': both['Age'].groupby(athlete).transform('min'),
    'ath_n_same_year': both.groupby([athlete, both['MeetYear']])['Age'].transform('size').astype('float64'),
    'ath_share_tested': (both['Tested'] == 'Yes').astype('float64').groupby(athlete).transform('mean'),
    'ath_n_fuzzy': fuzzy_count(sex + '|' + both['Country'] + '|' + both['State'] + '|' + both['Federation']),
    'ath0_n_fuzzy': fuzzy_count(sex + '|' + both['Country'] + '|' + both['State']),
})
H2.loc[~valid] = np.nan

# E：比赛的经验结构
experience = H['ath_years_since_first']
E = pd.DataFrame({
    'meet_mean_ath_n': H['ath_n'].groupby(meet).transform('mean'),
    'meet_mean_ath_exp': experience.groupby(meet).transform('mean'),
    'meet_share_veterans': (experience >= 3).astype('float64').where(experience.notna()).groupby(meet).transform('mean'),
    'meet_share_newcomers': (H['ath_n'] == 1).astype('float64').where(H['ath_n'].notna()).groupby(meet).transform('mean'),
})

ROUND3 = {'H2': H2, 'E': E}
for name, frame in ROUND3.items():
    require(len(frame) == len(both) and frame.index.equals(both.index), f'{name} 行数或索引不符')
    require(not np.isinf(frame.to_numpy(dtype='float64')).any(), f'{name} 含无穷大')
NUMERIC_BLOCKS = {**BLOCKS, **BLOCKS2, **ROUND3}          # P T M（07）、M2 D H S（09）、H2 E（本轮）
R2_BLOCKS = ['P', 'T', 'M', 'H']                          # 09 的 ALL2
EXTRAS = {'lgb': [], 'xgb': ['M2', 'S', 'D'], 'catboost': ['M2', 'C']}   # PM：09 中各模型单独有效的块
VARIANTS3 = {'R2': [], 'H2': ['H2'], 'E': ['E'], 'PM': ['PM']}
SCREEN3 = [v for v in VARIANTS3 if RUN_ONLY is None or v in RUN_ONLY or v == 'R2']
_FRAMES3 = {}


def blocks_for(variant, model):
    out = list(R2_BLOCKS)
    for b in VARIANTS3[variant]:
        out += EXTRAS[model] if b == 'PM' else [b]
    return out


def runs(model, variant):
    """该模型在这个变体上的特征与 R2 不同才需要训练（例如 PM 对 LightGBM 没有变化）。"""
    return variant == 'R2' or blocks_for(variant, model) != blocks_for('R2', model)


def frames3(variant, model):
    """(train, test, 类别列)。"""
    blocks = blocks_for(variant, model)
    kind = 'str' if model == 'catboost' else 'category'
    key = (tuple(blocks), kind)
    if key not in _FRAMES3:
        nums = [both[RAW_NUM], F0_EXTRA] + [NUMERIC_BLOCKS[b] for b in blocks if b in NUMERIC_BLOCKS]
        cats = [both[CAT]] + ([CB_ONLY[b] for b in blocks if b in CB_ONLY] if kind == 'str' else [])
        frame = pd.concat(nums + cats, axis=1)
        require(frame.columns.is_unique, f'{variant} {model} 列名重复')
        cat_cols = [c for part in cats for c in part.columns]
        if kind == 'str':
            frame[cat_cols] = frame[cat_cols].astype(str).astype(object)
        else:
            frame[cat_cols] = frame[cat_cols].astype('category')
        _FRAMES3[key] = (frame.iloc[:n_train].reset_index(drop=True),
                         frame.iloc[n_train:].reset_index(drop=True), cat_cols)
    return _FRAMES3[key]


new_cols = pd.concat([H2, E], axis=1)
display(pd.DataFrame({'train_mean': new_cols.iloc[:n_train].mean(), 'test_mean': new_cols.iloc[n_train:].mean(),
                      'nan_share': new_cols.isna().mean()}).round(4))
print({f'{v}/{m}': frames3(v, m)[0].shape[1] for v in VARIANTS3 for m in MODELS if runs(m, v)})
