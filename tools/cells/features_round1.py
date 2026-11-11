both = pd.concat([train[RAW_FEATURES], test[RAW_FEATURES]], ignore_index=True)

# Variables initiales et catégorie de poids: weight-class features from ensemble à dix plis et plusieurs graines
weight_class = pd.to_numeric(both['WeightClassKg'].str.replace('+', '', regex=False), errors='coerce')
WEIGHT_CLASS_FEATURES = pd.DataFrame({
    'WeightClass_num': weight_class,
    'Is_SHW': both['WeightClassKg'].str.endswith('+').astype('float64'),
    'BW_minus_Class': both['BodyweightKg'] - weight_class,
})

# Précision des enregistrements: explicit indicators for recording precision, such as half-years and integers.
age2 = (both['Age'] * 2).round()
require(np.allclose(both['Age'] * 2, age2), 'Age is not a multiple of 0.5')
bw = both['BodyweightKg']
is_int = np.isclose(bw, bw.round())
is_1dec = np.isclose(bw * 10, (bw * 10).round())
is_half = is_1dec & ~is_int & np.isclose((bw * 10).round() % 10, 5)
pounds = bw / 0.45359237
RECORDING_PRECISION_FEATURES = pd.DataFrame({
    'age_half': (age2 % 2).astype('float64'),
    'bw_precision': np.select([is_int, is_half, is_1dec], [0.0, 1.0, 2.0], default=3.0),
    'bw_lb_int': (np.abs(pounds - pounds.round()) < 0.015).astype('float64'),
})

# Déplacements: travel indicators; NaN when residence is unknown.
abroad = (both['Country'] != both['MeetCountry']).astype('float64').where(both['Country'] != 'Unknown')
state_known = ((both['State'] != 'Unknown') & (both['MeetState'] != 'Unknown')
               & (both['Country'] == both['MeetCountry']))
other_state = (both['State'] != both['MeetState']).astype('float64').where(state_known)
TRAVEL_FEATURES = pd.DataFrame({'abroad': abroad, 'other_state': other_state})

# Contexte de compétition: meet proxy = federation / year / country / state; unlabeled train+test statistics.
meet = (both['Federation'] + '|' + both['MeetYear'].astype(str) + '|'
        + both['MeetCountry'] + '|' + both['MeetState'])


def by_meet(values):
    return values.groupby(meet).transform('mean')


MEET_CONTEXT_FEATURES = pd.DataFrame({
    'meet_n': meet.map(meet.value_counts()).astype('float64'),
    'meet_sexeq_n': both.groupby([meet, both['Sex'], both['Equipment']])['Age'].transform('size').astype('float64'),
    'fed_year_n': both.groupby(['Federation', 'MeetYear'])['Age'].transform('size').astype('float64'),
    'meet_share_other_state': by_meet(other_state),
    'meet_share_abroad': by_meet(abroad),
    'meet_share_age_half': by_meet(RECORDING_PRECISION_FEATURES['age_half']),
    'meet_share_equipped': by_meet((both['Equipment'] != 'Raw').astype('float64')),
    'meet_share_male': by_meet((both['Sex'] == 'M').astype('float64')),
    'meet_mean_age': by_meet(both['Age']),
})

CONTEXT_FEATURE_BLOCKS = {BLOCK_RECORDING_PRECISION: RECORDING_PRECISION_FEATURES, BLOCK_TRAVEL: TRAVEL_FEATURES, BLOCK_MEET_CONTEXT: MEET_CONTEXT_FEATURES}
CONTEXT_VARIANTS = {VARIANT_WEIGHT_CLASS_BASELINE: [], BLOCK_RECORDING_PRECISION: [BLOCK_RECORDING_PRECISION], BLOCK_TRAVEL: [BLOCK_TRAVEL], BLOCK_MEET_CONTEXT: [BLOCK_MEET_CONTEXT], VARIANT_PRECISION_AND_TRAVEL: [BLOCK_RECORDING_PRECISION, BLOCK_TRAVEL], VARIANT_CONTEXT_COMBINED: [BLOCK_RECORDING_PRECISION, BLOCK_TRAVEL, BLOCK_MEET_CONTEXT]}
for name, frame in {VARIANT_WEIGHT_CLASS_BASELINE: WEIGHT_CLASS_FEATURES, **CONTEXT_FEATURE_BLOCKS}.items():
    require(len(frame) == len(both) and frame.index.equals(both.index), f'{name} row count or index mismatch')
    require(not np.isinf(frame.to_numpy(dtype='float64')).any(), f'{name} contains infinity')

_CONTEXT_FRAMES = {}


def context_feature_frames(variant, model):
    """(train, test) features. String categories for CatBoost; category dtype for LGB / XGB, using the unlabeled joint vocabulary."""
    kind = 'str' if model == MODEL_CATBOOST_ABSOLUTE else 'category'
    if (variant, kind) not in _CONTEXT_FRAMES:
        frame = pd.concat([both[RAW_NUM], WEIGHT_CLASS_FEATURES] + [CONTEXT_FEATURE_BLOCKS[b] for b in CONTEXT_VARIANTS[variant]] + [both[CAT]], axis=1)
        require(frame.columns.is_unique, f'{variant} duplicate column names')
        frame[CAT] = frame[CAT].astype(str).astype(object) if kind == 'str' else frame[CAT].astype('category')
        _CONTEXT_FRAMES[(variant, kind)] = (frame.iloc[:n_train].reset_index(drop=True),
                                    frame.iloc[n_train:].reset_index(drop=True))
    return _CONTEXT_FRAMES[(variant, kind)]


print({v: context_feature_frames(v, MODEL_LIGHTGBM)[0].shape[1] for v in CONTEXT_VARIANTS})
summary_new = pd.concat([RECORDING_PRECISION_FEATURES, TRAVEL_FEATURES, MEET_CONTEXT_FEATURES], axis=1)
display(pd.DataFrame({'train_mean': summary_new.iloc[:n_train].mean(), 'test_mean': summary_new.iloc[n_train:].mean(),
                      'nan_share': summary_new.isna().mean()}).round(4))
