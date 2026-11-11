_CONTEXT_FRAMES.clear()                     # Release cached feature frames from experiment 07.
CORE9 = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']
sex, equipment = both['Sex'], both['Equipment']
state = both['State'].where(both['State'] != 'Unknown')
country = both['Country'].where(both['Country'] != 'Unknown')
national = both['Federation'] + '|' + both['MeetYear'].astype(str) + '|' + both['MeetCountry']


def counts(key):
    return key.map(key.value_counts()).astype('float64')


def n_distinct(values, key):
    return values.groupby(key).transform('nunique').astype('float64')


# Contexte de compétition enrichi: additional meet context; meet and by_meet come from experiment 07.
EXTENDED_MEET_CONTEXT_FEATURES = pd.DataFrame({
    'meet_n_states': n_distinct(state, meet),
    'meet_n_countries': n_distinct(country, meet),
    'meet_n_wclass': n_distinct(both['WeightClassKg'], meet),
    'meet_share_tested': by_meet((both['Tested'] == 'Yes').astype('float64')),
    'meet_share_raw': by_meet((equipment == 'Raw').astype('float64')),
    'meet_share_multi': by_meet((equipment == 'Multi-ply').astype('float64')),
    'meet_share_bw2dec': by_meet((RECORDING_PRECISION_FEATURES['bw_precision'] == 3).astype('float64')),
    'meet_share_junior': by_meet((both['Age'] < 24).astype('float64')),
    'meet_share_master': by_meet((both['Age'] >= 40).astype('float64')),
    'meet_age_std': both['Age'].groupby(meet).transform('std'),
    'meet_sex_bw_mean': both['BodyweightKg'].groupby([meet, sex]).transform('mean'),
    'nat_n': counts(national),
    'nat_n_meetstates': n_distinct(both['MeetState'], national),
    'nat_n_states': n_distinct(state, national),
})

# Effectifs et rangs des divisions: division context
division = meet + '|' + sex + '|' + equipment + '|' + both['WeightClassKg']
age_division = meet + '|' + sex + '|' + equipment + '|' + both['AgeClass']
exact_key = pd.Series(pd.util.hash_pandas_object(both[RAW_FEATURES], index=False).to_numpy())
core_key = pd.Series(pd.util.hash_pandas_object(both[CORE9], index=False).to_numpy())
DIVISION_FEATURES = pd.DataFrame({
    'div_n': counts(division),
    'div_bw_rank': both['BodyweightKg'].groupby(division).rank(pct=True),
    'div_age_rank': both['Age'].groupby(division).rank(pct=True),
    'agediv_n': counts(age_division),
    'dup_exact_n': counts(exact_key) - 1,
    'dup_core_n': counts(core_key) - 1,
})

# Historique approché du sportif: athlete proxy history from floor(MeetYear - Age); birth-year estimates may differ by one.
valid = (both['Country'] != 'Unknown').to_numpy()
birth = np.floor(both['MeetYear'] - both['Age'] + 1e-9).astype(int).astype(str)
athlete = sex + '|' + both['Country'] + '|' + both['State'] + '|' + both['Federation'] + '|' + birth
athlete0 = sex + '|' + both['Country'] + '|' + both['State'] + '|' + birth
year = both['MeetYear'].astype('float64')
ATHLETE_HISTORY_FEATURES = pd.DataFrame({
    'ath_n': counts(athlete),
    'ath_n_years': n_distinct(year, athlete),
    'ath_years_since_first': year - year.groupby(athlete).transform('min'),
    'ath_years_to_last': year.groupby(athlete).transform('max') - year,
    'ath_n_equipment': n_distinct(equipment, athlete),
    'ath_bw_delta': both['BodyweightKg'] - both['BodyweightKg'].groupby(athlete).transform('mean'),
    'ath0_n': counts(athlete0),
    'ath0_n_feds': n_distinct(both['Federation'], athlete0),
})
ATHLETE_HISTORY_FEATURES.loc[~valid] = np.nan


# Facteur DOTS: DOTS scale (TotalKg = DOTS score * dots_scale)
def dots_scale(sex_values, bw_values):
    s = np.asarray(sex_values).astype(str)
    bw = np.asarray(bw_values, dtype='float64')
    m, f = np.clip(bw, 40, 210), np.clip(bw, 40, 150)
    poly_m = -1.0930e-6 * m**4 + 7.391293e-4 * m**3 - 0.1918759221 * m**2 + 24.0900756 * m - 307.75076
    poly_f = -1.0706e-6 * f**4 + 5.158568e-4 * f**3 - 0.1126655495 * f**2 + 13.6175032 * f - 57.96288
    return np.where(s == 'M', poly_m, np.where(s == 'F', poly_f, 0.5 * (poly_m + poly_f))) / 500.0


DOTS_SCALE_FEATURE = pd.DataFrame({'dots_scale': dots_scale(sex, both['BodyweightKg'])})

# Identifiants approchés: high-cardinality IDs for CatBoost ordered target statistics.
PROXY_IDENTIFIER_FEATURES = pd.DataFrame({'meet_id': meet, 'athlete_id': (athlete + '|' + equipment).where(valid, 'unknown')})

HISTORY_FEATURE_BLOCKS = {BLOCK_EXTENDED_MEET_CONTEXT: EXTENDED_MEET_CONTEXT_FEATURES, BLOCK_DIVISION_CONTEXT: DIVISION_FEATURES, BLOCK_ATHLETE_HISTORY: ATHLETE_HISTORY_FEATURES, BLOCK_DOTS_SCALE: DOTS_SCALE_FEATURE}
CATBOOST_CATEGORY_BLOCKS = {BLOCK_PROXY_IDENTIFIERS: PROXY_IDENTIFIER_FEATURES}
for name, frame in {**HISTORY_FEATURE_BLOCKS, **CATBOOST_CATEGORY_BLOCKS}.items():
    require(len(frame) == len(both) and frame.index.equals(both.index), f'{name} row count or index mismatch')
for name, frame in HISTORY_FEATURE_BLOCKS.items():
    require(not np.isinf(frame.to_numpy(dtype='float64')).any(), f'{name} contains infinity')
require(DOTS_SCALE_FEATURE['dots_scale'].min() > 0.3, 'Invalid DOTS scale')

CONTEXT_BASE_BLOCKS = [BLOCK_RECORDING_PRECISION, BLOCK_TRAVEL, BLOCK_MEET_CONTEXT]
HISTORY_VARIANTS = {VARIANT_CONTEXT_BASELINE: [], BLOCK_EXTENDED_MEET_CONTEXT: [BLOCK_EXTENDED_MEET_CONTEXT], BLOCK_DIVISION_CONTEXT: [BLOCK_DIVISION_CONTEXT], BLOCK_ATHLETE_HISTORY: [BLOCK_ATHLETE_HISTORY], BLOCK_DOTS_SCALE: [BLOCK_DOTS_SCALE], BLOCK_PROXY_IDENTIFIERS: [BLOCK_PROXY_IDENTIFIERS]}
MODELS_FOR = {BLOCK_PROXY_IDENTIFIERS: [MODEL_CATBOOST_ABSOLUTE]}      # Identifiants approchés changes CatBoost only; retain Référence avec contexte for LightGBM / XGBoost in the blend.
SCREEN = [v for v in HISTORY_VARIANTS if RUN_ONLY is None or v in RUN_ONLY or v == VARIANT_CONTEXT_BASELINE]
_HISTORY_FRAMES = {}


def models_for(variant):
    return MODELS_FOR.get(variant, MODELS)


def history_feature_frames(variant, model):
    """Retourner train, test et colonnes catégorielles : variables initiales, catégorie de poids, précision, déplacements et contexte ; les identifiants approchés sont ajoutés à CatBoost uniquement."""
    blocks = HISTORY_VARIANTS[variant]
    kind = 'str' if model == MODEL_CATBOOST_ABSOLUTE else 'category'
    key = (tuple(blocks), kind)
    if key not in _HISTORY_FRAMES:
        nums = ([both[RAW_NUM], WEIGHT_CLASS_FEATURES] + [CONTEXT_FEATURE_BLOCKS[b] for b in CONTEXT_BASE_BLOCKS]
                + [HISTORY_FEATURE_BLOCKS[b] for b in blocks if b in HISTORY_FEATURE_BLOCKS])
        cats = [both[CAT]] + ([CATBOOST_CATEGORY_BLOCKS[b] for b in blocks if b in CATBOOST_CATEGORY_BLOCKS] if kind == 'str' else [])
        frame = pd.concat(nums + cats, axis=1)
        require(frame.columns.is_unique, f'{variant} duplicate column names')
        cat_cols = [c for part in cats for c in part.columns]
        if kind == 'str':
            frame[cat_cols] = frame[cat_cols].astype(str).astype(object)
        else:
            frame[cat_cols] = frame[cat_cols].astype('category')
        _HISTORY_FRAMES[key] = (frame.iloc[:n_train].reset_index(drop=True),
                         frame.iloc[n_train:].reset_index(drop=True), cat_cols)
    return _HISTORY_FRAMES[key]


new_cols = pd.concat([EXTENDED_MEET_CONTEXT_FEATURES, DIVISION_FEATURES, ATHLETE_HISTORY_FEATURES, DOTS_SCALE_FEATURE], axis=1)
display(pd.DataFrame({'train_mean': new_cols.iloc[:n_train].mean(), 'test_mean': new_cols.iloc[n_train:].mean(),
                      'nan_share': new_cols.isna().mean()}).round(4))
print('Cardinality of C: ', {c: int(PROXY_IDENTIFIER_FEATURES[c].nunique()) for c in PROXY_IDENTIFIER_FEATURES.columns})
print('Columns by variant (LightGBM / XGBoost): ', {v: history_feature_frames(v, MODEL_LIGHTGBM)[0].shape[1] for v in HISTORY_VARIANTS if v != BLOCK_PROXY_IDENTIFIERS},
      '|CatBoost C: ', history_feature_frames(BLOCK_PROXY_IDENTIFIERS, MODEL_CATBOOST_ABSOLUTE)[0].shape[1])
