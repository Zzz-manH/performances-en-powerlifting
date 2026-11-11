_HISTORY_FRAMES.clear()                    # Release cached feature frames from experiment 09.
birth_num = np.floor(both['MeetYear'] - both['Age'] + 1e-9).astype(int)


def fuzzy_count(prefix_key):
    """Count rows with the same prefix and birth years b-1, b, b+1."""
    exact = prefix_key + '|' + birth_num.astype(str)
    table = exact.value_counts()
    total = exact.map(table).astype('float64')
    for d in (-1, 1):
        total = total + (prefix_key + '|' + (birth_num + d).astype(str)).map(table).fillna(0)
    return total


# Carrière approchée du sportif: career features; athlete/valid and Contexte de compétition enrichi from 09, Contexte de compétition/Déplacements from 07.
bw = both['BodyweightKg']
ATHLETE_CAREER_FEATURES = pd.DataFrame({
    'ath_max_meet_states': EXTENDED_MEET_CONTEXT_FEATURES['meet_n_states'].groupby(athlete).transform('max'),
    'ath_mean_meet_states': EXTENDED_MEET_CONTEXT_FEATURES['meet_n_states'].groupby(athlete).transform('mean'),
    'ath_max_meet_n': MEET_CONTEXT_FEATURES['meet_n'].groupby(athlete).transform('max'),
    'ath_max_nat_states': EXTENDED_MEET_CONTEXT_FEATURES['nat_n_states'].groupby(athlete).transform('max'),
    'ath_ever_abroad': TRAVEL_FEATURES['abroad'].groupby(athlete).transform('max'),
    'ath_n_wclass': n_distinct(both['WeightClassKg'], athlete),
    'ath_bw_range': bw.groupby(athlete).transform('max') - bw.groupby(athlete).transform('min'),
    'ath_first_age': both['Age'].groupby(athlete).transform('min'),
    'ath_n_same_year': both.groupby([athlete, both['MeetYear']])['Age'].transform('size').astype('float64'),
    'ath_share_tested': (both['Tested'] == 'Yes').astype('float64').groupby(athlete).transform('mean'),
    'ath_n_fuzzy': fuzzy_count(sex + '|' + both['Country'] + '|' + both['State'] + '|' + both['Federation']),
    'ath0_n_fuzzy': fuzzy_count(sex + '|' + both['Country'] + '|' + both['State']),
})
ATHLETE_CAREER_FEATURES.loc[~valid] = np.nan

# Expérience du plateau: experience profile of a meet
experience = ATHLETE_HISTORY_FEATURES['ath_years_since_first']
MEET_EXPERIENCE_FEATURES = pd.DataFrame({
    'meet_mean_ath_n': ATHLETE_HISTORY_FEATURES['ath_n'].groupby(meet).transform('mean'),
    'meet_mean_ath_exp': experience.groupby(meet).transform('mean'),
    'meet_share_veterans': (experience >= 3).astype('float64').where(experience.notna()).groupby(meet).transform('mean'),
    'meet_share_newcomers': (ATHLETE_HISTORY_FEATURES['ath_n'] == 1).astype('float64').where(ATHLETE_HISTORY_FEATURES['ath_n'].notna()).groupby(meet).transform('mean'),
})

CAREER_FEATURE_BLOCKS = {BLOCK_ATHLETE_CAREER: ATHLETE_CAREER_FEATURES, BLOCK_MEET_EXPERIENCE: MEET_EXPERIENCE_FEATURES}
for name, frame in CAREER_FEATURE_BLOCKS.items():
    require(len(frame) == len(both) and frame.index.equals(both.index), f'{name} row count or index mismatch')
    require(not np.isinf(frame.to_numpy(dtype='float64')).any(), f'{name} contains infinity')
NUMERIC_BLOCKS = {**CONTEXT_FEATURE_BLOCKS, **HISTORY_FEATURE_BLOCKS, **CAREER_FEATURE_BLOCKS}          # Précision des enregistrements Déplacements Contexte de compétition (07), Contexte de compétition enrichi Effectifs et rangs des divisions Historique approché du sportif Facteur DOTS (09), Carrière approchée du sportif Expérience du plateau (this round)
HISTORY_BASE_BLOCKS = [BLOCK_RECORDING_PRECISION, BLOCK_TRAVEL, BLOCK_MEET_CONTEXT, BLOCK_ATHLETE_HISTORY]                          # Contexte et historique du sportif from experiment 09
MODEL_SPECIFIC_BLOCKS = {MODEL_LIGHTGBM: [], MODEL_XGBOOST: [BLOCK_EXTENDED_MEET_CONTEXT, BLOCK_DOTS_SCALE, BLOCK_DIVISION_CONTEXT], MODEL_CATBOOST_ABSOLUTE: [BLOCK_EXTENDED_MEET_CONTEXT, BLOCK_PROXY_IDENTIFIERS]}   # Ajouts propres à chaque modèle: blocks selected separately by model in experiment 09.
CAREER_VARIANTS = {VARIANT_HISTORY_BASELINE: [], BLOCK_ATHLETE_CAREER: [BLOCK_ATHLETE_CAREER], BLOCK_MEET_EXPERIENCE: [BLOCK_MEET_EXPERIENCE], BLOCK_MODEL_SPECIFIC: [BLOCK_MODEL_SPECIFIC]}
SCREEN3 = [v for v in CAREER_VARIANTS if RUN_ONLY is None or v in RUN_ONLY or v == VARIANT_HISTORY_BASELINE]
_CAREER_FRAMES = {}


def feature_blocks_for_model(variant, model):
    out = list(HISTORY_BASE_BLOCKS)
    for b in CAREER_VARIANTS[variant]:
        out += MODEL_SPECIFIC_BLOCKS[model] if b == BLOCK_MODEL_SPECIFIC else [b]
    return out


def runs(model, variant):
    """Entraîner seulement si les variables diffèrent de la référence avec historique ; les ajouts propres au modèle laissent LightGBM inchangé."""
    return variant == VARIANT_HISTORY_BASELINE or feature_blocks_for_model(variant, model) != feature_blocks_for_model(VARIANT_HISTORY_BASELINE, model)


def career_feature_frames(variant, model):
    """(train, test, categorical columns)."""
    blocks = feature_blocks_for_model(variant, model)
    kind = 'str' if model == MODEL_CATBOOST_ABSOLUTE else 'category'
    key = (tuple(blocks), kind)
    if key not in _CAREER_FRAMES:
        nums = [both[RAW_NUM], WEIGHT_CLASS_FEATURES] + [NUMERIC_BLOCKS[b] for b in blocks if b in NUMERIC_BLOCKS]
        cats = [both[CAT]] + ([CATBOOST_CATEGORY_BLOCKS[b] for b in blocks if b in CATBOOST_CATEGORY_BLOCKS] if kind == 'str' else [])
        frame = pd.concat(nums + cats, axis=1)
        require(frame.columns.is_unique, f'{variant} {model} duplicate column names')
        cat_cols = [c for part in cats for c in part.columns]
        if kind == 'str':
            frame[cat_cols] = frame[cat_cols].astype(str).astype(object)
        else:
            frame[cat_cols] = frame[cat_cols].astype('category')
        _CAREER_FRAMES[key] = (frame.iloc[:n_train].reset_index(drop=True),
                         frame.iloc[n_train:].reset_index(drop=True), cat_cols)
    return _CAREER_FRAMES[key]


new_cols = pd.concat([ATHLETE_CAREER_FEATURES, MEET_EXPERIENCE_FEATURES], axis=1)
display(pd.DataFrame({'train_mean': new_cols.iloc[:n_train].mean(), 'test_mean': new_cols.iloc[n_train:].mean(),
                      'nan_share': new_cols.isna().mean()}).round(4))
print({f'{v}/{m}': career_feature_frames(v, m)[0].shape[1] for v in CAREER_VARIANTS for m in MODELS if runs(m, v)})
