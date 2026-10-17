N_ALL = len(both)
TEST_ROWS = np.arange(n_train, N_ALL)
REL_ALL = np.full(N_ALL, np.nan)
REL_ALL[:n_train] = y / dots_scale(train['Sex'], train['BodyweightKg'])    # 相对力量（DOTS 分）
TOTAL_ALL = np.full(N_ALL, np.nan)
TOTAL_ALL[:n_train] = y
BIRTH_ALL = pd.Series(np.floor(both['MeetYear'] - both['Age'] + 1e-9).astype(int).to_numpy())
YEAR_ALL = pd.Series(both['MeetYear'].astype(int).to_numpy())
BW_ALL = both['BodyweightKg'].to_numpy(dtype='float64')
VALID_ALL = (both['Country'] != 'Unknown').to_numpy()


def text_key_all(columns):
    out = both[columns[0]].astype(str)
    for c in columns[1:]:
        out = out + '|' + both[c].astype(str)
    return out.reset_index(drop=True)


PFX_ALL = {'A0': text_key_all(['Sex', 'Country', 'State']),
           'A1': text_key_all(['Sex', 'Country', 'State', 'Federation']),
           'A2': text_key_all(['Sex', 'Country', 'State', 'Federation', 'Equipment'])}
KB_SHIFT = {k: {d: p + '|' + (BIRTH_ALL + d).astype(str) for d in (-1, 0, 1)} for k, p in PFX_ALL.items()}
KB_A1 = KB_SHIFT['A1'][0]
KBY_A1 = KB_A1 + '|' + YEAR_ALL.astype(str)
DECAY_GAPS = range(-8, 9)
KBY_SHIFT = {d: KB_A1 + '|' + (YEAR_ALL + d).astype(str) for d in DECAY_GAPS}
MEET_ALL = {'meet': meet.reset_index(drop=True).astype(str),
            'div': text_key_all(['Federation', 'MeetYear', 'MeetCountry', 'MeetState', 'Sex', 'Equipment'])}
EXACT_ALL = pd.Series(pd.util.hash_pandas_object(both[RAW_FEATURES], index=False).to_numpy()).astype(str)
CORE_ALL = pd.Series(pd.util.hash_pandas_object(both[CORE9], index=False).to_numpy()).astype(str)


def agg(keys, values, ref, funcs=('sum', 'count', 'max')):
    frame = pd.DataFrame({'k': keys.iloc[ref].to_numpy(), 'v': values[ref]})
    return frame.groupby('k')['v'].agg(list(funcs))


def lookup(table, column, keys, fill):
    return keys.map(table[column]).fillna(fill).to_numpy(dtype='float64')


def athlete_block(ref, query):
    """ref：有成绩的参考行（训练行号）；query：要算特征的行（both 中的行号，不在 ref 里）。"""
    cols, m = {}, len(query)
    for k in ['A0', 'A1', 'A2']:
        table = agg(KB_SHIFT[k][0], REL_ALL, ref)
        bw_table = agg(KB_SHIFT[k][0], BW_ALL, ref, ('sum',))
        s, n, bws, mx = np.zeros(m), np.zeros(m), np.zeros(m), np.full(m, -np.inf)
        for d in (-1, 0, 1):                                       # 出生年 ±1
            kq = KB_SHIFT[k][d].iloc[query]
            s, n = s + lookup(table, 'sum', kq, 0), n + lookup(table, 'count', kq, 0)
            mx = np.maximum(mx, lookup(table, 'max', kq, -np.inf))
            bws = bws + lookup(bw_table, 'sum', kq, 0)
        has = n > 0
        cols[f'as_{k}_n'] = n
        cols[f'as_{k}_mean'] = np.where(has, s / np.maximum(n, 1), np.nan)
        cols[f'as_{k}_max'] = np.where(has, mx, np.nan)
        if k == 'A1':
            cols['as_A1_bw_diff'] = np.where(has, BW_ALL[query] - bws / np.maximum(n, 1), np.nan)
    table = agg(KBY_A1, REL_ALL, ref, ('sum', 'count'))          # 时间衰减与年份差（A1，精确出生年）
    ws, wn, gap_sum, gap_n, gap_min = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m), np.full(m, np.inf)
    for d in DECAY_GAPS:
        kq = KBY_SHIFT[d].iloc[query]
        sd, nd = lookup(table, 'sum', kq, 0), lookup(table, 'count', kq, 0)
        w = float(np.exp(-abs(d) / 3.0))
        ws, wn = ws + w * sd, wn + w * nd
        gap_sum, gap_n = gap_sum + abs(d) * nd, gap_n + nd
        gap_min = np.where(nd > 0, np.minimum(gap_min, abs(d)), gap_min)
    cols['as_A1_decay'] = np.where(wn > 0, ws / np.maximum(wn, 1e-12), np.nan)
    cols['as_A1_gap_mean'] = np.where(gap_n > 0, gap_sum / np.maximum(gap_n, 1), np.nan)
    cols['as_A1_gap_min'] = np.where(np.isfinite(gap_min), gap_min, np.nan)
    cols['as_same_equipment_share'] = np.where(cols['as_A1_n'] > 0,
                                               cols['as_A2_n'] / np.maximum(cols['as_A1_n'], 1), np.nan)
    frame = pd.DataFrame(cols)
    frame.loc[~VALID_ALL[query]] = np.nan                           # 国家未知时不做运动员匹配
    for name, keys in [('tw_exact', EXACT_ALL), ('tw_core', CORE_ALL)]:
        t = agg(keys, TOTAL_ALL, ref, ('sum', 'count'))
        kq = keys.iloc[query]
        s, n = lookup(t, 'sum', kq, 0), lookup(t, 'count', kq, 0)
        frame[f'{name}_total'] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        frame[f'{name}_n'] = n
    return frame


def meet_block(ref, query):
    cols = {}
    for name, keys in MEET_ALL.items():
        t = agg(keys, REL_ALL, ref)
        kq = keys.iloc[query]
        s, n, mx = lookup(t, 'sum', kq, 0), lookup(t, 'count', kq, 0), lookup(t, 'max', kq, np.nan)
        cols[f'ms_{name}_n'] = n
        cols[f'ms_{name}_mean'] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        cols[f'ms_{name}_max'] = np.where(n > 0, mx, np.nan)
    return pd.DataFrame(cols)


BLOCK_FUNCS = {'AS': athlete_block, 'MS': meet_block}
_FOLD_FEATS = {}


def nested_features(block, fold):
    """(训练行特征 n_train 行, 测试行特征)：外层训练部分用内层折外，外层验证行和测试行用整个训练部分。"""
    key = (block, fold)
    if key not in _FOLD_FEATS:
        fn = BLOCK_FUNCS[block]
        outer_train, outer_valid = np.flatnonzero(FOLD_IDS != fold), np.flatnonzero(FOLD_IDS == fold)
        parts = []
        inner = KFold(INNER_FEATURE_FOLDS, shuffle=True, random_state=100 + fold)
        for inner_ref, inner_query in inner.split(outer_train):
            part = fn(outer_train[inner_ref], outer_train[inner_query])
            part.index = outer_train[inner_query]
            parts.append(part)
        part = fn(outer_train, outer_valid)
        part.index = outer_valid
        parts.append(part)
        train_part = pd.concat(parts).sort_index()
        require(train_part.index.equals(pd.RangeIndex(n_train)), f'{block} 第 {fold} 折的嵌套特征行数不符')
        test_part = fn(outer_train, TEST_ROWS).reset_index(drop=True)
        _FOLD_FEATS[key] = (train_part, test_part)
    return _FOLD_FEATS[key]


ID_TRAIN = pd.DataFrame({'row_id': train['id'].to_numpy(dtype='float64')})
ID_TEST = pd.DataFrame({'row_id': test['id'].to_numpy(dtype='float64')})
VARIANTS4 = {'R3': [], 'AS': ['AS'], 'MS': ['MS']}
if ID_STRUCTURE:
    VARIANTS4['ID'] = ['ID']
FAMILY_OF = {'lgb': 'lgb', 'xgb': 'xgb', 'cb_rel': 'cb_rel', 'cb_allo': 'cb_rel', 'cb_gl': 'cb_rel'}


def frames_aug(variant, model, fold):
    base_tr, base_te, _ = FRAMES[FAMILY_OF[model]]
    blocks = VARIANTS4[variant]
    extra_tr = [nested_features(b, fold)[0] for b in blocks if b in BLOCK_FUNCS]
    extra_te = [nested_features(b, fold)[1] for b in blocks if b in BLOCK_FUNCS]
    if 'ID' in blocks:
        extra_tr.append(ID_TRAIN)
        extra_te.append(ID_TEST)
    if not extra_tr:
        return base_tr, base_te
    return pd.concat([base_tr] + extra_tr, axis=1), pd.concat([base_te] + extra_te, axis=1)


started = time.perf_counter()
demo_tr, demo_te = nested_features('AS', 1)
print(f'AS 第 1 折：{demo_tr.shape[1]} 列，用时 {time.perf_counter() - started:.0f}s')
display(pd.DataFrame({'train_nonnull': demo_tr.notna().mean(), 'test_nonnull': demo_te.notna().mean(),
                      'train_mean': demo_tr.mean(), 'test_mean': demo_te.mean()}).round(3))
corr = pd.Series({c: np.corrcoef(demo_tr[c].fillna(demo_tr[c].median()), y)[0, 1] for c in demo_tr.columns})
print('与 TotalKg 的相关（第 1 折，训练行为嵌套折外值）：'); display(corr.round(3).sort_values())
