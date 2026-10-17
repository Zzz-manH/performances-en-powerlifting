"""生成 14_pl_colab_gpu_feature_round4.ipynb（只生成文件并做语法检查，不在本地执行）。

环境、数据、特征、模型定义、第二层函数原样取自 11。
用法：/opt/anaconda3/bin/python3 tools/build_14_feature_round4.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '14_pl_colab_gpu_feature_round4.ipynb'
NB11 = nbf.read(ROOT / '11_pl_colab_gpu_final.ipynb', as_version=4)
ENV_CELL = NB11.cells[2].source.replace("EXPERIMENT_ID = 'final_v1'", "EXPERIMENT_ID = 'feature_round4_v1'")
DATA_CELL, FEATURE_CELL, ROUND2_CELL, FINAL_FEATURE_CELL = (NB11.cells[i].source for i in (4, 6, 8, 10))
MODEL_DEFS = NB11.cells[12].source.split('\n# ---- 预检')[0]
STAGE2_DEFS = NB11.cells[16].source.split('\n# 第二层只依赖训练行的 OOF')[0]
assert "EXPERIMENT_ID = 'feature_round4_v1'" in ENV_CELL, '11 的环境单元位置变了'
assert 'FOLD_IDS' in DATA_CELL and "BLOCKS = {'P': P" in FEATURE_CELL and "BLOCKS2 = {'M2': M2" in ROUND2_CELL
assert "FRAMES = {m: frames3('ALL3'" in FINAL_FEATURE_CELL, '11 的最终特征单元位置变了'
assert 'def gbdt_fit' in MODEL_DEFS and 'probe' not in MODEL_DEFS, '11 的模型单元位置变了'
assert 'def s3_features' in STAGE2_DEFS and 'alphas =' not in STAGE2_DEFS, '11 的第二层单元位置变了'


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md('''
# 14 · Powerlifting：第四轮特征工程——把「这个人有多强」放进模型（Colab GPU，五折）

**为什么还能做**：前几轮的特征（记录精度、比赛背景、运动员历史计数）都是**不看成绩**的描述；它们已经接近饱和，剩下的分组偏差多在 2 kg 以内。但效果最大的一步——同一运动员平滑——用的是**同一个人在训练集里的真实成绩**，而且目前只在模型之后做线性修正（S3 −0.14），模型本身看不到。本轮把这类**个人层面、基于成绩**的信息直接做成特征，让树模型非线性地使用（例如：间隔越短、同一装备、记录越多，越可信）。

为了不泄漏，所有基于成绩的特征都用**嵌套折外**计算：外层每一折里，训练部分再分 5 个内层折，每行的特征只用其他内层折的成绩；外层验证行和测试行用整个训练部分。验证行的成绩从不参与它自己的特征。

|块|内容|
|---|---|
|`AS` 运动员实力|近似同一个人（性别 + 国家 + 州 [+ 联盟] [+ 装备] + 出生年 ±1）其他记录的相对力量（DOTS 分）：个数、均值、最高值；时间衰减均值（离这次比赛越近权重越大）、与最近一次的年份差、平均年份差；体重变化；同装备记录比例；双胞胎（16 列 / 核心 9 列相同）的总成绩均值与个数|
|`MS` 比赛实力|同一场比赛（联盟 × 年份 × 比赛国 × 比赛州）以及同一组别（再 × 性别 × 装备）**其他选手**的相对力量：个数、均值、最高值|
|`ID`（视检查结果）|如果 `id` 的顺序里藏着比赛结构（第 6 节检查），加入 `id` 作为特征|

**另外两项**
- **`id` 结构检查**：按 `id` 排序后，相邻两行来自同一场比赛的比例是否明显高于随机。若是，说明 `id` 保留了原始数据的顺序，就把它当特征一起测试。
- **目标变量工程**：目前最强的单模型是「DOTS 相对力量」目标的 CatBoost。本轮再试两种力量公式作为目标：异速生长（总成绩 ÷ 体重^(2/3)）和 IPF GL 系数（按性别 × 是否装备）。看它们单独的分数，以及加入融合后的增益。

**协议**：种子 42 的五折（与 07–13 相同），内层 10% 早停；模型 LightGBM、XGBoost、cb_rel（三个融合）。基线 `R3` = 11 的 ALL3 分模型特征。达标的块（三模型融合平均 Δ ≤ −0.05 且 ≥ 4/5 折）自动合成 `ALL4`。最后在 `R3` 与 `ALL4` 上都加双胞胎 + S3 平滑，看**扣除平滑之后的净收益**（新特征与平滑有重叠）。导出两个提交文件做配对比较。

**操作**：Colab 选 **T4 GPU**，从上到下运行；约 1–1.5 小时，可续跑。
'''),
    md('## 1. 环境与配置（与 11 相同，实验名不同）'),
    code(ENV_CELL + '''

SCREEN_MODELS = ['lgb', 'xgb', 'cb_rel']        # 筛选用的三个模型（cb_rel 是最强的单模型）
TARGET_MODELS = ['cb_allo', 'cb_gl']            # 目标变量工程：两种力量公式
INNER_FEATURE_FOLDS = 5                         # 基于成绩的特征：外层训练部分内的嵌套折数
'''),
    md('## 2. 读取并核对数据（与 07 相同）'),
    code(DATA_CELL),
    md('## 3. 第一轮特征（与 07 相同）'),
    code(FEATURE_CELL),
    md('## 4. 第二轮特征（与 09 相同）'),
    code(ROUND2_CELL),
    md('## 5. 第三轮特征与各模型的最终特征表（与 11 相同，作为本轮基线 R3）'),
    code(FINAL_FEATURE_CELL),
    md('## 6. `id` 结构检查（只用 CPU，几秒）'),
    code('''
from scipy.stats import spearmanr

ids_all = pd.concat([train['id'], test['id']], ignore_index=True).to_numpy()
order = np.argsort(ids_all)
perm = np.random.default_rng(0).permutation(len(ids_all))
checks = {'同一场比赛': meet.astype(str).to_numpy(), '同一联盟': both['Federation'].astype(str).to_numpy(),
          '同一比赛年份': both['MeetYear'].to_numpy(), '同一国家': both['Country'].astype(str).to_numpy()}
rows = []
for name, values in checks.items():
    by_id = float((values[order][1:] == values[order][:-1]).mean())
    random = float((values[perm][1:] == values[perm][:-1]).mean())
    rows.append({'相邻两行': name, '按 id 排序': by_id, '随机顺序': random, '倍数': by_id / max(random, 1e-12)})
id_table = pd.DataFrame(rows).set_index('相邻两行')
display(id_table.round(4))
rho = float(spearmanr(ids_all, both['MeetYear'].to_numpy())[0])
print(f'id 与比赛年份的 Spearman 相关：{rho:.4f}')
ID_STRUCTURE = bool(id_table.loc['同一场比赛', '倍数'] >= 1.5 or abs(rho) >= 0.1)
print('id 含有比赛结构：', ID_STRUCTURE, '→', '加入 ID 变体' if ID_STRUCTURE else '不测试 id（顺序是随机的）')
id_table.to_csv(OUTPUT_DIR / 'summary_id_check.csv')
'''),
    md('## 7. 基于成绩的嵌套特征（AS 运动员实力、MS 比赛实力）'),
    code('''
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
'''),
    md('''
## 8. 模型定义（与 11 相同）+ 两种新的力量目标 + 预检

- `cb_allo`：目标 = 总成绩 ÷ (体重 / 80)^(2/3)（异速生长）。
- `cb_gl`：目标 = IPF GL 分，系数按性别 × 是否装备（Raw / Wraps 用无装备系数，Single-ply / Multi-ply / Unlimited 用装备系数）。
- 两者都按系数² 加权（与 cb_rel 相同），所以仍在最小化 kg 的 RMSE；参数与 cb_rel 相同（CatBoost C2）。
'''),
    code(MODEL_DEFS + '''


def gl_scale(sex_values, equipment_values, bw_values):
    sex = np.asarray(sex_values).astype(str)
    equipped = ~np.isin(np.asarray(equipment_values).astype(str), ['Raw', 'Wraps'])
    bw = np.maximum(np.asarray(bw_values, dtype='float64'), 30.0)
    coef = {('M', False): (1199.72839, 1025.18162, 0.00921), ('M', True): (1236.25115, 1449.21864, 0.01644),
            ('F', False): (610.32796, 1045.59282, 0.03048), ('F', True): (758.63878, 949.31382, 0.02435)}

    def curve(s, e):
        a, b, c = coef[(s, e)]
        return a - b * np.exp(-c * bw)

    male = np.where(equipped, curve('M', True), curve('M', False))
    female = np.where(equipped, curve('F', True), curve('F', False))
    return np.where(sex == 'M', male, np.where(sex == 'F', female, 0.5 * (male + female))) / 100.0


ALLO = (train['BodyweightKg'].to_numpy(dtype='float64') / 80.0) ** (2 / 3)
ALLO_TEST = (test['BodyweightKg'].to_numpy(dtype='float64') / 80.0) ** (2 / 3)
GL = gl_scale(train['Sex'], train['Equipment'], train['BodyweightKg'])
GL_TEST = gl_scale(test['Sex'], test['Equipment'], test['BodyweightKg'])
require(min(GL.min(), GL_TEST.min()) > 0 and min(ALLO.min(), ALLO_TEST.min()) > 0, '力量系数异常')
SCALE_OF = {'cb_rel': (SCALE, SCALE_TEST), 'cb_allo': (ALLO, ALLO_TEST), 'cb_gl': (GL, GL_TEST)}
for extra in TARGET_MODELS:
    FRAMES[extra] = FRAMES['cb_rel']
    PATIENCE[extra] = PATIENCE['cb_rel']


def fit_matrix(model, Xtr, Xte, a, s, v, seed=SEED, rounds=MAX_ROUNDS, test_rows=None):
    """用给定的特征表训练（内层拟合 a、早停 s），返回 (外层验证预测, 测试预测, 最佳轮数)。"""
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    if model in SCALE_OF:
        scale, scale_test = SCALE_OF[model]
        scale_t = scale_test if test_rows is None else scale_test[test_rows]
        ya, ys = y[a] / scale[a], y[s] / scale[s]
        norm = (scale[a] ** 2).mean()
        wa, ws = scale[a] ** 2 / norm, scale[s] ** 2 / norm
    else:
        ya, ys, wa, ws = y[a], y[s], None, None
    m, best = gbdt_fit(model, seed, Xtr.iloc[a], ya, wa, Xtr.iloc[s], ys, ws, rounds)
    p_val = np.asarray(gbdt_predict(model, m, best, Xtr.iloc[v]), dtype='float64')
    p_test = np.asarray(gbdt_predict(model, m, best, Xt), dtype='float64')
    if model in SCALE_OF:
        p_val, p_test = p_val * scale[v], p_test * scale_t
    return p_val, p_test, best


rng = np.random.default_rng(SEED)
probe = rng.choice(np.flatnonzero(FOLD_IDS != 1), 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
checks = [(m, v) for v in VARIANTS4 for m in SCREEN_MODELS] + [(m, 'R3') for m in TARGET_MODELS]
for model, variant in checks:
    try:
        Xtr, Xte = frames_aug(variant, model, 1)
        p_val, p_test, _ = fit_matrix(model, Xtr, Xte, pa, ps, pv, rounds=30, test_rows=np.arange(300))
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
        print(f'预检通过：{model:8s} {variant:3s}（{Xtr.shape[1]} 列）RMSE@30轮 {rmse(y[pv], p_val):.1f}')
    except Exception as exc:
        failed[(model, variant)] = repr(exc)
        print(f'预检失败，跳过：{model} {variant} → {exc!r}')
require(not any(v == 'R3' and m in SCREEN_MODELS for (m, v) in failed), '基线 R3 预检失败，请检查环境')
'''),
    md('## 9. 筛选：R3 与每个新块单独加入（五折，可续跑）'),
    code('''
def run_one(model, variant, fold):
    stem = f'{model}__{variant}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        return json.loads(json_path.read_text(encoding='utf-8')), 'cache'
    Xtr, Xte = frames_aug(variant, model, fold)
    fit_idx, valid_idx = np.flatnonzero(FOLD_IDS != fold), np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 07–13
    started = time.perf_counter()
    p_val, p_test, best = fit_matrix(model, Xtr, Xte, inner_fit, inner_stop, valid_idx)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    np.savez(npz_path, valid_idx=valid_idx, val=p_val, test=p_test)
    info = {'model': model, 'variant': variant, 'fold': fold, 'blocks': VARIANTS4[variant],
            'n_features': int(Xtr.shape[1]), 'best_iteration': best, 'rmse': rmse(y[valid_idx], p_val),
            'seconds': time.perf_counter() - started, 'gpu': GPU_NAME,
            'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info, f'{info["seconds"]:.0f}s'


def run_variants(variants, models):
    for fold in range(1, N_FOLDS + 1):
        fold_started = time.perf_counter()
        for model in models:
            for variant in variants:
                if (model, variant) in failed:
                    continue
                info, how = run_one(model, variant, fold)
                print(f'fold {fold}  {model:8s} {variant:4s}  RMSE {info["rmse"]:.3f}  '
                      f'最佳轮数 {info["best_iteration"]:5d}  ({how})')
        print(f'—— fold {fold} 完成，本次运行用时 {(time.perf_counter() - fold_started) / 60:.1f} 分钟')


run_variants(list(VARIANTS4), SCREEN_MODELS)
'''),
    md('## 10. 与 R3 配对比较（单模型 + 三模型融合），自动合成 ALL4'),
    code('''
def load_oof(model, variant):
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{model}__{variant}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    return oof, test_pred


def fold_rmse(pred):
    return pd.Series({k: rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)})


def paired_table(preds, base='R3'):
    base_folds = fold_rmse(preds[base])
    rows = []
    for name, pred in preds.items():
        delta = fold_rmse(pred) - base_folds
        rows.append({'variant': name, 'oof_rmse': rmse(y, pred), 'mean_delta': delta.mean(),
                     'folds_better': int((delta < 0).sum()), **{f'd_fold{k}': v for k, v in delta.items()}})
    table = pd.DataFrame(rows).set_index('variant')
    table['keep'] = (table['mean_delta'] <= KEEP_DELTA) & (table['folds_better'] >= 4)
    table.loc[base, 'keep'] = False
    return table.sort_values('oof_rmse')


def blend_variant(variant, models=SCREEN_MODELS):
    loaded = [load_oof(m, variant) for m in models]
    if any(item is None for item in loaded):
        return None
    P_oof = np.column_stack([o for o, _ in loaded])
    P_test = np.column_stack([t for _, t in loaded])
    w, _ = nnls(P_oof, y)
    return {'oof': P_oof @ w, 'test': P_test @ w, 'weights': [round(float(x), 3) for x in w]}


for model in SCREEN_MODELS:
    preds = {v: load_oof(model, v)[0] for v in VARIANTS4 if load_oof(model, v) is not None}
    if 'R3' in preds:
        print(f'\\n=== {model}')
        display(paired_table(preds).round(3))
BLENDS = {v: blend_variant(v) for v in VARIANTS4}
BLENDS = {v: b for v, b in BLENDS.items() if b is not None}
require('R3' in BLENDS, 'R3 还没有全部完成')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cbrel'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\\n=== 三模型 NNLS 融合（主要判读依据）')
display(blend_table.round(3))
blend_table.to_csv(OUTPUT_DIR / 'summary_blend_screen.csv')
KEPT4 = [v for v in VARIANTS4 if v != 'R3' and v in blend_table.index and bool(blend_table.loc[v, 'keep'])]
print('达标的新块：', KEPT4 or '无')
if len(KEPT4) >= 2:
    VARIANTS4['ALL4'] = [b for v in KEPT4 for b in VARIANTS4[v]]
    run_variants(['ALL4'], SCREEN_MODELS)
    BLENDS['ALL4'] = blend_variant('ALL4')
    display(paired_table({v: BLENDS[v]['oof'] for v in ['R3'] + KEPT4 + ['ALL4']}).round(3))
FINAL4 = 'ALL4' if 'ALL4' in BLENDS else (KEPT4[0] if KEPT4 else None)
if FINAL4 is None and len(blend_table) > 1:
    FINAL4 = blend_table.index[0] if blend_table.index[0] != 'R3' else blend_table.index[1]
print('与 R3 做净收益比较的变体：', FINAL4)
'''),
    md('## 11. 扣除第二层平滑后的净收益（双胞胎 + S3，交叉拟合，与 11 相同的做法）→ 导出提交文件'),
    code(STAGE2_DEFS + '''


def stage2(oof, test_pred):
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    p1 = apply_twins(oof, MATE, LEVEL, alphas)
    p1_cf = oof.copy()
    for h in (0, 1):
        a_h = {lv: fit_alpha((HALF != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        m = HALF == h
        p1_cf[m] = apply_twins(oof[m], MATE[m], LEVEL[m], a_h)
    r1 = y - p1
    F, FT = s3_features(r1)
    p2_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        p2_cf[m] = p1_cf[m] + F[m] @ np.linalg.lstsq(F[fit], r1[fit], rcond=None)[0]
    beta = np.linalg.lstsq(F, r1, rcond=None)[0]
    test_final = apply_twins(test_pred, T_MATE, T_LEVEL, alphas) + FT @ beta
    return {'oof': rmse(y, oof), '+双胞胎': rmse(y, p1_cf), '+S3': rmse(y, p2_cf),
            'fold_final': fold_rmse(p2_cf), 'test_final': test_final}


net_rows, NET = [], {}
for name in [v for v in ['R3', FINAL4] if v is not None]:
    NET[name] = stage2(BLENDS[name]['oof'], BLENDS[name]['test'])
    net_rows.append({'variant': name, **{k: NET[name][k] for k in ['oof', '+双胞胎', '+S3']}})
    path = OUTPUT_DIR / f'submission_{name}_blend3_stage2.csv'
    pd.DataFrame({'id': test['id'], TARGET: np.clip(NET[name]['test_final'], 0, None)}).to_csv(path, index=False)
    print('写出', path.name)
net_table = pd.DataFrame(net_rows).set_index('variant')
display(net_table.round(3))
if FINAL4 is not None:
    delta = NET[FINAL4]['fold_final'] - NET['R3']['fold_final']
    print(f'{FINAL4} 相对 R3（都加平滑后）：平均 {delta.mean():+.3f}，{int((delta < 0).sum())}/5 折更好')
net_table.to_csv(OUTPUT_DIR / 'summary_net_after_stage2.csv')
'''),
    md('## 12. 目标变量工程：异速生长 / IPF GL 作为目标（R3 特征，五折）'),
    code('''
for model in TARGET_MODELS:
    if (model, 'R3') in failed:
        continue
    for fold in range(1, N_FOLDS + 1):
        info, how = run_one(model, 'R3', fold)
        print(f'fold {fold}  {model:8s}  RMSE {info["rmse"]:.3f}  最佳轮数 {info["best_iteration"]}  ({how})')

base_models = SCREEN_MODELS
P_base = np.column_stack([load_oof(m, 'R3')[0] for m in base_models])


def nested_nnls(P):
    out = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        w, _ = nnls(P[fit], y[fit])
        out[va] = P[va] @ w
    return out


base_nested = nested_nnls(P_base)
rows = [{'model': m, 'oof': rmse(y, load_oof(m, 'R3')[0])} for m in base_models]
for model in TARGET_MODELS:
    loaded = load_oof(model, 'R3')
    if loaded is None:
        continue
    nested = nested_nnls(np.column_stack([P_base, loaded[0]]))
    delta = fold_rmse(nested) - fold_rmse(base_nested)
    rows.append({'model': model, 'oof': rmse(y, loaded[0]),
                 'corr_with_cb_rel': float(np.corrcoef(loaded[0], load_oof('cb_rel', 'R3')[0])[0, 1]),
                 'blend_delta_nested': rmse(y, nested) - rmse(y, base_nested), 'folds_better': int((delta < 0).sum())})
target_table = pd.DataFrame(rows).set_index('model')
print(f'R3 三模型融合（嵌套）：{rmse(y, base_nested):.3f}')
display(target_table.round(3))
target_table.to_csv(OUTPUT_DIR / 'summary_target_engineering.csv')
print('完成。请提交 submission_R3_blend3_stage2.csv 和 submission_<新变体>_blend3_stage2.csv 做配对比较；跑完告诉 Claude。')
'''),
]

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'language_info': {'name': 'python'},
    'accelerator': 'GPU',
    'colab': {'provenance': [], 'gpuType': 'T4'},
}
nbf.validate(nb)
for i, cell in enumerate(nb['cells']):                      # 语法检查：只编译，不执行
    if cell['cell_type'] == 'code':
        compile(cell['source'], f'{NAME}[cell {i}]', 'exec')
nbf.write(nb, ROOT / NAME)
print('wrote', ROOT / NAME, '| code cells compiled OK')
