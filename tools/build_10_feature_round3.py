"""生成 10_pl_colab_gpu_feature_round3.ipynb（只生成文件并做语法检查，不在本地执行）。

环境、数据、第一轮、第二轮特征单元原样取自 07 / 09，保证折号、内层早停划分、R2 特征与之前完全相同。
用法：/opt/anaconda3/bin/python3 tools/build_10_feature_round3.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '10_pl_colab_gpu_feature_round3.ipynb'
NB07 = nbf.read(ROOT / '07_pl_colab_gpu_feature_round1.ipynb', as_version=4)
NB09 = nbf.read(ROOT / '09_pl_colab_gpu_feature_round2.ipynb', as_version=4)
DATA_CELL, FEATURE_CELL = NB07.cells[4].source, NB07.cells[6].source
ROUND2_CELL = NB09.cells[8].source
ENV_CELL = NB09.cells[2].source.replace("EXPERIMENT_ID = 'feature_round2_v1'", "EXPERIMENT_ID = 'feature_round3_v1'")
FIT_DEFS = NB09.cells[10].source.split('\nrng = np.random.default_rng(SEED)')[0]
assert 'def require' in DATA_CELL and 'FOLD_IDS' in DATA_CELL, '07 的数据单元位置变了'
assert "BLOCKS = {'P': P" in FEATURE_CELL, '07 的特征单元位置变了'
assert "BLOCKS2 = {'M2': M2" in ROUND2_CELL and 'def frames2' in ROUND2_CELL, '09 的第二轮特征单元位置变了'
assert "EXPERIMENT_ID = 'feature_round3_v1'" in ENV_CELL and 'RUN_ONLY' in ENV_CELL, '09 的环境单元位置变了'
assert 'def fit_predict' in FIT_DEFS and 'preflight' not in FIT_DEFS and 'probe' not in FIT_DEFS, '09 的模型单元位置变了'


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md('''
# 10 · Powerlifting：第三轮特征 + 第二层平滑实验（Colab GPU，五折）

**08 / 09 的结果**

|实验|CV RMSE|公开榜|
|---|---:|---:|
|08 五折堆叠（07 的模型 + TabM、cb_rel、lgb_rel、RealMLP，全局 NNLS）|77.013|77.207|
|+ 双胞胎|76.757|77.018|
|+ 同一运动员平滑|**76.521**|**76.731**|
|v2 + 双胞胎 + 同一运动员平滑|76.774|77.125|

- 新模型：TabM 单独 77.636，是目前最好的单模型，加入 07 融合 −0.15（5/5 折）；cb_rel 77.761（−0.06）；RealMLP −0.07 但每折 15 分钟；lgb_rel −0.04。三种堆叠方式中全局 NNLS 最好。
- 同一运动员平滑：交叉拟合 −0.24（v2 上 −0.28），公开榜 −0.29 / −0.48，是目前最有效的一步。
- 09：运动员历史 `H` −0.13（5/5 折，唯一达标）；`C`（CatBoost 的比赛 / 运动员 ID）单模型 −0.12，融合只 −0.05；`M2` 对 CatBoost、XGBoost 各 −0.06，对 LightGBM +0.07；`D`、`S` 对 XGBoost 有效（−0.08 / −0.09），融合无效。残差诊断显示剩下的分组偏差多在 2 kg 以内。

**本轮基线 `R2`** = 09 的 ALL2（F0 + P + T + M + H，41 列），模型同 09（LightGBM、XGBoost、CatBoost C2）。

|块|内容|为什么|
|---|---|---|
|`H2` 运动员生涯|这个人参加过的最大比赛规模、最多来自几个州的比赛、是否出国比赛过、换过几个级别、体重变化范围、第一次出现的年龄、同一年出现次数、参加测试赛的比例、出生年 ±1 的模糊匹配计数|`H` 有效说明「这个人是谁」重要；「去过多高级别的比赛」把比赛级别（M）和个人（H）结合起来|
|`E` 比赛的经验结构|选手平均出现次数、平均经验年数、老将（经验 ≥ 3 年）比例、新人（只出现一次）比例|经验丰富的选手多的比赛级别更高|
|`PM` 按模型挑特征|CatBoost 加 `M2` + `C`；XGBoost 加 `M2` + `S` + `D`；LightGBM 不加|09 中各模型单独有效的块不同；也增加融合多样性|

**第二层平滑实验**（只用 CPU，几分钟）：在本轮最终融合上比较 5 种做法：08 原版（精确出生年）、出生年 ±1 模糊匹配、时间衰减（离这次比赛越近的成绩权重越大）、三者合并的线性组合、小 LightGBM。全部按「性别 + 国家 + 州」分两半交叉拟合（同一运动员的所有记录在同一半）。最好的一种写进提交文件，并用于最终集成（11）。

**判读**：同 07 / 09：三模型融合平均 ΔRMSE ≤ −0.05 kg 且 ≥ 4/5 折。达标的块自动合成 `ALL3`。

**操作**：Colab 选 **T4 GPU**，从上到下运行；约 1–1.5 小时，可续跑。
'''),
    md('## 1. 环境、GPU 与配置（与 09 相同，实验名不同）'),
    code(ENV_CELL),
    md('## 2. 读取并核对数据（与 07 相同）'),
    code(DATA_CELL),
    md('## 3. 第一轮特征（与 07 相同）'),
    code(FEATURE_CELL),
    md('## 4. 第二轮特征（与 09 相同）'),
    code(ROUND2_CELL),
    md('## 5. 第三轮特征块'),
    code('''
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
'''),
    md('## 6. 模型与预检（参数与 09 相同：LightGBM、XGBoost 用 v2 参数，CatBoost 用 C2）'),
    code(FIT_DEFS + '''

rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for variant in SCREEN3:
    for model in MODELS:
        if not runs(model, variant):
            continue
        Xtr, Xte, cat_cols = frames3(variant, model)
        try:
            p_val, p_test, _ = fit_predict(model, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                           Xte.iloc[:500], cat_cols, rounds=30)
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
            require(len(p_val) == len(pv) and len(p_test) == 500, '预测长度不符')
            print(f'预检通过：{model:8s} {variant:3s}（{Xtr.shape[1]} 列）RMSE@30轮 {rmse(y[pv], p_val):.1f}')
        except Exception as exc:
            failed[(model, variant)] = repr(exc)
            print(f'预检失败，跳过：{model} {variant} → {exc!r}')
require(not any(v == 'R2' for (_, v) in failed), '基线 R2 预检失败，请检查环境')
'''),
    md('## 7. 筛选：R2 与每个新块单独加入（五折，可续跑）'),
    code('''
CONFIG = {'purpose': 'feature round 3 on top of 09 ALL2; inner 10% early stop, score early-stopped model on outer fold',
          'seed': SEED, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'r2_blocks': R2_BLOCKS, 'extras': EXTRAS, 'screen': VARIANTS3, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def spec_of(model, variant):
    Xtr, _, cat_cols = frames3(variant, model)
    params = {'lgb': LGB_PARAMS, 'xgb': XGB_PARAMS, 'catboost': CB_PARAMS}[model]
    return {'model': model, 'variant': variant, 'blocks': blocks_for(variant, model),
            'columns': list(Xtr.columns), 'cat_cols': list(cat_cols), 'params': dict(params),
            'patience': PATIENCE[model]}


def run_one(model, variant, fold):
    stem = f'{model}__{variant}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    spec = spec_of(model, variant)
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{stem} 的定义已改变；请改 EXPERIMENT_ID 或变体名')
        return saved, 'cache'
    Xtr, Xte, cat_cols = frames3(variant, model)
    fit_idx = np.flatnonzero(FOLD_IDS != fold)
    valid_idx = np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 07
    started = time.perf_counter()
    p_val, p_test, best = fit_predict(model, Xtr.iloc[inner_fit], y[inner_fit], Xtr.iloc[inner_stop],
                                      y[inner_stop], Xtr.iloc[valid_idx], Xte, cat_cols)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    np.savez(npz_path, valid_idx=valid_idx, val=np.asarray(p_val, dtype='float64'),
             test=np.asarray(p_test, dtype='float64'))
    result = {'model': model, 'variant': variant, 'fold': fold, 'spec': spec, 'n_features': int(Xtr.shape[1]),
              'best_iteration': best, 'hit_limit': bool(best >= MAX_ROUNDS - PATIENCE[model]),
              'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started,
              'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result, f'{result["seconds"]:.0f}s'


def run_variants(variants):
    for fold in range(1, N_FOLDS + 1):
        fold_started = time.perf_counter()
        for model in MODELS:
            for variant in variants:
                if not runs(model, variant) or (model, variant) in failed:
                    continue
                result, how = run_one(model, variant, fold)
                print(f'fold {fold}  {model:8s} {variant:4s}  RMSE {result["rmse"]:.3f}  '
                      f'最佳轮数 {result["best_iteration"]:5d}  ({how})')
        print(f'—— fold {fold} 完成，本次运行用时 {(time.perf_counter() - fold_started) / 60:.1f} 分钟')


run_variants(SCREEN3)
'''),
    md('## 8. 与 R2 配对比较（单模型 + 三模型融合），自动合成 ALL3'),
    code('''
def load_oof(model, variant):
    """五折都完成时返回 (OOF, 五折平均测试预测)，否则 None。"""
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{model}__{variant}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{model} {variant} 的 OOF 不完整')
    return oof, test_pred


def fold_rmse(pred):
    return pd.Series({k: rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)})


def paired_table(preds, base='R2'):
    require(base in preds, f'缺少基线 {base}')
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


def blend_variant(variant):
    """三模型 NNLS 融合；某模型在该变体上特征不变时用它的 R2 结果。"""
    loaded = []
    for model in MODELS:
        item = load_oof(model, variant if runs(model, variant) else 'R2')
        if item is None:
            return None
        loaded.append(item)
    P_oof = np.column_stack([o for o, _ in loaded])
    P_test = np.column_stack([t for _, t in loaded])
    w, _ = nnls(P_oof, y)
    return {'oof': P_oof @ w, 'test': P_test @ w, 'weights': [round(float(v), 3) for v in w]}


results = pd.DataFrame([json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))])
for model in MODELS:
    preds = {}
    for variant in VARIANTS3:
        item = load_oof(model, variant) if runs(model, variant) else None
        if item is not None:
            preds[variant] = item[0]
    if 'R2' in preds:
        table = paired_table(preds)
        part = results[results['model'] == model].groupby('variant')
        table['mean_best_iter'] = part['best_iteration'].mean()
        table['minutes'] = part['seconds'].sum() / 60
        print(f'\\n=== {model}')
        display(table.round(3))
        table.to_csv(OUTPUT_DIR / f'summary_single_{model}.csv')

BLENDS = {}
for variant in VARIANTS3:
    item = blend_variant(variant)
    if item is not None:
        BLENDS[variant] = item
require('R2' in BLENDS, '三模型的 R2 还没有全部完成')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cb'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\\n=== 三模型 NNLS 融合（主要判读依据）')
display(blend_table.round(3))
blend_table.to_csv(OUTPUT_DIR / 'summary_blend_screen.csv')

KEPT = [v for v in ['H2', 'E', 'PM'] if v in blend_table.index and bool(blend_table.loc[v, 'keep'])]
VARIANTS3['ALL3'] = KEPT
print('达标的新块：', KEPT or '无', '→ ALL3 = R2 +', KEPT)
'''),
    md('## 9. ALL3：R2 + 全部达标的块（五折）→ 双胞胎后处理 → 导出提交文件'),
    code('''
if KEPT:
    for model in MODELS:
        if runs(model, 'ALL3'):
            Xtr, Xte, cat_cols = frames3('ALL3', model)
            p_val, p_test, _ = fit_predict(model, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                           Xte.iloc[:500], cat_cols, rounds=30)          # ALL3 的预检
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'ALL3 {model} 预检失败')
    run_variants(['ALL3'])
    BLENDS['ALL3'] = blend_variant('ALL3')
    final_table = paired_table({v: BLENDS[v]['oof'] for v in ['R2'] + KEPT + ['ALL3']})
    display(final_table.round(3))
    final_table.to_csv(OUTPUT_DIR / 'summary_blend_all3.csv')
    FINAL = 'ALL3'
else:
    print('没有达标的新块，保持 R2。')
    FINAL = 'R2'

CORE = CORE9


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def mate_stats(train_keys, query_keys=None):
    table = pd.DataFrame({'k': train_keys, 'y': y}).groupby('k')['y'].agg(['sum', 'count'])
    if query_keys is None:                      # 训练行：留一
        keys = pd.Series(train_keys)
        n = keys.map(table['count']).to_numpy() - 1
        s = keys.map(table['sum']).to_numpy() - y
    else:                                       # 测试行：全部训练行
        keys = pd.Series(query_keys)
        n = keys.map(table['count']).fillna(0).to_numpy()
        s = keys.map(table['sum']).fillna(0).to_numpy()
    return n, np.where(n > 0, s / np.maximum(n, 1), np.nan)


def twins(query_exact=None, query_core=None):
    n_ex, m_ex = mate_stats(row_keys(train, RAW_FEATURES), query_exact)
    n_co, m_co = mate_stats(row_keys(train, CORE), query_core)
    level = np.select([n_ex > 0, n_co > 0], ['exact', 'core'], default='none')
    return level, np.where(level == 'exact', m_ex, m_co)


def fit_alpha(mask, pred, mate):
    d = mate[mask] - pred[mask]
    denom = float(np.sum(d * d))
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / denom, 0, 1)) if denom > 0 else 0.0


def apply_twins(pred, mate, level, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


LEVEL, MATE = twins()
T_LEVEL, T_MATE = twins(row_keys(test, RAW_FEATURES), row_keys(test, CORE))
HALF = (pd.util.hash_pandas_object(train[['Sex', 'Country', 'State']], index=False).to_numpy() % 2).astype(int)
twin_rows = []
for name in (['R2', FINAL] if FINAL != 'R2' else ['R2']):
    oof, test_pred = BLENDS[name]['oof'], BLENDS[name]['test']
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    crossfit = oof.copy()
    for h in (0, 1):
        a = {lv: fit_alpha((HALF != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        crossfit[HALF == h] = apply_twins(oof[HALF == h], MATE[HALF == h], LEVEL[HALF == h], a)
    fixed = apply_twins(test_pred, T_MATE, T_LEVEL, alphas)
    twin_rows.append({'name': name, 'oof_rmse': rmse(y, oof), 'oof_rmse_twinfix_crossfit': rmse(y, crossfit),
                      'alpha_exact': alphas['exact'], 'alpha_core': alphas['core']})
    for suffix, pred in [('', test_pred), ('_twinfix', fixed)]:
        path = OUTPUT_DIR / f'submission_{name}_blend3{suffix}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('写出', path.name)
    np.save(OUTPUT_DIR / f'oof_{name}_blend3.npy', oof)
    np.save(OUTPUT_DIR / f'test_{name}_blend3.npy', test_pred)
twin_table = pd.DataFrame(twin_rows).set_index('name')
display(twin_table.round(3))
twin_table.to_csv(OUTPUT_DIR / 'summary_twins.csv')
'''),
    md('''
## 10. 第二层平滑实验（CPU，几分钟）

在 `FINAL` 融合加双胞胎之后，比较 5 种「同一运动员」残差平滑：

|做法|特征|
|---|---|
|`S0_exact`|08 原版：键 A0（性别 + 国家 + 州 + 出生年）、A1（+ 联盟）、A2（+ 装备）、A3（+ 级别），以及比赛键 M2；同键其他训练行的残差和 ÷（个数 + 收缩常数）|
|`S1_fuzzy`|同上，但出生年允许 ±1（整数年龄推算的出生年可能差 1）|
|`S2_decay`|同上，按比赛年份差加权：权重 exp(−年份差 / 3)，最多 ±8 年|
|`S3_all_linear`|三种特征合在一起，线性组合|
|`S4_all_gbm`|三种特征合在一起，小 LightGBM|

全部按「性别 + 国家 + 州」的哈希奇偶分两半交叉拟合（同一运动员的所有记录在同一半）：一半拟合、另一半评分。
'''),
    code('''
base_oof, base_test = BLENDS[FINAL]['oof'], BLENDS[FINAL]['test']
TR, TE = train.reset_index(drop=True), test.reset_index(drop=True)


def text_key(frame, columns):
    out = frame[columns[0]].astype(str)
    for c in columns[1:]:
        out = out + '|' + frame[c].astype(str)
    return out


BY_TR = np.floor(TR['MeetYear'] - TR['Age'] + 1e-9).astype(int)
BY_TE = np.floor(TE['MeetYear'] - TE['Age'] + 1e-9).astype(int)
YR_TR, YR_TE = TR['MeetYear'].astype(int), TE['MeetYear'].astype(int)
PREFIX = {'A0': ['Sex', 'Country', 'State'], 'A1': ['Sex', 'Country', 'State', 'Federation'],
          'A2': ['Sex', 'Country', 'State', 'Federation', 'Equipment'],
          'A3': ['Sex', 'Country', 'State', 'Federation', 'Equipment', 'WeightClassKg']}
SHRINK = {'A0': 5.0, 'A1': 5.0, 'A2': 3.0, 'A3': 2.0}
PFX_TR = {k: text_key(TR, c) for k, c in PREFIX.items()}
PFX_TE = {k: text_key(TE, c) for k, c in PREFIX.items()}
MEET_COLS = ['Federation', 'MeetYear', 'MeetCountry', 'MeetState', 'Sex', 'Equipment']
MEET_TR, MEET_TE = text_key(TR, MEET_COLS), text_key(TE, MEET_COLS)
VALID_TR = (TR['Country'] != 'Unknown').to_numpy(dtype='float64')
VALID_TE = (TE['Country'] != 'Unknown').to_numpy(dtype='float64')


def sums(keys, values):
    table = pd.DataFrame({'k': keys.to_numpy(), 'v': values}).groupby('k')['v'].agg(['sum', 'count'])
    return table['sum'], table['count']


def look(table, keys):
    s, n = table
    return keys.map(s).fillna(0).to_numpy(dtype='float64'), keys.map(n).fillna(0).to_numpy(dtype='float64')


def f_exact(k, r):
    ktr, kte = PFX_TR[k] + '|' + BY_TR.astype(str), PFX_TE[k] + '|' + BY_TE.astype(str)
    table = sums(ktr, r)
    s, n = look(table, ktr)
    st, nt = look(table, kte)
    return (s - r) / (n - 1 + SHRINK[k]), st / (nt + SHRINK[k])


def f_fuzzy(k, r):
    table = sums(PFX_TR[k] + '|' + BY_TR.astype(str), r)
    s, n, st, nt = np.zeros(n_train), np.zeros(n_train), np.zeros(len(TE)), np.zeros(len(TE))
    for d in (-1, 0, 1):
        a, b = look(table, PFX_TR[k] + '|' + (BY_TR + d).astype(str))
        s, n = s + a, n + b
        a, b = look(table, PFX_TE[k] + '|' + (BY_TE + d).astype(str))
        st, nt = st + a, nt + b
    return (s - r) / (n - 1 + SHRINK[k]), st / (nt + SHRINK[k])


def f_decay(k, r, tau=3.0, max_gap=8):
    base_tr = PFX_TR[k] + '|' + BY_TR.astype(str) + '|'
    base_te = PFX_TE[k] + '|' + BY_TE.astype(str) + '|'
    table = sums(base_tr + YR_TR.astype(str), r)
    s, n, st, nt = np.zeros(n_train), np.zeros(n_train), np.zeros(len(TE)), np.zeros(len(TE))
    for d in range(-max_gap, max_gap + 1):
        w = float(np.exp(-abs(d) / tau))
        a, b = look(table, base_tr + (YR_TR + d).astype(str))
        s, n = s + w * a, n + w * b
        a, b = look(table, base_te + (YR_TE + d).astype(str))
        st, nt = st + w * a, nt + w * b
    return (s - r) / (n - 1 + SHRINK[k]), st / (nt + SHRINK[k])     # d = 0 时自己权重为 1，已扣除


def f_meet(r, lam=20.0):
    table = sums(MEET_TR, r)
    s, n = look(table, MEET_TR)
    st, nt = look(table, MEET_TE)
    return (s - r) / (n - 1 + lam), st / (nt + lam)


def stage2_features(r, kind):
    builders = {'exact': f_exact, 'fuzzy': f_fuzzy, 'decay': f_decay}
    cols_tr, cols_te, names = [], [], []
    for kd in (['exact', 'fuzzy', 'decay'] if kind == 'all' else [kind]):
        for k in PREFIX:
            a, b = builders[kd](k, r)
            cols_tr.append(a * VALID_TR)
            cols_te.append(b * VALID_TE)
            names.append(f'{kd}_{k}')
    a, b = f_meet(r)
    cols_tr.append(a)
    cols_te.append(b)
    names.append('meet')
    return np.column_stack(cols_tr), np.column_stack(cols_te), names


alphas = {lv: fit_alpha(LEVEL == lv, base_oof, MATE) for lv in ['exact', 'core']}
p1 = apply_twins(base_oof, MATE, LEVEL, alphas)
p1_cf = base_oof.copy()
for h in (0, 1):
    a_h = {lv: fit_alpha((HALF != h) & (LEVEL == lv), base_oof, MATE) for lv in ['exact', 'core']}
    m = HALF == h
    p1_cf[m] = apply_twins(base_oof[m], MATE[m], LEVEL[m], a_h)
t1 = apply_twins(base_test, T_MATE, T_LEVEL, alphas)
r1 = y - p1
FEATS = {kind: stage2_features(r1, kind) for kind in ['exact', 'fuzzy', 'decay', 'all']}
GBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8,
                  subsample_freq=1, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)


def linear_fit_predict(F_fit, r_fit, F_apply):
    beta = np.linalg.lstsq(F_fit, r_fit, rcond=None)[0]
    return F_apply @ beta


def gbm_fit_predict(F_fit, r_fit, F_apply):
    return lgb.LGBMRegressor(**GBM_PARAMS).fit(F_fit, r_fit).predict(F_apply)


S2_VARIANTS = {'S0_exact': ('exact', linear_fit_predict), 'S1_fuzzy': ('fuzzy', linear_fit_predict),
               'S2_decay': ('decay', linear_fit_predict), 'S3_all_linear': ('all', linear_fit_predict),
               'S4_all_gbm': ('all', gbm_fit_predict)}
rows, S2_OOF, S2_TEST = [], {}, {}
for name, (kind, fn) in S2_VARIANTS.items():
    F, FT, names = FEATS[kind]
    pred_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        pred_cf[m] = p1_cf[m] + fn(F[fit], r1[fit], F[m])
    S2_OOF[name], S2_TEST[name] = pred_cf, t1 + fn(F, r1, FT)
    rows.append({'variant': name, 'n_features': len(names), 'crossfit_rmse': rmse(y, pred_cf),
                 'gain_vs_twins': rmse(y, p1_cf) - rmse(y, pred_cf),
                 **{f'fold{k}': v for k, v in fold_rmse(pred_cf).items()}})
s2_table = pd.DataFrame(rows).set_index('variant').sort_values('crossfit_rmse')
print(f'{FINAL} 融合：OOF {rmse(y, base_oof):.3f} → 加双胞胎（交叉拟合）{rmse(y, p1_cf):.3f}')
display(s2_table.round(3))
s2_table.to_csv(OUTPUT_DIR / 'summary_stage2_variants.csv')
BEST_S2 = s2_table.index[0]
for name in dict.fromkeys(['S0_exact', BEST_S2]):
    path = OUTPUT_DIR / f'submission_{FINAL}_blend3_twins_{name}.csv'
    pd.DataFrame({'id': test['id'], TARGET: np.clip(S2_TEST[name], 0, None)}).to_csv(path, index=False)
    print('写出', path.name)
print(f'交叉拟合最好的第二层：{BEST_S2}')
'''),
    md('## 11. 残差诊断（为最终集成找方向）'),
    code('''
def bucket(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(str)


groups = {
    'age_half': P['age_half'].iloc[:n_train].to_numpy(),
    'ath_n': bucket(H['ath_n'].iloc[:n_train].fillna(-1), [-1, 0, 2, 3, 6, 1e9], ['未知', '1', '2', '3-5', '6+']),
    'ath_years_since_first': bucket(H['ath_years_since_first'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 100],
                                    ['未知', '0', '1', '2-4', '5+']),
    'ath_ever_abroad': H2['ath_ever_abroad'].iloc[:n_train].fillna(-1).to_numpy(),
    'ath_max_meet_states': bucket(H2['ath_max_meet_states'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 10, 1e9],
                                  ['未知', '0', '1', '2-4', '5-9', '10+']),
    'meet_n_states': bucket(M2['meet_n_states'].iloc[:n_train], [0, 1, 2, 5, 10, 1e9], ['0', '1', '2-4', '5-9', '10+']),
    'dup_exact_n>0': (D['dup_exact_n'].iloc[:n_train] > 0).to_numpy(),
    'Sex×Equipment': (train['Sex'] + '|' + train['Equipment']).to_numpy(),
    'Tested': train['Tested'].to_numpy(),
    'MeetYear 段': bucket(train['MeetYear'], [1900, 2000, 2010, 2015, 2020, 2030],
                          ['<2000', '2000s', '2010-14', '2015-19', '2020+']),
}
columns = {'R2': BLENDS['R2']['oof'], FINAL: BLENDS[FINAL]['oof'], f'{FINAL}+双胞胎+{BEST_S2}': S2_OOF[BEST_S2]}
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
print('完成。跑完告诉 Claude，本地会从 Drive 同步 OUTPUT_DIR 的结果。')
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
