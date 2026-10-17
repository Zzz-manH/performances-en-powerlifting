"""生成 09_pl_colab_gpu_feature_round2.ipynb（只生成文件并做语法检查，不在本地执行）。

数据读取与第一轮特征两个单元原样取自 07，保证折号、内层早停划分、R1 特征与 07 完全相同。
用法：/opt/anaconda3/bin/python3 tools/build_09_feature_round2.py
"""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '09_pl_colab_gpu_feature_round2.ipynb'
NB07 = nbf.read(ROOT / '07_pl_colab_gpu_feature_round1.ipynb', as_version=4)
DATA_CELL, FEATURE_CELL = NB07.cells[4].source, NB07.cells[6].source
assert 'def require' in DATA_CELL and 'FOLD_IDS' in DATA_CELL, '07 的数据单元位置变了'
assert "BLOCKS = {'P': P" in FEATURE_CELL and 'def frames_for' in FEATURE_CELL, '07 的特征单元位置变了'
SHA = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['train.csv', 'test.csv']}


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md('''
# 09 · Powerlifting：第二轮特征工程（Colab GPU，五折）

**07 的结果**（五折、内层 10% 早停，三模型 NNLS 融合）：

|变体|CV RMSE|相对 F0|公开榜|
|---|---:|---:|---:|
|F0|77.956|—|（+ 双胞胎）77.947|
|T 跨地参赛|77.912|−0.04（未达标）||
|P 记录精度|77.631|−0.33（5/5 折）||
|M 比赛背景|77.594|−0.36（5/5 折）||
|ALL = P + T + M|77.283|−0.67（5/5 折）||
|ALL + CatBoost C2|77.210|−0.75|77.337|
|+ 双胞胎后处理|76.966||**77.158**|

比赛背景（M）和记录精度（P）几乎可以叠加；CatBoost 更强的正则化（C2：L2 10、Bernoulli 0.8）再 −0.22。M 有效说明「这是一场什么样的比赛」很重要，本轮沿这个方向继续挖，并第一次加入「同一运动员」的无标签历史信息。

**本轮基线 `R1`** = 07 的 ALL（F0 + P + T + M，33 列），三个模型：LightGBM、XGBoost（v2 参数）、CatBoost **C2**。

**新特征块**（全部无标签：逐行变换或 train+test 计数，不使用 TotalKg）：

|块|内容|为什么|
|---|---|---|
|`M2` 比赛背景加强|比赛内不同州 / 国家 / 级别的个数；测试、raw、multi-ply、两位小数体重、青年（<24 岁）、大师（≥40 岁）的比例；年龄标准差；同性别平均体重；「全国性」键（联盟×年份×比赛国）的规模、比赛州数、选手州数|来自 30 个州的比赛就是全国赛，选手更强；M 已证明比赛级别信号有效|
|`D` 组别背景|同比赛×性别×装备×级别的人数，体重 / 年龄在组内的百分位；同年龄组人数；与自己 16 列完全相同 / 核心 9 列相同的其他行数|组别竞争程度；同一成绩报多个组别的多是更有竞争力的选手|
|`H` 运动员历史|近似同一运动员（性别 + 国家 + 州 + 联盟 + 出生年）在数据中的出现次数、出现年份数、距第一次 / 最后一次出现的年数（经验）、用过几种装备、体重相对自己平均的偏差；不分联盟版本的出现次数与联盟数。国家未知时为空|经验和参赛频率反映训练水平；残差诊断显示同一运动员的残差相关 0.07–0.10|
|`S` 力量系数|DOTS 系数（体重-性别标准化）|树少用几次切分就能拟合体重曲线|
|`C` 高基数 ID（只给 CatBoost）|`meet_id`（比赛）、`athlete_id`（近似运动员 + 装备）作为类别特征，由 CatBoost 的有序目标统计编码（不泄漏）|直接学「这场比赛 / 这个人」的水平；LightGBM / XGBoost 对这种 ID 容易过拟合，所以不给它们|

**流程**：① 筛选：`R1` 与每个块单独加入（`C` 只跑 CatBoost，融合时 LightGBM / XGBoost 用 `R1` 的结果）；② 达标的块自动合成 `ALL2` 再跑五折；③ 与 `R1` 配对比较、双胞胎后处理、导出提交文件；④ 残差诊断，为下一轮找方向。

**判读**：与 07 相同：三模型融合平均 ΔRMSE ≤ −0.05 kg 且 ≥ 4/5 折。

**操作**：Colab 选 **T4 GPU**，从上到下运行。预计 1.5–2 小时，可续跑。若 08 还在跑，等它结束再开（Colab 一般一次只能用一个 GPU）。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(f'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import warnings

try:
    from google.colab import drive
except ModuleNotFoundError:
    drive = None
if drive is not None:
    drive.mount('/content/drive')

PROJECT_DIR = Path(os.environ.get(
    'PL_PROJECT_DIR',
    '/content/drive/MyDrive/Colab Notebooks/performances-en-powerlifting',
)).expanduser().resolve()
if not PROJECT_DIR.is_dir():
    raise FileNotFoundError(f'找不到项目目录：{{PROJECT_DIR}}；请修改 PROJECT_DIR')

CATBOOST_VERSION = '1.2.10'
try:
    installed_catboost = importlib.metadata.version('catboost')
except importlib.metadata.PackageNotFoundError:
    installed_catboost = None
if installed_catboost != CATBOOST_VERSION:
    if 'catboost' in sys.modules:
        raise RuntimeError('当前运行时已经导入另一版 CatBoost；请重启运行时后从头运行')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', f'catboost=={{CATBOOST_VERSION}}'])
    importlib.invalidate_caches()

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from IPython.display import display
from scipy.optimize import nnls
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings('ignore', category=UserWarning)
if catboost.__version__ != CATBOOST_VERSION:
    raise RuntimeError(f'需要 CatBoost {{CATBOOST_VERSION}}；实际为 {{catboost.__version__}}')
if shutil.which('nvidia-smi') is None:
    raise RuntimeError('未找到 nvidia-smi：请在 Colab 选择 T4 GPU 后从头运行')
gpu_probe = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                           text=True, capture_output=True, check=False)
if gpu_probe.returncode != 0 or not gpu_probe.stdout.strip():
    raise RuntimeError('NVIDIA GPU 不可用：请在 Colab 选择 T4 GPU 后从头运行')
GPU_NAME = gpu_probe.stdout.strip().splitlines()[0]

EXPERIMENT_ID = 'feature_round2_v1'
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
INNER_STOP_FRACTION = 0.10
MAX_ROUNDS = 20000
MODELS = ['lgb', 'xgb', 'catboost']    # 每折内的运行顺序：快的先跑
RUN_ONLY = None                        # 例如 ['R1', 'H']；None 表示全部筛选变体
KEEP_DELTA = -0.05                     # kg：平均 ΔRMSE ≤ −0.05 且 ≥ 4/5 折更好才保留
EXPECTED_SHA256 = {SHA!r}
VERSIONS = {{'catboost': catboost.__version__, 'xgboost': xgb.__version__, 'lightgbm': lgb.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}}
print('PROJECT_DIR :', PROJECT_DIR)
print('OUTPUT_DIR  :', OUTPUT_DIR)
print('GPU         :', GPU_NAME)
print('版本         :', VERSIONS)
'''),
    md('## 2. 读取并核对数据（与 07 相同）'),
    code(DATA_CELL),
    md('## 3. 第一轮特征（与 07 相同；R1 = F0 + P + T + M）'),
    code(FEATURE_CELL),
    md('## 4. 第二轮特征块'),
    code('''
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
'''),
    md('## 5. 模型与预检\n\nLightGBM、XGBoost 用 v2 参数；CatBoost 用 07 的 C2（depth 8、L2 10、Bernoulli 0.8）。预检在 3,000 行上每个（模型, 变体）训练 30 轮。'),
    code('''
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=20, verbose=-1)
XGB_PARAMS = dict(learning_rate=0.03, max_depth=8, min_child_weight=10.0, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=1.0, tree_method='hist', device='cuda', enable_categorical=True,
                  max_cat_to_onehot=1, eval_metric='rmse')
CB_PARAMS = dict(learning_rate=0.05, depth=8, l2_leaf_reg=10.0, bootstrap_type='Bernoulli', subsample=0.8,
                 border_count=128)                                   # 07 的 C2
PATIENCE = {'lgb': 200, 'xgb': 200, 'catboost': 300}


def fit_predict(model, Xa, ya, Xs, ys, Xv, Xt, cat_cols, rounds=MAX_ROUNDS):
    """在 (Xa, ya) 上训练、用 (Xs, ys) 早停；返回 (外层验证预测, 测试预测, 最佳轮数)。"""
    if model == 'lgb':
        m = lgb.LGBMRegressor(n_estimators=rounds, random_state=SEED, n_jobs=-1, **LGB_PARAMS)
        m.fit(Xa, ya, eval_set=[(Xs, ys)], callbacks=[lgb.early_stopping(PATIENCE['lgb'], verbose=False)])
        best = int(m.best_iteration_ or rounds)
        return m.predict(Xv, num_iteration=best), m.predict(Xt, num_iteration=best), best
    if model == 'xgb':
        m = xgb.XGBRegressor(n_estimators=rounds, random_state=SEED, early_stopping_rounds=PATIENCE['xgb'],
                             **XGB_PARAMS)
        m.fit(Xa, ya, eval_set=[(Xs, ys)], verbose=False)
        best = int(m.best_iteration) + 1
        return m.predict(Xv, iteration_range=(0, best)), m.predict(Xt, iteration_range=(0, best)), best
    if model == 'catboost':
        m = catboost.CatBoostRegressor(iterations=rounds, loss_function='RMSE', eval_metric='RMSE',
                                       od_type='Iter', od_wait=PATIENCE['catboost'], random_seed=SEED,
                                       verbose=0, task_type='GPU', devices='0', allow_writing_files=False,
                                       **CB_PARAMS)
        m.fit(catboost.Pool(Xa, ya, cat_features=cat_cols), eval_set=catboost.Pool(Xs, ys, cat_features=cat_cols),
              use_best_model=True)
        best = int(m.get_best_iteration()) + 1
        return m.predict(Xv), m.predict(Xt), best
    raise ValueError(model)


rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for variant in SCREEN:
    for model in models_for(variant):
        Xtr, Xte, cat_cols = frames2(variant, model)
        try:
            p_val, p_test, _ = fit_predict(model, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                           Xte.iloc[:500], cat_cols, rounds=30)
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
            require(len(p_val) == len(pv) and len(p_test) == 500, '预测长度不符')
            print(f'预检通过：{model:8s} {variant:3s}（{Xtr.shape[1]} 列）RMSE@30轮 {rmse(y[pv], p_val):.1f}')
        except Exception as exc:
            failed[(model, variant)] = repr(exc)
            print(f'预检失败，跳过：{model} {variant} → {exc!r}')
require(not any(v == 'R1' for (_, v) in failed), '基线 R1 预检失败，请检查环境')
'''),
    md('## 6. 筛选：R1 与每个新块单独加入（五折，可续跑）'),
    code('''
CONFIG = {'purpose': 'feature round 2 on top of 07 ALL; inner 10% early stop, score early-stopped model on outer fold',
          'seed': SEED, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'r1_blocks': R1_BLOCKS, 'screen_blocks': {k: v for k, v in VARIANT_BLOCKS.items()},
          'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def spec_of(model, variant):
    Xtr, _, cat_cols = frames2(variant, model)
    params = {'lgb': LGB_PARAMS, 'xgb': XGB_PARAMS, 'catboost': CB_PARAMS}[model]
    return {'model': model, 'variant': variant, 'blocks': list(VARIANT_BLOCKS[variant]),
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
    Xtr, Xte, cat_cols = frames2(variant, model)
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
                if model not in models_for(variant) or (model, variant) in failed:
                    continue
                result, how = run_one(model, variant, fold)
                print(f'fold {fold}  {model:8s} {variant:4s}  RMSE {result["rmse"]:.3f}  '
                      f'最佳轮数 {result["best_iteration"]:5d}  ({how})')
        print(f'—— fold {fold} 完成，本次运行用时 {(time.perf_counter() - fold_started) / 60:.1f} 分钟')


run_variants(SCREEN)
'''),
    md('## 7. 与 R1 配对比较（单模型 + 三模型融合），自动合成 ALL2'),
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


def paired_table(preds, base='R1'):
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
    """三模型 NNLS 融合；某模型不跑该变体时（如 C 只改 CatBoost）用它的 R1 结果。"""
    loaded = []
    for model in MODELS:
        item = load_oof(model, variant if model in models_for(variant) else 'R1')
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
    for variant in VARIANT_BLOCKS:
        item = load_oof(model, variant) if model in models_for(variant) else None
        if item is not None:
            preds[variant] = item[0]
    if 'R1' in preds:
        table = paired_table(preds)
        part = results[results['model'] == model].groupby('variant')
        table['mean_best_iter'] = part['best_iteration'].mean()
        table['minutes'] = part['seconds'].sum() / 60
        print(f'\\n=== {model}')
        display(table.round(3))
        table.to_csv(OUTPUT_DIR / f'summary_single_{model}.csv')

BLENDS = {}
for variant in VARIANT_BLOCKS:
    item = blend_variant(variant)
    if item is not None:
        BLENDS[variant] = item
require('R1' in BLENDS, '三模型的 R1 还没有全部完成')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cb'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\\n=== 三模型 NNLS 融合（主要判读依据）')
display(blend_table.round(3))
blend_table.to_csv(OUTPUT_DIR / 'summary_blend_screen.csv')

KEPT = [v for v in ['M2', 'D', 'H', 'S', 'C'] if v in blend_table.index and bool(blend_table.loc[v, 'keep'])]
VARIANT_BLOCKS['ALL2'] = KEPT
MODELS_FOR['ALL2'] = ['catboost'] if KEPT and all(b in CB_ONLY for b in KEPT) else MODELS
print('达标的新块：', KEPT or '无', '→ ALL2 =', ['R1'] + KEPT)
'''),
    md('## 8. ALL2：R1 + 全部达标的块（五折）→ 双胞胎后处理 → 导出提交文件'),
    code('''
if KEPT:
    for model in models_for('ALL2'):
        Xtr, Xte, cat_cols = frames2('ALL2', model)
        p_val, p_test, _ = fit_predict(model, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                       Xte.iloc[:500], cat_cols, rounds=30)          # ALL2 的预检
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'ALL2 {model} 预检失败')
    run_variants(['ALL2'])
    BLENDS['ALL2'] = blend_variant('ALL2')
    final_table = paired_table({v: BLENDS[v]['oof'] for v in ['R1'] + KEPT + ['ALL2']})
    display(final_table.round(3))
    final_table.to_csv(OUTPUT_DIR / 'summary_blend_all2.csv')
    FINAL = 'ALL2'
else:
    print('没有达标的新块，保持 R1。')
    FINAL = 'R1'

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
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / np.sum(d * d), 0, 1)) if mask.any() else 0.0


def apply_twins(pred, mate, level, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


LEVEL, MATE = twins()
T_LEVEL, T_MATE = twins(row_keys(test, RAW_FEATURES), row_keys(test, CORE))
half = (row_keys(train, CORE) % 2).astype(int)
twin_rows = []
for name in ['R1', FINAL] if FINAL != 'R1' else ['R1']:
    oof, test_pred = BLENDS[name]['oof'], BLENDS[name]['test']
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    crossfit = oof.copy()
    for h in (0, 1):
        a = {lv: fit_alpha((half != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        crossfit[half == h] = apply_twins(oof[half == h], MATE[half == h], LEVEL[half == h], a)
    fixed = apply_twins(test_pred, T_MATE, T_LEVEL, alphas)
    twin_rows.append({'name': name, 'oof_rmse': rmse(y, oof), 'oof_rmse_twinfix_crossfit': rmse(y, crossfit),
                      'alpha_exact': alphas['exact'], 'alpha_core': alphas['core']})
    for suffix, pred in [('', test_pred), ('_twinfix', fixed)]:
        path = OUTPUT_DIR / f'submission_{name}_blend3{suffix}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('写出', path.name)
    np.save(OUTPUT_DIR / f'oof_{name}_blend3.npy', oof)
twin_table = pd.DataFrame(twin_rows).set_index('name')
display(twin_table.round(3))
twin_table.to_csv(OUTPUT_DIR / 'summary_twins.csv')
'''),
    md('## 9. 残差诊断：还有哪些分组没被学到（为下一轮找方向）\n\n分组残差均值（真实 − 预测，kg）。绝对值大、行数多的分组就是下一轮的候选方向。'),
    code('''
def bucket(values, edges, labels):
    return pd.cut(values, edges, labels=labels, right=False).astype(str)


groups = {
    'age_half': P['age_half'].iloc[:n_train].to_numpy(),
    'bw_precision': P['bw_precision'].iloc[:n_train].to_numpy(),
    'meet_n 八分位': pd.qcut(M['meet_n'].iloc[:n_train], 8, labels=False, duplicates='drop').to_numpy(),
    'meet_n_states': bucket(M2['meet_n_states'].iloc[:n_train], [0, 1, 2, 5, 10, 1000], ['0', '1', '2-4', '5-9', '10+']),
    'ath_n': bucket(H['ath_n'].iloc[:n_train].fillna(-1), [-1, 0, 2, 3, 6, 1e9], ['未知', '1', '2', '3-5', '6+']),
    'ath_years_since_first': bucket(H['ath_years_since_first'].iloc[:n_train].fillna(-1), [-1, 0, 1, 2, 5, 100],
                                    ['未知', '0', '1', '2-4', '5+']),
    'dup_exact_n>0': (D['dup_exact_n'].iloc[:n_train] > 0).to_numpy(),
    'div_n': bucket(D['div_n'].iloc[:n_train], [1, 2, 4, 8, 1e9], ['1', '2-3', '4-7', '8+']),
    'Sex×Equipment': (train['Sex'] + '|' + train['Equipment']).to_numpy(),
    'Tested': train['Tested'].to_numpy(),
    'MeetYear 段': bucket(train['MeetYear'], [1900, 2000, 2010, 2015, 2020, 2030],
                          ['<2000', '2000s', '2010-14', '2015-19', '2020+']),
}
frames = []
for name in ['R1', FINAL] if FINAL != 'R1' else ['R1']:
    res = y - BLENDS[name]['oof']
    rows = []
    for g, key in groups.items():
        part = pd.DataFrame({'k': key, 'r': res}).groupby('k')['r'].agg(['size', 'mean'])
        part.index = pd.MultiIndex.from_product([[g], part.index.astype(str)], names=['signal', 'level'])
        rows.append(part)
    frames.append(pd.concat(rows).rename(columns={'mean': name}))
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
