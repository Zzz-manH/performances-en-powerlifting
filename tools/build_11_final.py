"""生成 11_pl_colab_gpu_final.ipynb（只生成文件并做语法检查，不在本地执行）。

数据、三轮特征的单元原样取自 07 / 09 / 10，保证特征定义与之前完全相同。
用法：/opt/anaconda3/bin/python3 tools/build_11_final.py
"""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '11_pl_colab_gpu_final.ipynb'
NB07 = nbf.read(ROOT / '07_pl_colab_gpu_feature_round1.ipynb', as_version=4)
NB09 = nbf.read(ROOT / '09_pl_colab_gpu_feature_round2.ipynb', as_version=4)
NB10 = nbf.read(ROOT / '10_pl_colab_gpu_feature_round3.ipynb', as_version=4)
DATA_CELL, FEATURE_CELL = NB07.cells[4].source, NB07.cells[6].source
ROUND2_CELL, ROUND3_CELL = NB09.cells[8].source, NB10.cells[10].source
assert 'def require' in DATA_CELL and 'FOLD_IDS' in DATA_CELL, '07 的数据单元位置变了'
assert "BLOCKS = {'P': P" in FEATURE_CELL, '07 的特征单元位置变了'
assert "BLOCKS2 = {'M2': M2" in ROUND2_CELL, '09 的第二轮特征单元位置变了'
assert "ROUND3 = {'H2': H2, 'E': E}" in ROUND3_CELL and 'def frames3' in ROUND3_CELL, '10 的第三轮特征单元位置变了'
SHA = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['train.csv', 'test.csv']}


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md('''
# 11 · Powerlifting：最终集成（Colab GPU）

**10 的结果**（五折单种子，三模型融合）

|实验|CV RMSE|公开榜|
|---|---:|---:|
|R2（09 的 ALL2）|77.079||
|+ H2 运动员生涯|76.993（−0.086，5/5 折）||
|+ E 比赛经验结构|77.007（−0.072，4/5 折）||
|+ PM 按模型挑特征|76.970（−0.110，5/5 折）||
|**ALL3 = 全部**|**76.876**（−0.203，5/5 折）||
|+ 双胞胎|76.649||
|+ S0 原版运动员平滑|76.539|76.775|
|+ **S3 合并平滑**（精确 + 出生年 ±1 + 时间衰减，线性）|**76.510**|**76.716**|

单模型在 ALL3 上：CatBoost 77.505、LightGBM 77.708、XGBoost 78.457。小 LightGBM 做第二层会过拟合（+0.21），不用。

**最终方案**

|项目|设置|
|---|---|
|特征|10 的 ALL3，按模型不同：CatBoost / cb_rel = R2 + H2 + E + M2 + 比赛 / 运动员 ID；XGBoost = R2 + H2 + E + M2 + S + D；LightGBM / lgb_rel / TabM = R2 + H2 + E|
|模型|LightGBM、XGBoost、CatBoost C2、`cb_rel`（DOTS 相对力量目标，C2 参数）、`lgb_rel`、TabM（分段线性嵌入）|
|验证|5 折 × 3 个种子（42、7、2024，每个种子不同的划分），外层训练部分内留 10% 早停；每个模型的 OOF 取种子平均|
|测试预测|**折平均**：每个模型 15 个折模型平均。**全量重训**：树模型在全部 105,000 行上重训（轮数 = 该种子 CV 平均最佳轮数 × 1.2），每个种子一次；TabM 没有全量版，用折平均。**混合**：两者各半|
|融合|全局 NNLS；用种子 42 五折嵌套给出诚实分数|
|第二层|双胞胎 + S3 合并平滑（10 中交叉拟合最好）|
|附加：数据清洗实验|训练时去掉疑似错误或不完整的成绩（男子总成绩 < 2.5 × 体重、女子 < 1.8 × 体重、年龄 < 5 岁），评估仍用全部行。种子 42，三个树模型。只作报告证据，不影响提交文件|

**输出**（`artifacts/final_v1/`）：`submission_final_full_final.csv`（推荐）、`submission_final_hybrid_final.csv`（备选）、`submission_final_fold_avg_final.csv`，以及不加第二层的 `*_base.csv`。HMDA 中全量重训的公开榜最好，混合版其次。

**操作**：Colab 选 **T4 GPU**，从上到下运行。每个种子约 40 分钟（TabM 约占一半）加全量重训约 7 分钟，三个种子共约 2.5 小时；数据清洗实验约 10 分钟。可续跑。**时间不够时**：任意一个种子跑完（第 6 节打印「种子 … 完成」）后可以中断，然后从第 7 节开始往下运行，会用已经完成的种子。
'''),
    md('## 1. 环境、GPU 与配置'),
    code(f'''
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
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
to_install = []
try:
    installed_catboost = importlib.metadata.version('catboost')
except importlib.metadata.PackageNotFoundError:
    installed_catboost = None
if installed_catboost != CATBOOST_VERSION:
    if 'catboost' in sys.modules:
        raise RuntimeError('当前运行时已经导入另一版 CatBoost；请重启运行时后从头运行')
    to_install.append(f'catboost=={{CATBOOST_VERSION}}')
try:
    importlib.metadata.version('pytabkit')
except importlib.metadata.PackageNotFoundError:
    to_install.append('pytabkit')
if to_install:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *to_install])
    importlib.invalidate_caches()

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from IPython.display import display
from scipy.optimize import nnls
from sklearn.model_selection import KFold, train_test_split

try:
    from pytabkit import TabM_D_Regressor
    PYTABKIT_ERROR = None
except Exception as exc:                       # 安装或导入失败时其余模型照常运行
    TabM_D_Regressor = None
    PYTABKIT_ERROR = repr(exc)

warnings.filterwarnings('ignore', category=UserWarning)
if catboost.__version__ != CATBOOST_VERSION:
    raise RuntimeError(f'需要 CatBoost {{CATBOOST_VERSION}}；实际为 {{catboost.__version__}}')
if shutil.which('nvidia-smi') is None or not torch.cuda.is_available():
    raise RuntimeError('没有可用的 GPU：请在 Colab 选择 T4 GPU 后从头运行')
GPU_NAME = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          text=True, capture_output=True, check=False).stdout.strip().splitlines()[0]

EXPERIMENT_ID = 'final_v1'
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42                                  # 07–10 的五折划分种子（第一个种子）
SEEDS = [42, 7, 2024]                      # 每个种子：不同的五折划分、内层早停划分和模型随机性
N_FOLDS = 5
INNER_STOP_FRACTION = 0.10
MAX_ROUNDS = 20000
FINAL_MODELS = ['lgb', 'xgb', 'lgb_rel', 'catboost', 'cb_rel', 'tabm']   # 每折内的运行顺序：快的先跑
FULL_REFIT = True                          # 树模型在全部训练行上重训
REFIT_FACTOR = 1.2                         # 全量重训轮数 = CV 平均最佳轮数 × 1.2（数据多约 39%）
RUN_CLEANING_TEST = True                   # 第 9 节：数据清洗实验（只作报告证据）
MODELS = ['lgb', 'xgb', 'catboost']        # 09 / 10 特征单元使用的三个模型族名
RUN_ONLY = None                            # 09 / 10 特征单元需要的变量，这里不用
KEEP_DELTA = -0.05
EXPECTED_SHA256 = {SHA!r}
VERSIONS = {{'catboost': catboost.__version__, 'xgboost': xgb.__version__, 'lightgbm': lgb.__version__,
            'torch': torch.__version__, 'pytabkit': None if PYTABKIT_ERROR else importlib.metadata.version('pytabkit'),
            'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}}
print('PROJECT_DIR :', PROJECT_DIR)
print('OUTPUT_DIR  :', OUTPUT_DIR)
print('GPU         :', GPU_NAME)
print('版本         :', VERSIONS)
if PYTABKIT_ERROR:
    print('pytabkit 不可用，TabM 会在预检中跳过：', PYTABKIT_ERROR)
'''),
    md('## 2. 读取并核对数据（与 07 相同）'),
    code(DATA_CELL),
    md('## 3. 第一轮特征（与 07 相同）'),
    code(FEATURE_CELL),
    md('## 4. 第二轮特征（与 09 相同）'),
    code(ROUND2_CELL),
    md('## 5. 第三轮特征（与 10 相同）与各模型的最终特征表'),
    code(ROUND3_CELL + '''

# ---- 最终特征：10 的 ALL3（三个块都达标）----
_FRAMES3.clear()
VARIANTS3['ALL3'] = ['H2', 'E', 'PM']
FAMILY = {'lgb': 'lgb', 'lgb_rel': 'lgb', 'tabm': 'lgb', 'xgb': 'xgb', 'catboost': 'catboost', 'cb_rel': 'catboost'}
FRAMES = {m: frames3('ALL3', FAMILY[m]) for m in FINAL_MODELS}
for m, (Xtr_m, Xte_m, cats_m) in FRAMES.items():
    print(f'{m:9s} {Xtr_m.shape[1]} 列（{len(cats_m)} 个类别列），块：{blocks_for("ALL3", FAMILY[m])}')
'''),
    md('''
## 6. 模型、预检、5 折 × 种子训练 + 全量重训（可续跑）

- LightGBM、XGBoost：v2 参数；CatBoost：C2（depth 8、L2 10、Bernoulli 0.8）。
- `cb_rel` / `lgb_rel`：目标 = TotalKg ÷ DOTS 系数，样本权重 = 系数²，预测后乘回 kg。
- TabM：pytabkit 默认参数 + 分段线性数值嵌入（预检失败时自动去掉嵌入）；数值缺失用外层训练部分的中位数填补并加缺失标记；目标先标准化。
- 全量重训只做树模型；轮数取该种子 CV 的平均最佳轮数 × `REFIT_FACTOR`。
'''),
    code('''
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=20, verbose=-1)
XGB_PARAMS = dict(learning_rate=0.03, max_depth=8, min_child_weight=10.0, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=1.0, tree_method='hist', device='cuda', enable_categorical=True,
                  max_cat_to_onehot=1, eval_metric='rmse')
CB_PARAMS = dict(learning_rate=0.05, depth=8, l2_leaf_reg=10.0, bootstrap_type='Bernoulli', subsample=0.8,
                 border_count=128)
TABM_PARAMS = dict(device='cuda', n_cv=1, n_refit=0, verbosity=0, num_emb_type='pwl')
TABM_FALLBACK = dict(device='cuda', n_cv=1, n_refit=0, verbosity=0)
PATIENCE = {'lgb': 200, 'lgb_rel': 200, 'xgb': 200, 'catboost': 300, 'cb_rel': 300}
GBDT = [m for m in FINAL_MODELS if m != 'tabm']

SCALE = dots_scale(train['Sex'], train['BodyweightKg'])           # dots_scale 来自 09 的特征单元
SCALE_TEST = dots_scale(test['Sex'], test['BodyweightKg'])
require(min(SCALE.min(), SCALE_TEST.min()) > 0.3, 'DOTS 系数异常')

X_NN, XT_NN, CATS_NN = FRAMES['tabm']
NUM_NN = [c for c in X_NN.columns if c not in CATS_NN]
NAN_NN = [c for c in NUM_NN if X_NN[c].isna().any() or XT_NN[c].isna().any()]


def target_and_weight(model, rows):
    if model.endswith('_rel'):
        return y[rows] / SCALE[rows], SCALE[rows] ** 2       # 相对力量误差 × scale = kg 误差
    return y[rows], None


def gbdt_fit(model, seed, Xa, ya, wa, Xs=None, ys=None, ws=None, rounds=MAX_ROUNDS):
    """Xs 为 None 时不早停（全量重训）。返回 (模型, 最佳轮数)。"""
    cats = FRAMES[model][2]
    if model in ('lgb', 'lgb_rel'):
        m = lgb.LGBMRegressor(n_estimators=rounds, random_state=seed, n_jobs=-1, **LGB_PARAMS)
        if Xs is None:
            m.fit(Xa, ya, sample_weight=wa)
            return m, rounds
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xs, ys)], eval_sample_weight=None if ws is None else [ws],
              callbacks=[lgb.early_stopping(PATIENCE[model], verbose=False)])
        return m, int(m.best_iteration_ or rounds)
    if model == 'xgb':
        if Xs is None:
            m = xgb.XGBRegressor(n_estimators=rounds, random_state=seed, **XGB_PARAMS)
            m.fit(Xa, ya, verbose=False)
            return m, rounds
        m = xgb.XGBRegressor(n_estimators=rounds, random_state=seed, early_stopping_rounds=PATIENCE['xgb'],
                             **XGB_PARAMS)
        m.fit(Xa, ya, eval_set=[(Xs, ys)], verbose=False)
        return m, int(m.best_iteration) + 1
    common = dict(iterations=rounds, loss_function='RMSE', eval_metric='RMSE', random_seed=seed, verbose=0,
                  task_type='GPU', devices='0', allow_writing_files=False, **CB_PARAMS)
    if Xs is None:
        m = catboost.CatBoostRegressor(**common)
        m.fit(catboost.Pool(Xa, ya, cat_features=cats, weight=wa))
        return m, rounds
    m = catboost.CatBoostRegressor(od_type='Iter', od_wait=PATIENCE[model], **common)
    m.fit(catboost.Pool(Xa, ya, cat_features=cats, weight=wa),
          eval_set=catboost.Pool(Xs, ys, cat_features=cats, weight=ws), use_best_model=True)
    return m, int(m.get_best_iteration()) + 1


def gbdt_predict(model, m, best, X):
    if model in ('lgb', 'lgb_rel'):
        return m.predict(X, num_iteration=best)
    if model == 'xgb':
        return m.predict(X, iteration_range=(0, best))
    return m.predict(X)


def nn_frames(*frames):
    """数值缺失用第一个表（外层训练部分）的中位数填补并加缺失标记；类别保持 category。"""
    median = frames[0][NUM_NN].median()
    out = []
    for frame in frames:
        z = frame.copy()
        for c in NAN_NN:
            z[f'{c}_isna'] = z[c].isna().astype('float64')
        z[NUM_NN] = z[NUM_NN].fillna(median).fillna(0.0)
        out.append(z.reset_index(drop=True))
    return out


def tabm_fit_predict(seed, a, s, v, Xt, overrides=None):
    require(TabM_D_Regressor is not None, f'pytabkit 不可用：{PYTABKIT_ERROR}')
    params = dict(TABM_PARAMS, random_state=seed, **(overrides or {}))
    unknown = set(params) - set(inspect.signature(TabM_D_Regressor.__init__).parameters)
    require(not unknown, f'TabM 不支持参数 {sorted(unknown)}')
    Za, Zs, Zv, Zt = nn_frames(X_NN.iloc[a], X_NN.iloc[s], X_NN.iloc[v], Xt)
    mu, sd = y[a].mean(), y[a].std()
    m = TabM_D_Regressor(**params)
    m.fit(Za, (y[a] - mu) / sd, Zs, (y[s] - mu) / sd, cat_col_names=[c for c in Za.columns if c in CATS_NN])
    p_val = np.asarray(m.predict(Zv), dtype='float64').reshape(-1) * sd + mu
    p_test = np.asarray(m.predict(Zt), dtype='float64').reshape(-1) * sd + mu
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test


def fit_fold(model, seed, a, s, v, test_rows=None, rounds=MAX_ROUNDS, overrides=None):
    """a / s / v：内层拟合 / 内层早停 / 外层验证 的训练行号。返回 (验证预测, 测试预测, 最佳轮数或 None)。"""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale_t = SCALE_TEST if test_rows is None else SCALE_TEST[test_rows]
    if model == 'tabm':
        p_val, p_test = tabm_fit_predict(seed, a, s, v, Xt, overrides)
        return p_val, p_test, None
    ya, wa = target_and_weight(model, a)
    ys, ws = target_and_weight(model, s)
    if wa is not None:
        norm = wa.mean()
        wa, ws = wa / norm, ws / norm
    m, best = gbdt_fit(model, seed, Xtr.iloc[a], ya, wa, Xtr.iloc[s], ys, ws, rounds)
    p_val, p_test = gbdt_predict(model, m, best, Xtr.iloc[v]), gbdt_predict(model, m, best, Xt)
    if model.endswith('_rel'):
        p_val, p_test = np.asarray(p_val) * SCALE[v], np.asarray(p_test) * scale_t
    return np.asarray(p_val, dtype='float64'), np.asarray(p_test, dtype='float64'), best


def fit_full(model, seed, rounds, test_rows=None):
    """全部训练行、固定轮数、不早停。返回测试预测。"""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale_t = SCALE_TEST if test_rows is None else SCALE_TEST[test_rows]
    rows = np.arange(n_train)
    ya, wa = target_and_weight(model, rows)
    if wa is not None:
        wa = wa / wa.mean()
    m, _ = gbdt_fit(model, seed, Xtr, ya, wa, rounds=rounds)
    p_test = gbdt_predict(model, m, rounds, Xt)
    if model.endswith('_rel'):
        p_test = np.asarray(p_test) * scale_t
    return np.asarray(p_test, dtype='float64')


# ---- 预检：每个模型在 3,000 行上训练几轮；树模型再在全部行上做 30 轮的全量重训 ----
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for model in FINAL_MODELS:
    attempts = ['main', 'fallback'] if model == 'tabm' else ['main']
    for attempt in attempts:
        if attempt == 'fallback':
            TABM_PARAMS = dict(TABM_FALLBACK)
        started = time.perf_counter()
        try:
            p_val, p_test, _ = fit_fold(model, SEED, pa, ps, pv, test_rows=np.arange(500), rounds=30,
                                        overrides={'n_epochs': 3})
            require(len(p_val) == len(pv) and len(p_test) == 500, '预测长度不符')
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
            if model in GBDT and FULL_REFIT:
                p_full = fit_full(model, SEED, 30, test_rows=np.arange(500))
                require(np.isfinite(p_full).all() and len(p_full) == 500, '全量重训预测无效')
            failed.pop(model, None)
            print(f'预检通过：{model:9s}（{attempt}）RMSE@少量轮 {rmse(y[pv], p_val):.1f}，'
                  f'用时 {time.perf_counter() - started:.0f}s')
            break
        except Exception as exc:
            failed[model] = repr(exc)
            print(f'预检失败：{model}（{attempt}）→ {exc!r}')
RUN_MODELS = [m for m in FINAL_MODELS if m not in failed]
print('将运行：', RUN_MODELS, '｜跳过：', failed or '无')
require('catboost' in RUN_MODELS, 'CatBoost 预检失败，请检查环境')

# ---- 五折 × 种子 ----
CONFIG = {'purpose': 'final ensemble: ALL3 per-model features, 5-fold x seeds, inner 10% early stop, full refit',
          'seeds': SEEDS, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'refit_factor': REFIT_FACTOR, 'final_models': FINAL_MODELS, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def folds_for(seed):
    ids = np.zeros(n_train, dtype=int)
    for k, (_, valid) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n_train)), start=1):
        ids[valid] = k
    return ids


FOLDS_BY_SEED = {s: folds_for(s) for s in SEEDS}
if SEED in FOLDS_BY_SEED:
    require(np.array_equal(FOLDS_BY_SEED[SEED], FOLD_IDS), '种子 42 的五折与 07–10 不一致')


def spec_of(model):
    params = {'lgb': LGB_PARAMS, 'lgb_rel': LGB_PARAMS, 'xgb': XGB_PARAMS, 'catboost': CB_PARAMS,
              'cb_rel': CB_PARAMS, 'tabm': TABM_PARAMS}[model]
    return {'model': model, 'columns': list(FRAMES[model][0].columns), 'cat_cols': list(FRAMES[model][2]),
            'params': dict(params), 'target': 'dots_relative_weighted' if model.endswith('_rel') else 'kg'}


def save_result(stem, arrays, info):
    np.savez(OUTPUT_DIR / f'{stem}.npz', **arrays)
    info = {**info, 'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUTPUT_DIR / f'{stem}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


def load_cached(stem, spec):
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{stem} 的定义已改变；请改 EXPERIMENT_ID')
        return saved
    return None


def run_cv(model, seed, fold):
    stem, spec = f'{model}__s{seed}__fold{fold}', spec_of(model)
    saved = load_cached(stem, spec)
    if saved is not None:
        return saved, 'cache'
    ids = FOLDS_BY_SEED[seed]
    fit_idx, valid_idx = np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=seed)
    started = time.perf_counter()
    p_val, p_test, best = fit_fold(model, seed, inner_fit, inner_stop, valid_idx)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val, 'test': p_test},
                       {'model': model, 'seed': seed, 'fold': fold, 'spec': spec, 'best_iteration': best,
                        'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started})
    return info, f'{info["seconds"]:.0f}s'


def run_full(model, seed):
    stem, spec = f'{model}__s{seed}__full', spec_of(model)
    saved = load_cached(stem, spec)
    if saved is not None:
        return saved, 'cache'
    bests = [json.loads((OUTPUT_DIR / f'{model}__s{seed}__fold{k}.json').read_text(encoding='utf-8'))['best_iteration']
             for k in range(1, N_FOLDS + 1)]
    rounds = int(round(np.mean(bests) * REFIT_FACTOR))
    started = time.perf_counter()
    p_test = fit_full(model, seed, rounds)
    require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    info = save_result(stem, {'test': p_test}, {'model': model, 'seed': seed, 'spec': spec, 'rounds': rounds,
                                                'cv_best_iterations': bests,
                                                'seconds': time.perf_counter() - started})
    return info, f'{info["seconds"]:.0f}s'


for seed in SEEDS:
    seed_started = time.perf_counter()
    for fold in range(1, N_FOLDS + 1):
        for model in RUN_MODELS:
            info, how = run_cv(model, seed, fold)
            best = info['best_iteration']
            print(f'种子 {seed:4d} fold {fold}  {model:9s} RMSE {info["rmse"]:.3f}'
                  + (f'  最佳轮数 {best}' if best else '') + f'  ({how})')
    if FULL_REFIT:
        for model in RUN_MODELS:
            if model in GBDT:
                info, how = run_full(model, seed)
                print(f'种子 {seed:4d} 全量重训 {model:9s} 轮数 {info["rounds"]}  ({how})')
    print(f'==== 种子 {seed} 完成，本次运行用时 {(time.perf_counter() - seed_started) / 60:.1f} 分钟 ====')
'''),
    md('## 7. 每个模型的成绩、NNLS 融合（诚实的嵌套分数）与三种测试预测\n\n只使用五折全部完成的种子；中途中断后从这里开始运行也可以。'),
    code('''
def load_seed(model, seed):
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{model}__s{seed}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{model} 种子 {seed} 的 OOF 不完整')
    return oof, test_pred


def fold_rmse(pred, ids=FOLD_IDS):
    return pd.Series({k: rmse(y[ids == k], pred[ids == k]) for k in range(1, N_FOLDS + 1)})


PER_MODEL = {}
rows = []
for model in FINAL_MODELS:
    loaded = {s: load_seed(model, s) for s in SEEDS}
    loaded = {s: v for s, v in loaded.items() if v is not None}
    if not loaded:
        continue
    oof = np.mean([o for o, _ in loaded.values()], axis=0)
    fold_avg = np.mean([t for _, t in loaded.values()], axis=0)
    fulls = [np.load(OUTPUT_DIR / f'{model}__s{s}__full.npz')['test'] for s in loaded
             if (OUTPUT_DIR / f'{model}__s{s}__full.npz').is_file()]
    full = np.mean(fulls, axis=0) if fulls else fold_avg          # TabM 没有全量重训，用折平均
    PER_MODEL[model] = {'oof': oof, 'fold_avg': fold_avg, 'full': full}
    rows.append({'model': model, 'seeds': list(loaded), 'full_refits': len(fulls),
                 **{f'oof_seed{s}': rmse(y, o) for s, (o, _) in loaded.items()},
                 'oof_seed_avg': rmse(y, oof)})
model_table = pd.DataFrame(rows).set_index('model').sort_values('oof_seed_avg')
display(model_table.round(3))
model_table.to_csv(OUTPUT_DIR / 'summary_models.csv')

NAMES = list(PER_MODEL)
P = np.column_stack([PER_MODEL[n]['oof'] for n in NAMES])
WEIGHTS, _ = nnls(P, y)
NESTED = np.zeros(n_train)
for k in range(1, N_FOLDS + 1):                       # 诚实分数：每一折的权重只用其他四折拟合
    fit, va = FOLD_IDS != k, FOLD_IDS == k
    w_k, _ = nnls(P[fit], y[fit])
    NESTED[va] = P[va] @ w_k
BLEND_TEST = {kind: np.column_stack([PER_MODEL[n][kind] for n in NAMES]) @ WEIGHTS for kind in ['fold_avg', 'full']}
BLEND_TEST['hybrid'] = 0.5 * (BLEND_TEST['fold_avg'] + BLEND_TEST['full'])
print('NNLS 权重：', dict(zip(NAMES, WEIGHTS.round(3))), '｜权重和', round(float(WEIGHTS.sum()), 4))
print(f'融合 OOF：权重同一份 OOF 拟合 {rmse(y, P @ WEIGHTS):.3f}，嵌套（诚实）{rmse(y, NESTED):.3f}')
print('三种测试预测之间的平均绝对差：',
      {f'{a}-{b}': round(float(np.abs(BLEND_TEST[a] - BLEND_TEST[b]).mean()), 2)
       for a, b in [('fold_avg', 'full'), ('fold_avg', 'hybrid'), ('full', 'hybrid')]})
pd.DataFrame({'model': NAMES, 'weight': WEIGHTS}).to_csv(OUTPUT_DIR / 'summary_weights.csv', index=False)
'''),
    md('''
## 8. 第二层：双胞胎 + S3 合并平滑（与 10 相同的做法）→ 导出提交文件

双胞胎：训练集中 16 列完全相同（exact）或核心 9 列相同（core）的行，把预测往其真实成绩拉 `alpha` 倍。
S3：同一运动员（A0–A3 键）的精确出生年、出生年 ±1、时间衰减三种收缩残差均值，加比赛键，线性组合。
全部按「性别 + 国家 + 州」分两半交叉拟合评估。
'''),
    code('''
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


def s3_features(r):
    cols_tr, cols_te = [], []
    for builder in (f_exact, f_fuzzy, f_decay):
        for k in PREFIX:
            a, b = builder(k, r)
            cols_tr.append(a * VALID_TR)
            cols_te.append(b * VALID_TE)
    a, b = f_meet(r)
    cols_tr.append(a)
    cols_te.append(b)
    return np.column_stack(cols_tr), np.column_stack(cols_te)


# 第二层只依赖训练行的 OOF（嵌套融合），对三种测试预测使用同一组 alpha / 系数
alphas = {lv: fit_alpha(LEVEL == lv, NESTED, MATE) for lv in ['exact', 'core']}
p1 = apply_twins(NESTED, MATE, LEVEL, alphas)
p1_cf = NESTED.copy()
for h in (0, 1):
    a_h = {lv: fit_alpha((HALF != h) & (LEVEL == lv), NESTED, MATE) for lv in ['exact', 'core']}
    m = HALF == h
    p1_cf[m] = apply_twins(NESTED[m], MATE[m], LEVEL[m], a_h)
r1 = y - p1
F, FT = s3_features(r1)
p2_cf = p1_cf.copy()
for h in (0, 1):
    fit, m = HALF != h, HALF == h
    beta_h = np.linalg.lstsq(F[fit], r1[fit], rcond=None)[0]
    p2_cf[m] = p1_cf[m] + F[m] @ beta_h
BETA = np.linalg.lstsq(F, r1, rcond=None)[0]
STAGE2_ADJ = FT @ BETA
stage_table = pd.DataFrame([{'step': '融合（嵌套）', 'oof_rmse': rmse(y, NESTED)},
                            {'step': '+ 双胞胎（交叉拟合）', 'oof_rmse': rmse(y, p1_cf)},
                            {'step': '+ S3 平滑（交叉拟合）', 'oof_rmse': rmse(y, p2_cf)}]).set_index('step')
display(stage_table.round(3))
stage_table.to_csv(OUTPUT_DIR / 'summary_stage2.csv')
print('alpha：', {k: round(v, 3) for k, v in alphas.items()}, '｜测试中有双胞胎的行：', int((T_LEVEL != 'none').sum()),
      '｜S3 平均调整幅度：', round(float(np.abs(STAGE2_ADJ).mean()), 2), 'kg')

for kind, pred in BLEND_TEST.items():
    final = apply_twins(pred, T_MATE, T_LEVEL, alphas) + STAGE2_ADJ
    for stage, values in [('base', pred), ('final', final)]:
        path = OUTPUT_DIR / f'submission_final_{kind}_{stage}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(values, 0, None)}).to_csv(path, index=False)
        print('写出', path.name)
np.save(OUTPUT_DIR / 'oof_nested_blend.npy', NESTED)
np.save(OUTPUT_DIR / 'oof_final_crossfit.npy', p2_cf)
'''),
    md('''
## 9. 数据清洗实验（只作报告证据，不影响提交文件）

问题：训练时去掉「疑似错误或不完整」的成绩，模型会不会更好？
- 疑似行：男子总成绩 < 2.5 × 体重、女子（含 Mx）< 1.8 × 体重（多为只完成单项或失败的比赛），或年龄 < 5 岁。只用训练行自己的标签判断。
- 做法：种子 42、同样的五折和内层划分，只把疑似行从内层拟合和早停集合中去掉；**外层验证仍然用全部行**评分（测试集里同样有这类行）。
- 与第 6 节种子 42 的同模型结果逐折配对比较。
'''),
    code('''
if RUN_CLEANING_TEST:
    ratio = y / train['BodyweightKg'].to_numpy(dtype='float64')
    SUSPICIOUS = (np.where(train['Sex'].to_numpy() == 'M', ratio < 2.5, ratio < 1.8)
                  | (train['Age'].to_numpy(dtype='float64') < 5))
    print(f'疑似行：{int(SUSPICIOUS.sum())}（{SUSPICIOUS.mean():.2%}）')
    clean_rows = []
    for model in [m for m in ['lgb', 'xgb', 'catboost'] if m in RUN_MODELS]:
        if not all((OUTPUT_DIR / f'{model}__s{SEED}__fold{k}.npz').is_file() for k in range(1, N_FOLDS + 1)):
            print(f'{model}：种子 {SEED} 的五折还没跑完，跳过对照')
            continue
        base_folds, clean_folds, base_normal, clean_normal = [], [], [], []
        for fold in range(1, N_FOLDS + 1):
            stem, spec = f'clean_{model}__s{SEED}__fold{fold}', {**spec_of(model), 'cleaning': 'drop_suspicious'}
            saved = load_cached(stem, spec)
            ids = FOLDS_BY_SEED[SEED]
            fit_idx, valid_idx = np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
            if saved is None:
                inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)
                inner_fit, inner_stop = inner_fit[~SUSPICIOUS[inner_fit]], inner_stop[~SUSPICIOUS[inner_stop]]
                started = time.perf_counter()
                p_val, _, best = fit_fold(model, SEED, inner_fit, inner_stop, valid_idx, test_rows=np.arange(10))
                save_result(stem, {'valid_idx': valid_idx, 'val': p_val},
                            {'model': model, 'fold': fold, 'spec': spec, 'best_iteration': best,
                             'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started})
            clean_val = np.load(OUTPUT_DIR / f'{stem}.npz')['val']
            base_val = np.load(OUTPUT_DIR / f'{model}__s{SEED}__fold{fold}.npz')['val']
            normal = ~SUSPICIOUS[valid_idx]
            base_folds.append(rmse(y[valid_idx], base_val))
            clean_folds.append(rmse(y[valid_idx], clean_val))
            base_normal.append(rmse(y[valid_idx][normal], base_val[normal]))
            clean_normal.append(rmse(y[valid_idx][normal], clean_val[normal]))
        delta = np.array(clean_folds) - np.array(base_folds)
        clean_rows.append({'model': model, 'base_all_rows': np.mean(base_folds), 'clean_all_rows': np.mean(clean_folds),
                           'delta_all_rows': delta.mean(), 'folds_better': int((delta < 0).sum()),
                           'base_normal_rows': np.mean(base_normal), 'clean_normal_rows': np.mean(clean_normal)})
    clean_table = pd.DataFrame(clean_rows).set_index('model')
    display(clean_table.round(3))
    clean_table.to_csv(OUTPUT_DIR / 'summary_cleaning_test.csv')
    print('delta_all_rows < 0 表示去掉疑似行后在全部验证行上更好；normal_rows 只看非疑似的验证行。')
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
