"""生成 08_pl_colab_gpu_model_zoo.ipynb（只生成文件并做语法检查，不在本地执行）。

数据读取与特征块两个单元原样取自 07，保证特征、折号、内层早停划分与 07 完全相同。
用法：/opt/anaconda3/bin/python3 tools/build_08_model_zoo.py
"""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '08_pl_colab_gpu_model_zoo.ipynb'
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
# 08 · Powerlifting：其他模型 + 堆叠 + 第二层「同一运动员」平滑（Colab GPU，五折）

**先说预期**：v2 的四类模型（LightGBM、XGBoost、CatBoost、MLP）单独都在 78.1–78.4，OOF 相关 0.990–0.995。四种完全不同的算法几乎学到同一个函数，说明瓶颈主要在**特征里的信息量**：预测误差约 77 kg，大部分来自特征描述不了的个人训练水平。换模型通常只能靠「错误不同」在融合里再降 0.1–0.3。能明显降分的还是数据结构：双胞胎、记录精度（07）、同一运动员（本 notebook 第 7 节）。

|模型|为什么可能有用|
|---|---|
|`realmlp`：RealMLP-TD（pytabkit，NeurIPS 2024）|调好默认参数的 MLP（鲁棒缩放、数值嵌入、类别嵌入），比 v2 自写 MLP（单种子 80.05）强得多的神经网络基线|
|`tabm`：TabM（pytabkit，ICLR 2025）|一次训练 = 32 个共享权重的 MLP 集成，加分段线性数值嵌入。v2 的 MLP 单种子 80.05 → 3 种子平均 78.40，方差正是神经网络的主要问题，TabM 正好解决它|
|`cb_rel`、`lgb_rel`：「相对力量」目标|用 DOTS 公式（力量举通用的体重-性别系数）把 TotalKg 换成相对力量，模型学「这个人相对多强」，预测后再乘回 kg。按系数² 加权，所以仍在最小化 kg 的 RMSE。乘法结构与普通模型的加法结构不同，错误也不同|
|第 6 节：堆叠|同一五折嵌套比较三种融合：全局 NNLS、按性别×装备分段 NNLS、残差元模型（LightGBM）|
|第 7 节：同一运动员残差平滑|训练集中「近似同一运动员」（性别、国家、州、联盟、出生年、装备、级别）的其他成绩，残差相关 0.07–0.10。把他们收缩后的平均残差加到预测上，是双胞胎后处理的推广。只用训练标签和 OOF，按运动员键分两半交叉拟合评估；同时直接用于 v2 的 10 折融合，写出可提交的文件|

**没放进来的**：TabPFN-2.5（面向 ≤ 5 万行、建议 A100/H100，我们有 10.5 万行和 T4）；TabR（需要 conda 安装的 faiss）；AutoGluon（HMDA 中在 T4 上内存不足）；log 目标（对 kg 的 RMSE 有系统偏差）。

**协议**：与 07 完全相同（同样的五折、同样的内层 10% 早停划分，数据与特征单元原样取自 07），所以 OOF 能直接和 07 的 LightGBM / XGBoost / CatBoost 融合。特征变体默认取 07 融合最好的那个（07 没跑完就用 `F0`）。

**判读**：每个新模型看两点：单独的 RMSE，以及加入 07 融合后的 ΔRMSE（≤ −0.05 且 ≥ 4/5 折才算有用）。第 6、7 节报告的都是诚实分数（嵌套 / 交叉拟合）。

**操作**：建议 **07 跑完再跑**（会自动读取 07 的结果）。Colab 选 **T4 GPU**，从上到下运行。预检约 2–3 分钟（含安装 pytabkit），全程约 1–1.5 小时，可断线续跑。
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
from IPython.display import display
from scipy.optimize import nnls
from sklearn.model_selection import KFold, train_test_split

try:
    from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
    PYTABKIT_ERROR = None
except Exception as exc:                       # 安装或导入失败时其余模型照常运行
    RealMLP_TD_Regressor = TabM_D_Regressor = None
    PYTABKIT_ERROR = repr(exc)

warnings.filterwarnings('ignore', category=UserWarning)
if catboost.__version__ != CATBOOST_VERSION:
    raise RuntimeError(f'需要 CatBoost {{CATBOOST_VERSION}}；实际为 {{catboost.__version__}}')
if shutil.which('nvidia-smi') is None or not torch.cuda.is_available():
    raise RuntimeError('没有可用的 GPU：请在 Colab 选择 T4 GPU 后从头运行')
GPU_NAME = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          text=True, capture_output=True, check=False).stdout.strip().splitlines()[0]

SEED = 42
N_FOLDS = 5
INNER_STOP_FRACTION = 0.10
MAX_ROUNDS = 20000
FEATURE_SET = 'auto'       # 'auto'：07 融合最好的特征变体（07 已完成时），否则 F0；也可手动写 'F0'、'P'、'PT'、'ALL' 等
ZOO = ['lgb_rel', 'cb_rel', 'realmlp', 'tabm']   # 每折内的运行顺序：快的先跑
RUN_ONLY = None            # 例如 ['realmlp']；None 表示全部
STAGE2_MIN_GAIN = 0.02     # kg：运动员平滑在交叉拟合下至少改善这么多才用于提交文件
R1_DIR = PROJECT_DIR / 'artifacts' / 'feature_round1_v1'   # 07 的结果
V2_DIR = PROJECT_DIR / 'outputs_v2'                        # v2 的逐折预测
if FEATURE_SET == 'auto':
    summary07 = R1_DIR / 'summary_blend.csv'
    FEATURE_SET = str(pd.read_csv(summary07)['variant'].iloc[0]) if summary07.is_file() else 'F0'
EXPERIMENT_ID = f'model_zoo_v1_{{FEATURE_SET}}'
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_SHA256 = {SHA!r}
VERSIONS = {{'catboost': catboost.__version__, 'lightgbm': lgb.__version__, 'torch': torch.__version__,
            'pytabkit': None if PYTABKIT_ERROR else importlib.metadata.version('pytabkit'),
            'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}}
print('PROJECT_DIR :', PROJECT_DIR)
print('特征变体     :', FEATURE_SET, '（07 结果存在）' if (R1_DIR / 'summary_blend.csv').is_file() else '（07 尚未完成）')
print('OUTPUT_DIR  :', OUTPUT_DIR)
print('GPU         :', GPU_NAME)
print('版本         :', VERSIONS)
if PYTABKIT_ERROR:
    print('pytabkit 不可用，RealMLP / TabM 会在预检中跳过：', PYTABKIT_ERROR)
'''),
    md('## 2. 读取并核对数据（与 07 相同）'),
    code(DATA_CELL),
    md('## 3. 特征块（与 07 相同）'),
    code(FEATURE_CELL),
    md('''
## 4. 模型与预检

- `lgb_rel` / `cb_rel`：目标 = TotalKg ÷ DOTS 系数（相对力量），样本权重 = 系数²，预测后乘回 kg；参数与 v2 / 07 相同。
- `realmlp` / `tabm`：pytabkit 的默认参数（TD = tuned defaults），目标先在外层训练部分上标准化；数值缺失用外层训练部分的中位数填补并加缺失标记（pytabkit 不接受 NaN）。
- 预检：每个模型在 3,000 行上训练几轮。TabM 的分段线性嵌入若报错，自动改用不带嵌入的 TabM。
'''),
    code('''
require(FEATURE_SET in VARIANTS, f'未知特征变体：{FEATURE_SET}；可选 {list(VARIANTS)}')
X_cat, Xt_cat = frames_for(FEATURE_SET, 'lgb')        # 类别为 category：LightGBM、神经网络
X_str, Xt_str = frames_for(FEATURE_SET, 'catboost')   # 类别为字符串：CatBoost
NUM_COLS = [c for c in X_cat.columns if c not in CAT]
NAN_COLS = [c for c in NUM_COLS if X_cat[c].isna().any() or Xt_cat[c].isna().any()]
print(f'特征变体 {FEATURE_SET}：{X_cat.shape[1]} 列 = {len(NUM_COLS)} 数值（{len(NAN_COLS)} 列有缺失）+ {len(CAT)} 类别')

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=20, verbose=-1)                   # = v2 / 07
CB_PARAMS = dict(learning_rate=0.05, depth=8, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0,
                 border_count=128)                                                    # = v2 / 07 的 C1
NN_COMMON = dict(device='cuda', random_state=SEED, n_cv=1, n_refit=0, verbosity=0)
NN_PARAMS = {'realmlp': dict(NN_COMMON), 'tabm': dict(NN_COMMON, num_emb_type='pwl')}
NN_FALLBACK = {'tabm': dict(NN_COMMON)}         # 预检失败时去掉分段线性数值嵌入


def dots_scale(frame):
    """DOTS 系数的分母 / 500：TotalKg = DOTS 分 × scale。体重截断到公式的有效范围；Mx 取男女平均。"""
    sex = frame['Sex'].astype(str).to_numpy()
    bw = frame['BodyweightKg'].to_numpy(dtype='float64')
    m, f = np.clip(bw, 40, 210), np.clip(bw, 40, 150)
    poly_m = -1.0930e-6 * m**4 + 7.391293e-4 * m**3 - 0.1918759221 * m**2 + 24.0900756 * m - 307.75076
    poly_f = -1.0706e-6 * f**4 + 5.158568e-4 * f**3 - 0.1126655495 * f**2 + 13.6175032 * f - 57.96288
    return np.where(sex == 'M', poly_m, np.where(sex == 'F', poly_f, 0.5 * (poly_m + poly_f))) / 500.0


SCALE, SCALE_TEST = dots_scale(train), dots_scale(test)
require(np.isfinite(SCALE).all() and np.isfinite(SCALE_TEST).all() and min(SCALE.min(), SCALE_TEST.min()) > 0.3,
        'DOTS 系数异常')
relative = y / SCALE
print(f'变异系数：kg 目标 {y.std() / y.mean():.3f} → 相对力量 {relative.std() / relative.mean():.3f}')


def nn_frames(*frames):
    """神经网络输入：数值缺失用第一个表（外层训练部分）的中位数填补并加缺失标记；类别保持 category。"""
    median = frames[0][NUM_COLS].median()
    out = []
    for frame in frames:
        z = frame.copy()
        for c in NAN_COLS:
            z[f'{c}_isna'] = z[c].isna().astype('float64')
        z[NUM_COLS] = z[NUM_COLS].fillna(median).fillna(0.0)
        out.append(z.reset_index(drop=True))
    return out


def fit_model(name, a, s, v, test_rows=None, rounds=MAX_ROUNDS, nn_overrides=None):
    """a / s / v：内层拟合 / 内层早停 / 外层验证 的训练行号。返回 (验证预测, 测试预测, 最佳轮数或 None)。"""
    Xt_c = Xt_cat if test_rows is None else Xt_cat.iloc[test_rows]
    Xt_s = Xt_str if test_rows is None else Xt_str.iloc[test_rows]
    scale_t = SCALE_TEST if test_rows is None else SCALE_TEST[test_rows]
    if name in ('lgb_rel', 'cb_rel'):
        wa, ws = SCALE[a] ** 2, SCALE[s] ** 2        # 相对力量误差 × scale = kg 误差，所以按 scale² 加权
        norm = wa.mean()
        wa, ws = wa / norm, ws / norm
        ra, rs = y[a] / SCALE[a], y[s] / SCALE[s]
        if name == 'lgb_rel':
            m = lgb.LGBMRegressor(n_estimators=rounds, random_state=SEED, n_jobs=-1, **LGB_PARAMS)
            m.fit(X_cat.iloc[a], ra, sample_weight=wa, eval_set=[(X_cat.iloc[s], rs)], eval_sample_weight=[ws],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            best = int(m.best_iteration_ or rounds)
            p_val, p_test = m.predict(X_cat.iloc[v], num_iteration=best), m.predict(Xt_c, num_iteration=best)
        else:
            cats = [c for c in X_str.columns if c in CAT]
            m = catboost.CatBoostRegressor(iterations=rounds, loss_function='RMSE', eval_metric='RMSE',
                                           od_type='Iter', od_wait=300, random_seed=SEED, verbose=0,
                                           task_type='GPU', devices='0', allow_writing_files=False, **CB_PARAMS)
            m.fit(catboost.Pool(X_str.iloc[a], ra, cat_features=cats, weight=wa),
                  eval_set=catboost.Pool(X_str.iloc[s], rs, cat_features=cats, weight=ws), use_best_model=True)
            best = int(m.get_best_iteration()) + 1
            p_val, p_test = m.predict(X_str.iloc[v]), m.predict(Xt_s)
        return np.asarray(p_val, dtype='float64') * SCALE[v], np.asarray(p_test, dtype='float64') * scale_t, best
    cls = {'realmlp': RealMLP_TD_Regressor, 'tabm': TabM_D_Regressor}[name]
    require(cls is not None, f'pytabkit 不可用：{PYTABKIT_ERROR}')
    params = dict(NN_PARAMS[name], **(nn_overrides or {}))
    unknown = set(params) - set(inspect.signature(cls.__init__).parameters)
    require(not unknown, f'{name} 不支持参数 {sorted(unknown)}')
    Za, Zs, Zv, Zt = nn_frames(X_cat.iloc[a], X_cat.iloc[s], X_cat.iloc[v], Xt_c)
    mu, sd = y[a].mean(), y[a].std()
    m = cls(**params)
    m.fit(Za, (y[a] - mu) / sd, Zs, (y[s] - mu) / sd, cat_col_names=[c for c in Za.columns if c in CAT])
    p_val = np.asarray(m.predict(Zv), dtype='float64').reshape(-1) * sd + mu
    p_test = np.asarray(m.predict(Zt), dtype='float64').reshape(-1) * sd + mu
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test, None


rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for name in ZOO:
    for attempt in (['main', 'fallback'] if name in NN_FALLBACK else ['main']):
        if attempt == 'fallback':
            NN_PARAMS[name] = dict(NN_FALLBACK[name])
        started = time.perf_counter()
        try:
            p_val, p_test, _ = fit_model(name, pa, ps, pv, test_rows=np.arange(500), rounds=30,
                                         nn_overrides={'n_epochs': 3})
            require(len(p_val) == len(pv) and len(p_test) == 500, '预测长度不符')
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
            failed.pop(name, None)
            print(f'预检通过：{name:8s}（{attempt}）RMSE {rmse(y[pv], p_val):.1f}，用时 {time.perf_counter() - started:.0f}s')
            break
        except Exception as exc:
            failed[name] = repr(exc)
            print(f'预检失败：{name}（{attempt}）→ {exc!r}')
print('将运行：', [n for n in ZOO if n not in failed], '；跳过：', failed or '无')
'''),
    md('## 5. 五折训练（可续跑）\n\n按折运行，每次训练后立即把验证 / 测试预测和指标写入 `OUTPUT_DIR`。'),
    code('''
CONFIG = {'purpose': 'model zoo on the 07 folds; inner 10% early stop (same split as 07); score on outer fold',
          'feature_set': FEATURE_SET, 'seed': SEED, 'n_folds': N_FOLDS,
          'inner_stop_fraction': INNER_STOP_FRACTION, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def spec_of(name):
    params = LGB_PARAMS if name == 'lgb_rel' else CB_PARAMS if name == 'cb_rel' else NN_PARAMS[name]
    return {'model': name, 'variant': FEATURE_SET, 'columns': list(X_cat.columns), 'params': dict(params),
            'target': 'dots_relative_weighted' if name.endswith('_rel') else 'standardized_kg'}


def run_one(name, fold):
    json_path, npz_path = OUTPUT_DIR / f'{name}__fold{fold}.json', OUTPUT_DIR / f'{name}__fold{fold}.npz'
    spec = spec_of(name)
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{name} fold {fold} 的定义已改变；请改 EXPERIMENT_ID')
        return saved, 'cache'
    fit_idx = np.flatnonzero(FOLD_IDS != fold)
    valid_idx = np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 07
    started = time.perf_counter()
    p_val, p_test, best = fit_model(name, inner_fit, inner_stop, valid_idx)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{name} fold {fold} 预测含非有限值')
    np.savez(npz_path, valid_idx=valid_idx, val=p_val, test=p_test)
    result = {'model': name, 'fold': fold, 'spec': spec, 'best_iteration': best,
              'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started,
              'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result, f'{result["seconds"]:.0f}s'


selected = [n for n in ZOO if (RUN_ONLY is None or n in RUN_ONLY) and n not in failed]
for fold in range(1, N_FOLDS + 1):
    fold_started = time.perf_counter()
    for name in selected:
        result, how = run_one(name, fold)
        best = result['best_iteration']
        print(f'fold {fold}  {name:8s}  RMSE {result["rmse"]:.3f}' + (f'  最佳轮数 {best}' if best else '') + f'  ({how})')
    print(f'—— fold {fold} 完成，本次运行用时 {(time.perf_counter() - fold_started) / 60:.1f} 分钟')
'''),
    md('''
## 6. 结果：单独表现、加入 07 融合的增益、三种堆叠方式（嵌套比较）

上半部分的融合权重在同一份 OOF 上拟合（略乐观，但对所有模型相同）。下半部分「嵌套」= 每一折的融合方式只用其他四折拟合，是诚实分数。
'''),
    code('''
def load_oof(directory, stem):
    oof, pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = directory / f'{stem}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{stem} 的 OOF 不完整')
    return oof, pred


def fold_rmse(pred):
    return np.array([rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)])


MODELS = {}
for tag in ['lgb', 'xgb', 'catboost_C1', 'catboost_C2', 'catboost_C3']:
    loaded = load_oof(R1_DIR, f'{tag}__{FEATURE_SET}')
    if loaded is not None:
        MODELS[f'07_{tag}'] = loaded
BASE_NAMES = list(MODELS)
for name in ZOO:
    loaded = load_oof(OUTPUT_DIR, name)
    if loaded is not None:
        MODELS[name] = loaded
require(MODELS, '还没有任何完成的模型')
print('07 的模型：', BASE_NAMES or '无（07 未完成或特征变体不同）', '｜本 notebook 的模型：',
      [n for n in ZOO if n in MODELS])


def nnls_oof(names_):
    P_ = np.column_stack([MODELS[n][0] for n in names_])
    w, _ = nnls(P_, y)
    return P_ @ w


base_oof = nnls_oof(BASE_NAMES) if BASE_NAMES else None
rows = []
for name, (oof, _) in MODELS.items():
    row = {'model': name, 'oof_rmse': rmse(y, oof), **{f'fold{k}': v for k, v in enumerate(fold_rmse(oof), 1)}}
    if base_oof is not None and name not in BASE_NAMES:
        blended = nnls_oof(BASE_NAMES + [name])
        row.update({'corr_with_07_blend': float(np.corrcoef(oof, base_oof)[0, 1]),
                    'blend_with_07': rmse(y, blended),
                    'delta_vs_07_blend': rmse(y, blended) - rmse(y, base_oof),
                    'folds_better': int((fold_rmse(blended) < fold_rmse(base_oof)).sum())})
    rows.append(row)
standalone = pd.DataFrame(rows).set_index('model').sort_values('oof_rmse')
if base_oof is not None:
    print(f'07 模型的 NNLS 融合 OOF RMSE：{rmse(y, base_oof):.3f}')
display(standalone.round(3))
standalone.to_csv(OUTPUT_DIR / 'summary_models.csv')

# ---- 三种堆叠方式，五折嵌套 ----
NAMES = list(MODELS)
P = np.column_stack([MODELS[n][0] for n in NAMES])
PT = np.column_stack([MODELS[n][1] for n in NAMES])
segment = (both['Sex'].astype(str) + '|' + both['Equipment'].astype(str)).to_numpy()
SEG, SEG_T = segment[:n_train], segment[n_train:]
META = pd.DataFrame({'Sex': both['Sex'].astype(str).astype('category'),
                     'Equipment': both['Equipment'].astype(str).astype('category'),
                     'BodyweightKg': both['BodyweightKg'].to_numpy(dtype='float64'),
                     'Age': both['Age'].to_numpy(dtype='float64')})
META_TR, META_TE = META.iloc[:n_train].reset_index(drop=True), META.iloc[n_train:].reset_index(drop=True)
META_PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=500, subsample=0.8,
                   subsample_freq=1, reg_lambda=10.0, random_state=SEED, n_jobs=-1, verbose=-1)


def global_nnls(fit, apply_P):
    w, _ = nnls(P[fit], y[fit])
    return apply_P @ w


def segment_nnls(fit, apply_P, apply_seg, min_rows=3000):
    out = global_nnls(fit, apply_P)
    for s in np.unique(apply_seg):
        m_fit = fit & (SEG == s)
        if m_fit.sum() >= min_rows:
            w_s, _ = nnls(P[m_fit], y[m_fit])
            m_apply = apply_seg == s
            out[m_apply] = apply_P[m_apply] @ w_s
    return out


def residual_meta(fit, apply_P, apply_meta):
    """先全局 NNLS，再用小 LightGBM 按性别 / 装备 / 体重 / 年龄 / 模型分歧修正残差。"""
    w, _ = nnls(P[fit], y[fit])
    base_fit, base_apply = P[fit] @ w, apply_P @ w
    X_fit = META_TR[fit].assign(blend=base_fit, spread=P[fit].std(axis=1))
    X_apply = apply_meta.reset_index(drop=True).assign(blend=base_apply, spread=apply_P.std(axis=1))
    model = lgb.LGBMRegressor(**META_PARAMS).fit(X_fit, y[fit] - base_fit)
    return base_apply + model.predict(X_apply)


METHODS = {
    'global_nnls': lambda fit, va: global_nnls(fit, P[va]),
    'segment_nnls': lambda fit, va: segment_nnls(fit, P[va], SEG[va]),
    'residual_meta': lambda fit, va: residual_meta(fit, P[va], META_TR[va]),
}
HONEST = {}
for method, fn in METHODS.items():
    pred = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        pred[va] = fn(fit, va)
    HONEST[method] = pred
stack_table = pd.DataFrame({m: {'nested_oof_rmse': rmse(y, p), **{f'fold{k}': v for k, v in enumerate(fold_rmse(p), 1)}}
                            for m, p in HONEST.items()}).T.sort_values('nested_oof_rmse')
display(stack_table.round(3))
stack_table.to_csv(OUTPUT_DIR / 'summary_stacking.csv')
BEST_METHOD = stack_table.index[0]
everything = np.ones(n_train, dtype=bool)
STACK_TEST = {'global_nnls': lambda: global_nnls(everything, PT),
              'segment_nnls': lambda: segment_nnls(everything, PT, SEG_T),
              'residual_meta': lambda: residual_meta(everything, PT, META_TE)}[BEST_METHOD]()
STACK_OOF = HONEST[BEST_METHOD]
print(f'选用的堆叠方式：{BEST_METHOD}（嵌套 OOF RMSE {rmse(y, STACK_OOF):.3f}），使用 {len(NAMES)} 个模型：{NAMES}')
'''),
    md('''
## 7. 第二层：双胞胎 + 同一运动员残差平滑（同时用于 v2）

1. **双胞胎**（与 07 第 9 节相同）：训练集中 16 列完全相同（exact）或核心 9 列相同（core）的行，把预测往其真实成绩拉 `alpha` 倍。
2. **同一运动员**：出生年 ≈ `floor(MeetYear − Age)`（x.5 年龄时恰好是出生年）。对 5 个键（A0 性别+国家+州+出生年；A1 再加联盟；A2 再加装备；A3 再加级别；M2 = 比赛×性别×装备），计算同键**其他**训练行的残差和 ÷（个数 + 收缩常数），用线性回归组合后加到预测上。国家未知的行不做运动员匹配。
3. 诚实评估：按 A0 键的哈希奇偶分两半（同一运动员的所有行在同一半），一半拟合系数、另一半评分。交叉拟合改善 ≥ `STAGE2_MIN_GAIN` 才在提交文件里使用运动员平滑。

输出：本 notebook 五折堆叠和 v2 各写三个文件：`base`（原预测）、`twins`（只加双胞胎）、`final`（双胞胎 + 运动员平滑）。
'''),
    code('''
CORE = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']
ATHLETE_KEYS = {
    'A0': ['Sex', 'Country', 'State', 'birth_year'],
    'A1': ['Sex', 'Country', 'State', 'Federation', 'birth_year'],
    'A2': ['Sex', 'Country', 'State', 'Federation', 'birth_year', 'Equipment'],
    'A3': ['Sex', 'Country', 'State', 'Federation', 'birth_year', 'Equipment', 'WeightClassKg'],
    'M2': ['Federation', 'MeetYear', 'MeetCountry', 'MeetState', 'Sex', 'Equipment'],
}
SHRINK = {'A0': 5.0, 'A1': 5.0, 'A2': 3.0, 'A3': 2.0, 'M2': 20.0}


def keyed(frame):
    out = frame[RAW_FEATURES].copy()
    out['birth_year'] = np.floor(out['MeetYear'] - out['Age'] + 1e-9)
    return out


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


KT, KQ = keyed(train), keyed(test)
EX_TR, EX_TE = row_keys(KT, RAW_FEATURES), row_keys(KQ, RAW_FEATURES)
CO_TR, CO_TE = row_keys(KT, CORE), row_keys(KQ, CORE)
AK_TR = {k: row_keys(KT, cols) for k, cols in ATHLETE_KEYS.items()}
AK_TE = {k: row_keys(KQ, cols) for k, cols in ATHLETE_KEYS.items()}
VALID_TR = (train['Country'] != 'Unknown').to_numpy(dtype='float64')
VALID_TE = (test['Country'] != 'Unknown').to_numpy(dtype='float64')
HALF = (AK_TR['A0'] % 2).astype(int)


def group_sum(keys_tr, values, keys_query=None):
    """训练行：同键其他训练行（不含自己）的和与个数；查询行：同键全部训练行的和与个数。"""
    table = pd.DataFrame({'k': keys_tr, 'v': values}).groupby('k')['v'].agg(['sum', 'count'])
    if keys_query is None:
        k = pd.Series(keys_tr)
        return k.map(table['sum']).to_numpy() - values, k.map(table['count']).to_numpy() - 1
    k = pd.Series(keys_query)
    return k.map(table['sum']).fillna(0).to_numpy(), k.map(table['count']).fillna(0).to_numpy()


def twin_mates(query=False):
    s_ex, n_ex = group_sum(EX_TR, y, EX_TE if query else None)
    s_co, n_co = group_sum(CO_TR, y, CO_TE if query else None)
    level = np.select([n_ex > 0, n_co > 0], ['exact', 'core'], default='none')
    mate = np.where(level == 'exact', s_ex / np.maximum(n_ex, 1), s_co / np.maximum(n_co, 1))
    return level, mate


TW_LEVEL, TW_MATE = twin_mates()
TW_LEVEL_T, TW_MATE_T = twin_mates(query=True)


def fit_alpha(mask, pred):
    d = TW_MATE[mask] - pred[mask]
    denom = float(np.sum(d * d))
    return float(np.clip(np.sum((y[mask] - pred[mask]) * d) / denom, 0, 1)) if denom > 0 else 0.0


def apply_twins(pred, level, mate, alphas):
    out = pred.copy()
    for lv, a in alphas.items():
        m = level == lv
        out[m] = pred[m] + a * (mate[m] - pred[m])
    return out


def athlete_features(residual, query=False):
    cols = []
    for k in ATHLETE_KEYS:
        s, n = group_sum(AK_TR[k], residual, AK_TE[k] if query else None)
        f = s / (n + SHRINK[k])
        if k != 'M2':
            f = f * (VALID_TE if query else VALID_TR)
        cols.append(f)
    return np.column_stack(cols)


def stage2(oof, test_pred, label):
    alphas = {lv: fit_alpha(TW_LEVEL == lv, oof) for lv in ['exact', 'core']}
    p1 = apply_twins(oof, TW_LEVEL, TW_MATE, alphas)
    p1_cf = oof.copy()
    for h in (0, 1):
        a_h = {lv: fit_alpha((HALF != h) & (TW_LEVEL == lv), oof) for lv in ['exact', 'core']}
        m = HALF == h
        p1_cf[m] = apply_twins(oof[m], TW_LEVEL[m], TW_MATE[m], a_h)
    t1 = apply_twins(test_pred, TW_LEVEL_T, TW_MATE_T, alphas)

    r1 = y - p1
    F, FT = athlete_features(r1), athlete_features(r1, query=True)
    p2_cf = p1_cf.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        beta_h = np.linalg.lstsq(F[fit], r1[fit], rcond=None)[0]
        p2_cf[m] = p1_cf[m] + F[m] @ beta_h
    beta = np.linalg.lstsq(F, r1, rcond=None)[0]
    gain = rmse(y, p1_cf) - rmse(y, p2_cf)
    use_athlete = bool(gain >= STAGE2_MIN_GAIN)
    t2 = t1 + FT @ beta if use_athlete else t1.copy()
    row = {'base': label, 'oof_rmse': rmse(y, oof), 'twins_crossfit': rmse(y, p1_cf),
           'twins_athlete_crossfit': rmse(y, p2_cf), 'athlete_gain': gain, 'athlete_used': use_athlete,
           'alpha_exact': alphas['exact'], 'alpha_core': alphas['core'],
           **{f'beta_{k}': float(b) for k, b in zip(ATHLETE_KEYS, beta)},
           'test_rows_twins': int((TW_LEVEL_T != 'none').sum()),
           'test_mean_abs_athlete_adj': float(np.abs(t2 - t1).mean())}
    return row, {'base': test_pred, 'twins': t1, 'final': t2}


rows, exports = [], {}
row, preds = stage2(STACK_OOF, STACK_TEST, f'08 五折堆叠（{BEST_METHOD}）')
rows.append(row)
exports['zoo5f'] = preds

V2_SEEDS = {'lgb': [42], 'xgb': [42, 7, 2024], 'catboost': [42, 7, 2024], 'mlp': [42, 7, 2024]}
try:
    oofs, tests = [], []
    for model, seeds in V2_SEEDS.items():
        oof, pred = np.zeros(n_train), np.zeros(len(test))
        for seed in seeds:
            for i, (_, valid) in enumerate(KFold(10, shuffle=True, random_state=seed).split(np.arange(n_train))):
                saved = np.load(V2_DIR / f'{model}_s{seed}_f{i}.npz')
                require(len(saved['val']) == len(valid) and len(saved['test']) == len(test), 'v2 文件大小不符')
                oof[valid] += saved['val'] / len(seeds)
                pred += saved['test'] / (10 * len(seeds))
        oofs.append(oof)
        tests.append(pred)
    w2, _ = nnls(np.column_stack(oofs), y)
    row, preds = stage2(np.column_stack(oofs) @ w2, np.column_stack(tests) @ w2, 'v2（10 折 × 种子，NNLS）')
    rows.append(row)
    exports['v2'] = preds
except FileNotFoundError as exc:
    print('没有找到 v2 的逐折文件，跳过 v2：', exc)

stage2_table = pd.DataFrame(rows).set_index('base')
display(stage2_table.round(3).T)
stage2_table.to_csv(OUTPUT_DIR / 'summary_stage2.csv')
for name, preds in exports.items():
    for kind, pred in preds.items():
        path = OUTPUT_DIR / f'submission_{name}_{kind}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('写出', path.name)
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
