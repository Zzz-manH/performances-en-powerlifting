"""生成 07_pl_colab_gpu_feature_round1.ipynb（只生成文件并做语法检查，不在本地执行）。

用法：/opt/anaconda3/bin/python3 tools/build_07_feature_round1.py
"""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '07_pl_colab_gpu_feature_round1.ipynb'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


SHA = {name: sha256(ROOT / name) for name in ['train.csv', 'test.csv']}

cells = [
    md('''
# 07 · Powerlifting：第一轮新特征（记录精度、跨地参赛、比赛背景）——Colab GPU，五折

**为什么做**：v2（10 折 × 3 种子，LGB / XGB / CatBoost / MLP 的 NNLS 融合，OOF RMSE **77.302**）的残差（真实 − 预测）在下面这些分组上明显不为 0，说明现有模型没学到这些信号（2026-10-16 本地只读分析）：

|信号|残差均值（kg）|解释|
|---|---|---|
|年龄是整数 vs x.5|+2.6 / −3.6；USAPL 内 **+19.0 / −5.5**，FPR 内 +17.0 / −1.3，IPF 内 +13.4 / −3.4|x.5 表示只知道出生年份；树按阈值切 Age，学不到「是不是 .5」|
|体重记录精度|整数 kg −2.9，两位小数 +1.3；FFForce、WRPF 内两种精度差 10 kg 以上|称重 / 录入精度对应赛事级别与数据来源|
|跨州参赛 State≠MeetState|+3.6（同州 −1.8）|去外州比赛的多是更高级别的赛事|
|出国参赛 Country≠MeetCountry|+1.6（本国 −0.5）|国际赛事|
|比赛规模最小的 1/8（联盟×年份×比赛国×比赛州）|−3.6|小型地方赛|

这和 HMDA 中有效的「LTV 申报精度」是同一类信号。

**特征块**（全部是逐行变换或 train+test 的无标签计数，不使用 TotalKg）：

|块|新列|
|---|---|
|`F0` 基线|v2 的 19 列：3 数值 + 13 类别 + 体重级别 3 列（`WeightClass_num`、`Is_SHW`、`BW_minus_Class`）|
|`P` 记录精度|`age_half`（1 = x.5）、`bw_precision`（0 整数 / 1 x.5 / 2 其他一位小数 / 3 两位小数）、`bw_lb_int`（体重恰为整磅）|
|`T` 跨地参赛|`abroad`、`other_state`（居住地未知时为 NaN）|
|`M` 比赛背景|`meet_n`、`meet_sexeq_n`、`fed_year_n`，比赛内跨州 / 出国 / x.5 年龄 / 装备 / 男性比例，比赛平均年龄|

变体：`F0`、`P`、`T`、`M`、`PT`（P+T）、`ALL`（P+T+M），都在 F0 之上添加。

**协议**：`KFold(5, shuffle=True, random_state=42)`（与 pipeline、v1 及技能分析相同的五折）。每折在外层训练部分内留出 10% 早停，**直接用早停模型给外层验证打分**（不重训；所有变体相同，配对比较公平，外层验证不参与选轮数）。三个模型都用 v2 参数：LightGBM（CPU）、XGBoost（GPU）、CatBoost（GPU，`C1`）。每个变体再做三模型 NNLS 融合。第 8 节在最佳特征集上加试两组更强正则化的 CatBoost 参数（`C2`、`C3`）。第 9 节评估「训练集双胞胎」后处理并导出提交文件。

**判读**：相对 `F0`，平均 ΔRMSE ≤ **−0.05 kg** 且 **≥ 4/5 折**更好才算有效，以三模型融合为主、单模型方向一致为辅。有效的块进入 08（10 折 × 多种子 + MLP + 全量重训）。

**操作**：Colab 选 **T4 GPU**，从上到下运行。第 4 节的预检在一分钟内暴露参数或数据问题；每次训练后立即写入 Drive，断线后从头运行会跳过已完成的部分。预计 1.5–2 小时（CatBoost 占大部分；第 1 折跑完会打印各模型耗时）。
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

EXPERIMENT_ID = 'feature_round1_v1'
OUTPUT_DIR = PROJECT_DIR / 'artifacts' / EXPERIMENT_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
N_FOLDS = 5
INNER_STOP_FRACTION = 0.10
MAX_ROUNDS = 20000
MODELS = ['lgb', 'xgb', 'catboost']    # 每折内的运行顺序：快的先跑
RUN_ONLY = None                        # 例如 ['F0', 'P']；None 表示全部变体
RUN_PARAM_CHECK = True                 # 第 8 节：CatBoost C2 / C3（在最佳特征集上）
KEEP_DELTA = -0.05                     # kg：平均 ΔRMSE ≤ −0.05 且 ≥ 4/5 折更好才保留
EXPECTED_SHA256 = {SHA!r}
VERSIONS = {{'catboost': catboost.__version__, 'xgboost': xgb.__version__, 'lightgbm': lgb.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}}
print('PROJECT_DIR :', PROJECT_DIR)
print('OUTPUT_DIR  :', OUTPUT_DIR)
print('GPU         :', GPU_NAME)
print('版本         :', VERSIONS)
'''),
    md('## 2. 读取并核对数据'),
    code('''
def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(a, b):
    a, b = np.asarray(a, dtype='float64'), np.asarray(b, dtype='float64')
    return float(np.sqrt(np.mean((a - b) ** 2)))


for name, expected in EXPECTED_SHA256.items():
    require(sha256_file(PROJECT_DIR / name) == expected, f'{name} 与本地版本不同；请检查 Drive 上的数据')

TARGET = 'TotalKg'
train = pd.read_csv(PROJECT_DIR / 'train.csv')
test = pd.read_csv(PROJECT_DIR / 'test.csv')
RAW_NUM = ['Age', 'BodyweightKg', 'MeetYear']
RAW_FEATURES = [c for c in test.columns if c != 'id']
CAT = [c for c in RAW_FEATURES if c not in RAW_NUM]          # 不依赖 object / str dtype（pandas 2 与 3 都可用）
require(train.shape == (105000, 18) and test.shape == (45000, 17), '数据形状不符')
require(len(CAT) == 13 and all(pd.api.types.is_numeric_dtype(train[c]) for c in RAW_NUM)
        and not any(pd.api.types.is_numeric_dtype(train[c]) for c in CAT), '列分组不符')
require(not train[RAW_FEATURES].isna().any().any() and not test[RAW_FEATURES].isna().any().any(), '出现空值')
y = train[TARGET].to_numpy(dtype='float64')
n_train = len(train)

FOLD_IDS = np.zeros(n_train, dtype=int)
for k, (_, valid) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n_train)), start=1):
    FOLD_IDS[valid] = k
print(f'train {train.shape}  test {test.shape}；每折验证行数：{np.bincount(FOLD_IDS)[1:].tolist()}')
'''),
    md('## 3. 特征块'),
    code('''
both = pd.concat([train[RAW_FEATURES], test[RAW_FEATURES]], ignore_index=True)

# F0：v2 的体重级别特征
weight_class = pd.to_numeric(both['WeightClassKg'].str.replace('+', '', regex=False), errors='coerce')
F0_EXTRA = pd.DataFrame({
    'WeightClass_num': weight_class,
    'Is_SHW': both['WeightClassKg'].str.endswith('+').astype('float64'),
    'BW_minus_Class': both['BodyweightKg'] - weight_class,
})

# P：记录精度（树按阈值切数值，学不到「是不是 .5」「是不是整数」）
age2 = (both['Age'] * 2).round()
require(np.allclose(both['Age'] * 2, age2), 'Age 不是 0.5 的整数倍')
bw = both['BodyweightKg']
is_int = np.isclose(bw, bw.round())
is_1dec = np.isclose(bw * 10, (bw * 10).round())
is_half = is_1dec & ~is_int & np.isclose((bw * 10).round() % 10, 5)
pounds = bw / 0.45359237
P = pd.DataFrame({
    'age_half': (age2 % 2).astype('float64'),
    'bw_precision': np.select([is_int, is_half, is_1dec], [0.0, 1.0, 2.0], default=3.0),
    'bw_lb_int': (np.abs(pounds - pounds.round()) < 0.015).astype('float64'),
})

# T：跨地参赛（居住地未知时为 NaN）
abroad = (both['Country'] != both['MeetCountry']).astype('float64').where(both['Country'] != 'Unknown')
state_known = ((both['State'] != 'Unknown') & (both['MeetState'] != 'Unknown')
               & (both['Country'] == both['MeetCountry']))
other_state = (both['State'] != both['MeetState']).astype('float64').where(state_known)
T = pd.DataFrame({'abroad': abroad, 'other_state': other_state})

# M：比赛背景；「比赛」≈ 联盟 × 年份 × 比赛国 × 比赛州（train+test 无标签统计）
meet = (both['Federation'] + '|' + both['MeetYear'].astype(str) + '|'
        + both['MeetCountry'] + '|' + both['MeetState'])


def by_meet(values):
    return values.groupby(meet).transform('mean')


M = pd.DataFrame({
    'meet_n': meet.map(meet.value_counts()).astype('float64'),
    'meet_sexeq_n': both.groupby([meet, both['Sex'], both['Equipment']])['Age'].transform('size').astype('float64'),
    'fed_year_n': both.groupby(['Federation', 'MeetYear'])['Age'].transform('size').astype('float64'),
    'meet_share_other_state': by_meet(other_state),
    'meet_share_abroad': by_meet(abroad),
    'meet_share_age_half': by_meet(P['age_half']),
    'meet_share_equipped': by_meet((both['Equipment'] != 'Raw').astype('float64')),
    'meet_share_male': by_meet((both['Sex'] == 'M').astype('float64')),
    'meet_mean_age': by_meet(both['Age']),
})

BLOCKS = {'P': P, 'T': T, 'M': M}
VARIANTS = {'F0': [], 'P': ['P'], 'T': ['T'], 'M': ['M'], 'PT': ['P', 'T'], 'ALL': ['P', 'T', 'M']}
for name, frame in {'F0': F0_EXTRA, **BLOCKS}.items():
    require(len(frame) == len(both) and frame.index.equals(both.index), f'{name} 行数或索引不符')
    require(not np.isinf(frame.to_numpy(dtype='float64')).any(), f'{name} 含无穷大')

_FRAMES = {}


def frames_for(variant, model):
    """(train, test) 特征表。CatBoost 的类别为字符串；LGB / XGB 为 category（词表来自 train+test，无标签）。"""
    kind = 'str' if model == 'catboost' else 'category'
    if (variant, kind) not in _FRAMES:
        frame = pd.concat([both[RAW_NUM], F0_EXTRA] + [BLOCKS[b] for b in VARIANTS[variant]] + [both[CAT]], axis=1)
        require(frame.columns.is_unique, f'{variant} 列名重复')
        frame[CAT] = frame[CAT].astype(str).astype(object) if kind == 'str' else frame[CAT].astype('category')
        _FRAMES[(variant, kind)] = (frame.iloc[:n_train].reset_index(drop=True),
                                    frame.iloc[n_train:].reset_index(drop=True))
    return _FRAMES[(variant, kind)]


print({v: frames_for(v, 'lgb')[0].shape[1] for v in VARIANTS})
summary_new = pd.concat([P, T, M], axis=1)
display(pd.DataFrame({'train_mean': summary_new.iloc[:n_train].mean(), 'test_mean': summary_new.iloc[n_train:].mean(),
                      'nan_share': summary_new.isna().mean()}).round(4))
'''),
    md('## 4. 模型与预检'),
    code('''
LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, cat_smooth=20, verbose=-1)
XGB_PARAMS = dict(learning_rate=0.03, max_depth=8, min_child_weight=10.0, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=1.0, tree_method='hist', device='cuda', enable_categorical=True,
                  max_cat_to_onehot=1, eval_metric='rmse')
CB_PARAMS = {
    'C1': dict(learning_rate=0.05, depth=8, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0,
               border_count=128),                                    # = v2
    'C2': dict(learning_rate=0.05, depth=8, l2_leaf_reg=10.0, bootstrap_type='Bernoulli', subsample=0.8,
               border_count=128),                                    # 更强 L2 + 行抽样
    'C3': dict(learning_rate=0.05, depth=6, l2_leaf_reg=10.0, bootstrap_type='Bernoulli', subsample=0.8,
               border_count=128),                                    # 更浅
}
PATIENCE = {'lgb': 200, 'xgb': 200, 'catboost': 300}


def fit_predict(model, key, Xa, ya, Xs, ys, Xv, Xt, rounds=MAX_ROUNDS):
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
        return (m.predict(Xv, iteration_range=(0, best)), m.predict(Xt, iteration_range=(0, best)), best)
    if model == 'catboost':
        cats = [c for c in Xa.columns if c in CAT]
        m = catboost.CatBoostRegressor(iterations=rounds, loss_function='RMSE', eval_metric='RMSE',
                                       od_type='Iter', od_wait=PATIENCE['catboost'], random_seed=SEED,
                                       verbose=0, task_type='GPU', devices='0', allow_writing_files=False,
                                       **CB_PARAMS[key])
        m.fit(catboost.Pool(Xa, ya, cat_features=cats), eval_set=catboost.Pool(Xs, ys, cat_features=cats),
              use_best_model=True)
        best = int(m.get_best_iteration()) + 1
        return m.predict(Xv), m.predict(Xt), best
    raise ValueError(model)


def tag_of(model, key):
    return f'catboost_{key}' if model == 'catboost' else model


# 预检：每个（模型, 变体）在 3,000 行上训练 30 轮，几秒内暴露问题，避免跑了一小时才报错
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
checks = [(m, v, 'C1') for v in VARIANTS for m in MODELS] + [('catboost', 'F0', 'C2'), ('catboost', 'F0', 'C3')]
failed = {}
for model, variant, key in checks:
    Xtr, Xte = frames_for(variant, model)
    try:
        p_val, p_test, _ = fit_predict(model, key, Xtr.iloc[pa], y[pa], Xtr.iloc[ps], y[ps], Xtr.iloc[pv],
                                       Xte.iloc[:500], rounds=30)
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
        require(len(p_val) == len(pv) and len(p_test) == 500, '预测长度不符')
        print(f'预检通过：{tag_of(model, key):12s} {variant:4s}（{Xtr.shape[1]} 列）RMSE@30轮 {rmse(y[pv], p_val):.1f}')
    except Exception as exc:
        failed[(model, variant, key)] = repr(exc)
        print(f'预检失败，跳过：{tag_of(model, key)} {variant} → {exc!r}')
require(not any(v == 'F0' and k == 'C1' for (_, v, k) in failed), '基线 F0 预检失败，请检查环境')
'''),
    md('## 5. 五折筛选（可续跑）\n\n按折运行：第 1 折的全部变体跑完就能先看到一次配对比较。每次训练的验证 / 测试预测和指标立即写入 `OUTPUT_DIR`。'),
    code('''
CONFIG = {'purpose': 'feature round 1 (P/T/M blocks); inner 10% early stop, score early-stopped model on outer fold',
          'seed': SEED, 'n_folds': N_FOLDS, 'inner_stop_fraction': INNER_STOP_FRACTION,
          'variants': VARIANTS, 'data_sha256': EXPECTED_SHA256}
config_path = OUTPUT_DIR / 'config.json'
if config_path.is_file():
    require(json.loads(config_path.read_text(encoding='utf-8')) == CONFIG,
            f'{OUTPUT_DIR.name} 已有不同配置的结果；改动配置时请改 EXPERIMENT_ID')
else:
    config_path.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=1), encoding='utf-8')


def spec_of(model, variant, key):
    Xtr, _ = frames_for(variant, model)
    params = {'lgb': LGB_PARAMS, 'xgb': XGB_PARAMS, 'catboost': CB_PARAMS[key]}[model]
    return {'model': model, 'variant': variant, 'params_key': key, 'columns': list(Xtr.columns),
            'params': dict(params), 'patience': PATIENCE[model]}


def run_one(model, variant, key, fold):
    stem = f'{tag_of(model, key)}__{variant}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    spec = spec_of(model, variant, key)
    if json_path.is_file() and npz_path.is_file():
        saved = json.loads(json_path.read_text(encoding='utf-8'))
        require(saved['spec'] == spec, f'{stem} 的定义已改变；请改 EXPERIMENT_ID 或变体名')
        return saved, 'cache'
    Xtr, Xte = frames_for(variant, model)
    fit_idx = np.flatnonzero(FOLD_IDS != fold)
    valid_idx = np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)
    started = time.perf_counter()
    p_val, p_test, best = fit_predict(model, key, Xtr.iloc[inner_fit], y[inner_fit], Xtr.iloc[inner_stop],
                                      y[inner_stop], Xtr.iloc[valid_idx], Xte)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    np.savez(npz_path, valid_idx=valid_idx, val=np.asarray(p_val, dtype='float64'),
             test=np.asarray(p_test, dtype='float64'))
    result = {'tag': tag_of(model, key), 'variant': variant, 'fold': fold, 'spec': spec,
              'n_features': int(Xtr.shape[1]), 'best_iteration': best,
              'hit_limit': bool(best >= MAX_ROUNDS - PATIENCE[model]),
              'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started,
              'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result, f'{result["seconds"]:.0f}s'


selected = [v for v in VARIANTS if RUN_ONLY is None or v in RUN_ONLY]
for fold in range(1, N_FOLDS + 1):
    fold_started = time.perf_counter()
    for model in MODELS:
        for variant in selected:
            if (model, variant, 'C1') in failed:
                continue
            result, how = run_one(model, variant, 'C1', fold)
            print(f'fold {fold}  {tag_of(model, "C1"):12s} {variant:4s}  RMSE {result["rmse"]:.3f}  '
                  f'最佳轮数 {result["best_iteration"]:5d}  ({how})')
    print(f'—— fold {fold} 完成，本次运行用时 {(time.perf_counter() - fold_started) / 60:.1f} 分钟')
'''),
    md('## 6. 与 F0 配对比较（单模型 + 三模型融合）'),
    code('''
def load_results():
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUTPUT_DIR.glob('*__fold*.json'))]
    require(rows, '还没有完成的结果')
    return pd.DataFrame(rows)


def load_oof(tag, variant):
    """五折都完成时返回 (OOF, 五折平均测试预测)，否则 (None, None)。"""
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{tag}__{variant}__fold{fold}.npz'
        if not path.is_file():
            return None, None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{tag} {variant} 的 OOF 不完整')
    return oof, test_pred


def fold_rmse(pred):
    return pd.Series({k: rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)})


def paired_table(preds, base='F0'):
    """preds: {变体: OOF}；与 base 在同一五折上配对比较。"""
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


results = load_results()
SINGLE, TESTS = {}, {}
for tag in ['lgb', 'xgb', 'catboost_C1']:
    preds = {}
    for variant in VARIANTS:
        oof, test_pred = load_oof(tag, variant)
        if oof is not None:
            preds[variant], TESTS[(tag, variant)] = oof, test_pred
    if 'F0' in preds:
        SINGLE[tag] = paired_table(preds)
        part = results[results['tag'] == tag].groupby('variant')
        SINGLE[tag]['mean_best_iter'] = part['best_iteration'].mean()
        SINGLE[tag]['minutes'] = part['seconds'].sum() / 60
        print(f'\\n=== {tag}')
        display(SINGLE[tag].round(3))

BLENDS = {}
for variant in VARIANTS:
    if all((tag, variant) in TESTS for tag in ['lgb', 'xgb', 'catboost_C1']):
        P_oof = np.column_stack([load_oof(tag, variant)[0] for tag in ['lgb', 'xgb', 'catboost_C1']])
        P_test = np.column_stack([TESTS[(tag, variant)] for tag in ['lgb', 'xgb', 'catboost_C1']])
        w, _ = nnls(P_oof, y)
        BLENDS[variant] = {'oof': P_oof @ w, 'test': P_test @ w, 'weights': w.round(3).tolist()}
require('F0' in BLENDS, '三模型的 F0 还没有全部完成')
blend_table = paired_table({v: b['oof'] for v, b in BLENDS.items()})
blend_table['weights_lgb_xgb_cb'] = [str(BLENDS[v]['weights']) for v in blend_table.index]
print('\\n=== 三模型 NNLS 融合（主要判读依据）')
display(blend_table.round(3))
BEST = blend_table.index[0]
print(f'融合 OOF 最好的变体：{BEST}；达到保留标准的变体：{blend_table.index[blend_table["keep"]].tolist() or "无"}')

pd.concat({tag: t for tag, t in SINGLE.items()}, names=['model']).to_csv(OUTPUT_DIR / 'summary_single.csv')
blend_table.to_csv(OUTPUT_DIR / 'summary_blend.csv')
'''),
    md('## 7. 残差核对：新特征有没有吸收这些信号\n\n分组残差均值（真实 − 预测，kg）。有效时，`BEST` 一列应比 `F0` 更接近 0。'),
    code('''
def residual_table(pred):
    res = y - pred
    groups = {
        'age_half': P['age_half'].iloc[:n_train].to_numpy(),
        'bw_precision': P['bw_precision'].iloc[:n_train].to_numpy(),
        'other_state(-1=未知)': T['other_state'].iloc[:n_train].fillna(-1).to_numpy(),
        'abroad(-1=未知)': T['abroad'].iloc[:n_train].fillna(-1).to_numpy(),
        'meet_n 八分位': pd.qcut(M['meet_n'].iloc[:n_train], 8, labels=False, duplicates='drop').to_numpy(),
    }
    frames = []
    for name, key in groups.items():
        g = pd.DataFrame({'k': key, 'r': res}).groupby('k')['r'].agg(['size', 'mean'])
        g.index = pd.MultiIndex.from_product([[name], g.index], names=['signal', 'level'])
        frames.append(g)
    return pd.concat(frames)


check = residual_table(BLENDS['F0']['oof']).rename(columns={'mean': 'F0'})
check[BEST] = residual_table(BLENDS[BEST]['oof'])['mean']
display(check.round(2))
'''),
    md('## 8. CatBoost 正则化参数（在最佳特征集上，可续跑）\n\n`C2` = depth 8、L2 10、Bernoulli 0.8；`C3` = depth 6、L2 10、Bernoulli 0.8。与同特征的 `C1`（v2 参数）配对比较。'),
    code('''
if RUN_PARAM_CHECK:
    for fold in range(1, N_FOLDS + 1):
        for key in ['C2', 'C3']:
            if ('catboost', 'F0', key) in failed:
                continue
            result, how = run_one('catboost', BEST, key, fold)
            print(f'fold {fold}  catboost_{key}  {BEST:4s}  RMSE {result["rmse"]:.3f}  '
                  f'最佳轮数 {result["best_iteration"]:5d}  ({how})')
    preds = {}
    for key in ['C1', 'C2', 'C3']:
        oof, test_pred = load_oof(f'catboost_{key}', BEST)
        if oof is not None:
            preds[key], TESTS[(f'catboost_{key}', BEST)] = oof, test_pred
    param_table = paired_table(preds, base='C1')
    display(param_table.round(3))
    param_table.to_csv(OUTPUT_DIR / 'summary_catboost_params.csv')
    BEST_CB = param_table.index[0] if param_table.loc[param_table.index[0], 'keep'] else 'C1'
else:
    BEST_CB = 'C1'
tags = ['lgb', 'xgb', f'catboost_{BEST_CB}']
P_oof = np.column_stack([load_oof(tag, BEST)[0] for tag in tags])
P_test = np.column_stack([TESTS[(tag, BEST)] for tag in tags])
w, _ = nnls(P_oof, y)
FINAL = {'name': f'{BEST}_blend3_{BEST_CB}', 'oof': P_oof @ w, 'test': P_test @ w}
print(f'最终组合：特征 {BEST}，CatBoost {BEST_CB}，权重 {dict(zip(tags, w.round(3)))}，'
      f'OOF RMSE {rmse(y, FINAL["oof"]):.3f}（F0 融合 {rmse(y, BLENDS["F0"]["oof"]):.3f}）')
'''),
    md('''
## 9. 训练集「双胞胎」后处理 + 导出提交文件

约 1% 的行在训练集中有 16 个特征完全相同的「双胞胎」，其中约 73% 的 TotalKg 完全相同（同一次成绩登记在多个组别）。树模型不会记忆单行，所以把这些行的预测往双胞胎的真实成绩拉：`pred + alpha × (双胞胎均值 − pred)`。`alpha` 按匹配层级（`exact` = 16 列全同；`core` = 年龄、体重、年份、性别、联盟、比赛地、居住地相同）在训练 OOF 上估计，按 core 键分两半交叉拟合报告诚实的 RMSE。

导出的文件来自五折单种子、每折只用 90% 外层训练数据的模型，**只用于在公开榜上配对验证特征效果**（同协议 `F0` 对 `BEST`）；最终提交来自 08。
'''),
    code('''
CORE = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def mate_stats(train_keys, query_keys=None):
    table = pd.DataFrame({'k': train_keys, 'y': y}).groupby('k')['y'].agg(['sum', 'count'])
    if query_keys is None:                       # 训练行：留一
        keys = pd.Series(train_keys)
        n = keys.map(table['count']).to_numpy() - 1
        s = keys.map(table['sum']).to_numpy() - y
    else:                                        # 测试行：全部训练行
        keys = pd.Series(query_keys)
        n = keys.map(table['count']).fillna(0).to_numpy()
        s = keys.map(table['sum']).fillna(0).to_numpy()
    return n, np.where(n > 0, s / np.maximum(n, 1), np.nan)


def twins(train_keys_exact, train_keys_core, query_exact=None, query_core=None):
    n_ex, m_ex = mate_stats(train_keys_exact, query_exact)
    n_co, m_co = mate_stats(train_keys_core, query_core)
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


ex_tr, co_tr = row_keys(train, RAW_FEATURES), row_keys(train, CORE)
ex_te, co_te = row_keys(test, RAW_FEATURES), row_keys(test, CORE)
LEVEL, MATE = twins(ex_tr, co_tr)
T_LEVEL, T_MATE = twins(ex_tr, co_tr, ex_te, co_te)
half = (co_tr % 2).astype(int)

twin_rows, exports = [], {'F0_blend3': BLENDS['F0'], FINAL['name']: FINAL}
for name, item in exports.items():
    oof = item['oof']
    alphas = {lv: fit_alpha(LEVEL == lv, oof, MATE) for lv in ['exact', 'core']}
    crossfit = oof.copy()
    for h in (0, 1):
        a = {lv: fit_alpha((half != h) & (LEVEL == lv), oof, MATE) for lv in ['exact', 'core']}
        crossfit[half == h] = apply_twins(oof[half == h], MATE[half == h], LEVEL[half == h], a)
    test_fixed = apply_twins(item['test'], T_MATE, T_LEVEL, alphas)
    twin_rows.append({'name': name, 'oof_rmse': rmse(y, oof), 'oof_rmse_twinfix_crossfit': rmse(y, crossfit),
                      'alpha_exact': alphas['exact'], 'alpha_core': alphas['core'],
                      'test_rows_exact': int((T_LEVEL == 'exact').sum()), 'test_rows_core': int((T_LEVEL == 'core').sum())})
    for suffix, pred in [('', item['test']), ('_twinfix', test_fixed)]:
        path = OUTPUT_DIR / f'submission_{name}{suffix}.csv'
        pd.DataFrame({'id': test['id'], TARGET: np.clip(pred, 0, None)}).to_csv(path, index=False)
        print('写出', path.name)
twin_table = pd.DataFrame(twin_rows).set_index('name')
display(twin_table.round(3))
twin_table.to_csv(OUTPUT_DIR / 'summary_twins.csv')
np.save(OUTPUT_DIR / 'oof_F0_blend3.npy', BLENDS['F0']['oof'])
np.save(OUTPUT_DIR / f'oof_{FINAL["name"]}.npy', FINAL['oof'])
print('完成。请把 OUTPUT_DIR 里的 summary_*.csv 结果告诉 Claude（本地会从 Drive 同步）。')
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

# 语法检查：逐个 code cell 编译（不执行）
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        compile(cell['source'], f'{NAME}[cell {i}]', 'exec')

nbf.write(nb, ROOT / NAME)
print('wrote', ROOT / NAME, '| code cells compiled OK')
