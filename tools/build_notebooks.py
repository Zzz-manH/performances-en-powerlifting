"""生成项目的 5 本 notebook。

    python tools/build_notebooks.py <输出目录>

notebook 里的训练代码来自在 Colab 上跑过的实验 07–15，原样保存在 tools/cells/*.py；本脚本只负责组装：
统一的环境设置、说明文字、只读的结果展示，以及放在训练开关后面的原始训练代码。
项目根目录里的 notebook 带有 Colab 运行输出，修改时先输出到其他目录，确认后再复制。
"""
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

TOOLS = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else TOOLS.parent


def cell(name):
    return (TOOLS / 'cells' / f'{name}.py').read_text(encoding='utf-8').strip('\n')


def cells(*names):
    return '\n\n\n'.join(cell(n) for n in names)


def md(text):
    # 先去掉缩进再插入 notebook 对照表：表格行没有缩进，提前插入会让 dedent 失效
    text = textwrap.dedent(text).strip()
    return nbf.v4.new_markdown_cell(text.replace('{NOTEBOOK_MAP}', NOTEBOOK_MAP.strip()))


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip('\n'))


def raw_code(text):
    return nbf.v4.new_code_cell(text.strip('\n'))


def gated(flag, body, setup='', env=True, off=None):
    """训练代码只在开关打开时执行；关闭时本节只读取已保存的结果。"""
    lines = [f'if {flag}:']
    if env:
        lines.append('    ensure_training_env()')
    if setup:
        lines.append(textwrap.indent(textwrap.dedent(setup).strip('\n'), '    '))
    lines.append(textwrap.indent(body.strip('\n'), '    '))
    lines.append('else:')
    message = off or f'训练开关关闭（{flag} = False）：本节只读取 artifacts/ 中已保存的结果。'
    lines.append(f'    print("{message}")')
    return nbf.v4.new_code_cell('\n'.join(lines))


def save(cells_, name):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells_
    nb['metadata'] = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'colab': {'provenance': []},
    }
    nbf.validate(nb)
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / name)
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'{name}[{i}]', 'exec')
    print('wrote', OUT / name)


# ============================================================================= 共用单元格
SETUP = '''
# 环境、路径与通用设置（5 本 notebook 相同）
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import nnls
from sklearn.model_selection import KFold, train_test_split

try:
    from google.colab import drive
except ModuleNotFoundError:
    drive = None
if drive is not None:
    drive.mount('/content/drive')


def find_project_dir():
    candidates = []
    if os.environ.get('PL_PROJECT_DIR'):
        candidates.append(Path(os.environ['PL_PROJECT_DIR']))
    candidates.append(Path('/content/drive/MyDrive/Colab Notebooks/performances-en-powerlifting'))
    here = Path.cwd().resolve()
    candidates += [here, *here.parents]
    for candidate in candidates:
        if (candidate / 'data' / 'train.csv').is_file() and (candidate / 'data' / 'test.csv').is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError('找不到项目目录（应包含 data/train.csv 与 data/test.csv）；请设置环境变量 PL_PROJECT_DIR')


PROJECT_DIR = find_project_dir()
DATA_DIR = PROJECT_DIR / 'data'                 # 原始 CSV（只读）
ARTIFACTS = PROJECT_DIR / 'artifacts'           # 每本 notebook 的结果在与它同名的子目录
SEED = 42                                       # 固定五折：KFold(5, shuffle=True, random_state=42)
N_FOLDS = 5
INNER_STOP_FRACTION = 0.10                      # 外层训练部分内留 10% 早停
MAX_ROUNDS = 20000
MODELS = ['lgb', 'xgb', 'catboost']             # 特征消融用的三个模型（每折内的运行顺序：快的先跑）
RUN_ONLY = None                                 # 只跑部分变体时写成列表；None 表示全部
KEEP_DELTA = -0.05                              # kg：平均 ΔRMSE ≤ −0.05 且 ≥ 4/5 折更好才保留
EXPECTED_SHA256 = {'train.csv': '62319814116506fb95a51f1185dbacdb170ab055642ee4c347a30584093d0856',
                   'test.csv': '563c6fa29f9e535740ef1f09b25954f00fbe242c81c0e5a7343dc18aa962ba46'}
CATBOOST_VERSION = '1.2.10'
GPU_NAME = None
VERSIONS = {'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})

TRAIN = globals().get('TRAIN', {})              # 01 与 05 没有训练开关
if sum(bool(v) for v in TRAIN.values()) > 1:
    raise ValueError('一次只打开一个训练开关：不同实验的代码共用变量名。跑完一个、重启运行时后再打开下一个')
TRAINING = any(TRAIN.values())


def ensure_training_env():
    """只在训练时调用：安装固定版本的 CatBoost 与 pytabkit，导入训练用的库，确认 Colab 分配了 GPU。"""
    global catboost, lgb, xgb, torch, RealMLP_TD_Regressor, TabM_D_Regressor, PYTABKIT_ERROR, GPU_NAME, VERSIONS
    to_install = []
    try:
        installed_catboost = importlib.metadata.version('catboost')
    except importlib.metadata.PackageNotFoundError:
        installed_catboost = None
    if installed_catboost != CATBOOST_VERSION:
        if 'catboost' in sys.modules:
            raise RuntimeError('当前运行时已经导入另一版 CatBoost；请重启运行时后从头运行')
        to_install.append(f'catboost=={CATBOOST_VERSION}')
    try:
        importlib.metadata.version('pytabkit')
    except importlib.metadata.PackageNotFoundError:
        to_install.append('pytabkit')
    if to_install:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *to_install])
        importlib.invalidate_caches()
    import catboost
    import lightgbm as lgb
    import torch
    import xgboost as xgb
    try:
        from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
        PYTABKIT_ERROR = None
    except Exception as exc:                    # 安装或导入失败时其余模型照常运行
        RealMLP_TD_Regressor = TabM_D_Regressor = None
        PYTABKIT_ERROR = repr(exc)
    warnings.filterwarnings('ignore', category=UserWarning)
    if catboost.__version__ != CATBOOST_VERSION:
        raise RuntimeError(f'需要 CatBoost {CATBOOST_VERSION}；实际为 {catboost.__version__}')
    if shutil.which('nvidia-smi') is None or not torch.cuda.is_available():
        raise RuntimeError('训练需要 GPU：请在 Colab「更改运行时类型」中选择 T4 GPU 后从头运行')
    GPU_NAME = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              text=True, capture_output=True, check=False).stdout.strip().splitlines()[0]
    VERSIONS = {'catboost': catboost.__version__, 'xgboost': xgb.__version__, 'lightgbm': lgb.__version__,
                'torch': torch.__version__,
                'pytabkit': None if PYTABKIT_ERROR else importlib.metadata.version('pytabkit'),
                'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}
    print('GPU         :', GPU_NAME)
    print('版本         :', VERSIONS)
    if PYTABKIT_ERROR:
        print('pytabkit 不可用，RealMLP / TabM 会在预检中跳过：', PYTABKIT_ERROR)


print(f'项目：{PROJECT_DIR}')
print(f'Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__}')
if TRAINING:
    print('打开的训练开关：', [k for k, v in TRAIN.items() if v])
'''

TRAIN_NOTE = '''
**训练开关**：默认全部为 `False`，只读取 `artifacts/` 中已保存的结果并生成表格和图，Colab 普通 CPU 运行时即可，几分钟跑完。

把某一项改为 `True` 并选择 **T4 GPU** 运行时，会按原协议重新训练对应的实验。每个模型训练完立即写入 `artifacts/`，断线后从头运行会跳过已完成的部分。**一次只打开一个开关**（不同实验的代码共用变量名），跑完后重启运行时再打开下一个。
'''

DATA = cell('data_load')

RESULT_HELPERS = '''
# 只读的结果工具：读取训练时保存的 summary_*.csv 和逐折预测，不训练
def read_summary(path, index_col=0):
    path = Path(path)
    if not path.is_file():
        print(f'缺少 {path.relative_to(PROJECT_DIR)}：请确认 artifacts/ 已从 Drive 同步，或打开对应的训练开关')
        return None
    return pd.read_csv(path, index_col=index_col)


def show(table, columns=None, digits=3):
    if table is not None:
        shown = table if columns is None else table[[c for c in columns if c in table.columns]]
        display(shown.round(digits))


def plot_fold_deltas(table, baseline, title):
    """配对比较表（列 mean_delta、d_fold1–5）：柱 = 五折平均 ΔRMSE，点 = 单折，虚线 = 保留门槛 −0.05。"""
    if table is None:
        return
    rows = table.drop(index=baseline, errors='ignore').sort_values('mean_delta', ascending=False)
    folds = rows[[f'd_fold{k}' for k in range(1, N_FOLDS + 1)]].to_numpy(dtype='float64')
    colours = ['#4c72b0' if v <= KEEP_DELTA else '#c44e52' if v > 0 else '#999999' for v in rows['mean_delta']]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.6))
    ax.barh(range(len(rows)), rows['mean_delta'], color=colours)
    for i in range(len(rows)):
        ax.scatter(folds[i], np.full(folds.shape[1], i), color='black', s=12, zorder=3)
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(KEEP_DELTA, color='#4c72b0', lw=0.8, ls='--')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows.index.tolist())
    ax.set_xlabel(f'RMSE change vs {baseline} (kg); bar = 5-fold mean, dots = single folds')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def read_folds(directory, stem):
    """一个模型五折的 (OOF, 五折平均测试预测)；缺任何一折时返回 None。"""
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for k in range(1, N_FOLDS + 1):
        path = Path(directory) / f'{stem}__fold{k}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    return (oof, test_pred) if not np.isnan(oof).any() else None
'''

FEATURES = cells('features_round1', 'features_round2', 'features_round3')

NOTEBOOK_MAP = '''
| notebook | 内容 |
|---|---|
| **01_data_exploration** | 读取与核对原始数据、探索分析、记录精度与「双胞胎」、数据质量检查、固定五折 |
| **02_feature_engineering** | 四轮特征（记录精度、跨地参赛、比赛背景、运动员历史与生涯、基于成绩的特征）及五折消融（实验 07、09、10、14） |
| **03_model_selection** | 早期基线、CatBoost 参数、其他模型与相对力量目标、堆叠方式、目标变量工程、模型大比较（实验 08、13） |
| **04_final_model** | 最终集成：8 个模型 × 5 折 × 3 种子 + 全量重训、NNLS 融合、双胞胎 + 同一运动员平滑、提交文件（实验 11、12、15） |
| **05_error_analysis** | 最终残差、同一运动员与双胞胎结构、第二层做法比较、数据清洗实验、未采用的方法、交叉验证与公开榜 |

原始数据在 `data/`；结果保存在 `artifacts/` 下与 notebook 同名的目录。「实验 07–15」是实验的原始编号，代码注释和 RESULTS.md 沿用这些编号。
'''

# ============================================================================= 01
OVERVIEW = '''
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
require(sample_submission.columns.tolist() == ['id', TARGET] and len(sample_submission) == len(test)
        and set(sample_submission['id']) == set(test['id']), '提交模板与测试集不一致')
both_raw = pd.concat([train[RAW_FEATURES], test[RAW_FEATURES]], ignore_index=True)
overview = pd.DataFrame({
    '类型': ['数值' if c in RAW_NUM else '类别' for c in RAW_FEATURES],
    '取值个数（train+test）': [int(both_raw[c].nunique()) for c in RAW_FEATURES],
    '最常见取值': [str(both_raw[c].value_counts().index[0]) for c in RAW_FEATURES],
    '最常见取值占比': [float(both_raw[c].value_counts(normalize=True).iloc[0]) for c in RAW_FEATURES],
    'Unknown 占比': [float((both_raw[c].astype(str) == 'Unknown').mean()) for c in RAW_FEATURES],
}, index=RAW_FEATURES)
display(overview.round(3))
display(train[RAW_NUM + [TARGET]].describe().T.round(2))
print('提交模板：', sample_submission.shape, '，列', sample_submission.columns.tolist())
'''

TARGET_PLOTS = '''
sex = train['Sex'].to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].hist(y, bins=80, color='#4c72b0')
axes[0].set_title('TotalKg, all training rows')
axes[0].set_xlabel('kg')
for value, colour in [('M', '#4c72b0'), ('F', '#dd8452')]:
    axes[1].hist(y[sex == value], bins=70, alpha=0.55, density=True, color=colour, label=value)
axes[1].legend()
axes[1].set_title('TotalKg by sex')
axes[1].set_xlabel('kg')
shown_rows = np.random.default_rng(SEED).choice(n_train, 20000, replace=False)
axes[2].scatter(train['BodyweightKg'].to_numpy()[shown_rows], y[shown_rows], s=2, alpha=0.25,
                c=np.where(sex[shown_rows] == 'M', '#4c72b0', '#dd8452'))
axes[2].set_xlabel('BodyweightKg')
axes[2].set_ylabel('TotalKg')
axes[2].set_title('TotalKg vs bodyweight (20,000 random rows; blue = M)')
plt.tight_layout()
plt.show()
display(train.groupby(['Sex', 'Equipment'])[TARGET].agg(['size', 'mean', 'std']).round(1))
'''

GROUP_PLOTS = '''
def mean_by(values, order=None):
    table = pd.DataFrame({'k': np.asarray(values).astype(str), 'y': y}).groupby('k')['y'].agg(['size', 'mean'])
    return table if order is None else table.reindex([o for o in order if o in table.index])


age_cut = pd.cut(train['Age'], [0, 18, 24, 35, 40, 50, 60, np.inf], right=False)
year_cut = pd.cut(train['MeetYear'], [-np.inf, 2000, 2010, 2015, 2020, np.inf], right=False)
largest_feds = train['Federation'].value_counts().head(12).index.tolist()
panels = [('mean TotalKg by age', mean_by(age_cut.astype(str), [str(c) for c in age_cut.cat.categories])),
          ('mean TotalKg by meet year', mean_by(year_cut.astype(str), [str(c) for c in year_cut.cat.categories])),
          ('mean TotalKg, 12 largest federations', mean_by(train['Federation'], largest_feds))]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (title, table) in zip(axes, panels):
    ax.bar(range(len(table)), table['mean'], color='#4c72b0')
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table.index.tolist(), rotation=60, ha='right', fontsize=8)
    ax.axhline(y.mean(), color='black', ls='--', lw=1)
    ax.set_title(title)
plt.tight_layout()
plt.show()
'''

PRECISION = '''
def safe_mean(values):
    return float(np.mean(values)) if len(values) else np.nan


age_half = ((train['Age'] * 2).round() % 2).astype(int).to_numpy()      # 1 = 年龄是 x.5（只登记了出生年份）
bw = train['BodyweightKg']
is_int = np.isclose(bw, bw.round())
is_1dec = np.isclose(bw * 10, (bw * 10).round())
is_half = is_1dec & ~is_int & np.isclose((bw * 10).round() % 10, 5)
bw_precision = np.select([is_int, is_half, is_1dec], ['整数', 'x.5', '其他一位小数'], default='两位小数')
precision_rows = []
for fed in ['（全部）'] + train['Federation'].value_counts().head(8).index.tolist():
    m = np.ones(n_train, dtype=bool) if fed == '（全部）' else (train['Federation'] == fed).to_numpy()
    precision_rows.append({'联盟': fed, '行数': int(m.sum()), 'x.5 年龄占比': age_half[m].mean(),
                           '平均总成绩：整数年龄': safe_mean(y[m & (age_half == 0)]),
                           '平均总成绩：x.5 年龄': safe_mean(y[m & (age_half == 1)]),
                           '两位小数体重占比': float((bw_precision[m] == '两位小数').mean())})
display(pd.DataFrame(precision_rows).set_index('联盟').round(3))
display(pd.DataFrame({'行数': pd.Series(bw_precision).value_counts(),
                      '平均总成绩': pd.Series(y).groupby(bw_precision).mean()}).round(1))
'''

TWINS = '''
def twin_summary():
    core = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']
    exact_tr = pd.util.hash_pandas_object(train[RAW_FEATURES], index=False).to_numpy()
    core_tr = pd.util.hash_pandas_object(train[core], index=False).to_numpy()
    exact_te = pd.util.hash_pandas_object(test[RAW_FEATURES], index=False).to_numpy()
    core_te = pd.util.hash_pandas_object(test[core], index=False).to_numpy()
    n_exact = pd.Series(exact_tr).map(pd.Series(exact_tr).value_counts()).to_numpy() - 1    # 不含自己
    n_core = pd.Series(core_tr).map(pd.Series(core_tr).value_counts()).to_numpy() - 1
    level = np.select([n_exact > 0, n_core > 0], ['exact', 'core'], default='none')
    same_total = pd.DataFrame({'k': exact_tr, 'y': y}).duplicated(keep=False).to_numpy()  # 同键中有成绩完全相同的行
    test_exact = np.isin(exact_te, exact_tr)
    test_core = ~test_exact & np.isin(core_te, core_tr)
    display(pd.DataFrame([
        ('训练行：训练集中有 16 列完全相同的其他行（exact）', int((level == 'exact').sum())),
        ('训练行：只有核心 9 列相同（core）', int((level == 'core').sum())),
        ('exact 训练行中，TotalKg 与某个双胞胎完全相同的比例', f"{same_total[level == 'exact'].mean():.1%}"),
        ('测试行：训练集中有 exact 双胞胎', int(test_exact.sum())),
        ('测试行：训练集中只有 core 双胞胎', int(test_core.sum())),
    ], columns=['项目', '数量']).set_index('项目'))
    example_keys = pd.Series(exact_tr[level == 'exact']).drop_duplicates().head(3).to_numpy()
    example = train[np.isin(exact_tr, example_keys)].assign(key=exact_tr[np.isin(exact_tr, example_keys)])
    display(example.sort_values(['key', TARGET]).drop(columns='key').head(8))


twin_summary()
'''

QUALITY = '''
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) 对抗验证：区分训练行与测试行
encoded = both_raw.copy()
for c in CAT:
    encoded[c] = encoded[c].astype('category').cat.codes
is_test = np.r_[np.zeros(n_train), np.ones(len(test))]
adversarial_auc = cross_val_score(HistGradientBoostingClassifier(max_iter=200, random_state=SEED), encoded, is_test,
                                  cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring='roc_auc').mean()

# 2) id 结构：按 id 排序后相邻两行是否更常来自同一场比赛（id 是否保留了原始数据的顺序）
ids_all = pd.concat([train['id'], test['id']], ignore_index=True).to_numpy()
order = np.argsort(ids_all)
perm = np.random.default_rng(0).permutation(len(ids_all))
meet_key = (both_raw['Federation'] + '|' + both_raw['MeetYear'].astype(str) + '|' + both_raw['MeetCountry']
            + '|' + both_raw['MeetState']).to_numpy()
id_rows = []
for name, values in {'同一场比赛': meet_key, '同一联盟': both_raw['Federation'].to_numpy(),
                     '同一比赛年份': both_raw['MeetYear'].to_numpy(), '同一国家': both_raw['Country'].to_numpy()}.items():
    by_id = float((values[order][1:] == values[order][:-1]).mean())
    shuffled = float((values[perm][1:] == values[perm][:-1]).mean())
    id_rows.append({'相邻两行': name, '按 id 排序': by_id, '随机顺序': shuffled, '倍数': by_id / max(shuffled, 1e-12)})
display(pd.DataFrame(id_rows).set_index('相邻两行').round(4))
display(pd.DataFrame([
    ('训练 / 测试对抗验证 AUC（0.5 = 分布一致）', f'{adversarial_auc:.4f}'),
    ('id 与 TotalKg 的 Spearman 相关', f"{float(spearmanr(train['id'], y)[0]):+.4f}"),
    ('id 与比赛年份的 Spearman 相关（train+test）', f"{float(spearmanr(ids_all, both_raw['MeetYear'])[0]):+.4f}"),
    ('id 范围：训练 / 测试', f"{train['id'].min()}–{train['id'].max()} / {test['id'].min()}–{test['id'].max()}"),
], columns=['检查', '结果']).set_index('检查'))

# 3) 疑似错误或不完整的成绩（只用训练标签判断；是否删除见 05 的数据清洗实验）
ratio = y / train['BodyweightKg'].to_numpy(dtype='float64')
suspicious = (np.where(train['Sex'].to_numpy() == 'M', ratio < 2.5, ratio < 1.8)
              | (train['Age'].to_numpy(dtype='float64') < 5))
print(f'疑似行（男子总成绩 < 2.5 × 体重、女子 < 1.8 × 体重，或年龄 < 5 岁）：{int(suspicious.sum())} 行（{suspicious.mean():.2%}）')
display(train.loc[suspicious, ['Age', 'BodyweightKg', 'Sex', 'Equipment', 'Federation', 'MeetYear', TARGET]].head(8))
'''

FOLDS_TABLE = '''
display(pd.DataFrame({'验证行数': np.bincount(FOLD_IDS)[1:],
                      'TotalKg 均值': [y[FOLD_IDS == k].mean() for k in range(1, N_FOLDS + 1)],
                      'TotalKg 标准差': [y[FOLD_IDS == k].std() for k in range(1, N_FOLDS + 1)]},
                     index=pd.Index(range(1, N_FOLDS + 1), name='fold')).round(2))
'''

nb01 = [
    md('''
    # 01 · 数据读取与分析

    **任务**：根据比赛记录预测力量举选手的三项总成绩 `TotalKg`（kg），评估指标 RMSE（越低越好）。训练集 105,000 行、测试集 45,000 行；16 个特征，其中 3 个数值（年龄、体重、比赛年份）、13 个类别（性别、装备、年龄组、级别、是否药检、国籍、州、联盟、比赛地等）。

    本 notebook 读取 `data/` 中的原始 CSV，核对数据，做探索分析和数据质量检查，并说明之后所有实验共用的**固定五折**。它只读数据、不训练，也不写结果文件；Colab 普通 CPU 运行时即可。

    {NOTEBOOK_MAP}
    '''),
    code(SETUP),
    md('''
    ## 1. 读取并核对原始数据

    核对 SHA-256（与所有实验时的数据逐字节一致）、形状、列分组和空值，并生成固定五折 `FOLD_IDS`。之后每本 notebook 都用同一个单元格读取数据。
    '''),
    raw_code(DATA),
    md('## 2. 字段概览\n\n原始数据没有空值，缺失一律写成 `Unknown`（主要在州、比赛州、上级联盟）。'),
    code(OVERVIEW),
    md('''
    ## 3. 目标变量 TotalKg

    男女分布差别很大，体重与总成绩强相关但离散度很大：同样体重的选手，总成绩可以相差几百公斤。RMSE 约 76 kg 的误差，大部分来自特征里看不出的个人训练水平。
    '''),
    code(TARGET_PLOTS),
    md('## 4. 年龄、比赛年份与联盟\n\n虚线是全部训练行的平均总成绩。'),
    code(GROUP_PLOTS),
    md('''
    ## 5. 记录精度：数据里的隐藏信号

    - **年龄 x.5**：只登记了出生年份时，年龄写成 x.5。不同联盟、不同级别的赛事登记习惯不同。
    - **体重的小数位数**：整数、x.5、一位小数、两位小数，对应不同的称重和录入方式。

    树模型按阈值切分数值，学不到「是不是 .5」「有几位小数」，所以 02 把它们做成显式特征（`P` 块）。在 v2 模型的残差里，USAPL 内整数年龄与 x.5 年龄的残差均值相差 24.5 kg，这是第一轮特征最强的证据。
    '''),
    code(PRECISION),
    md('''
    **从上表看**：
    - USAPL 有 79% 的年龄是 x.5：整数年龄的平均总成绩 498 kg，x.5 年龄 455 kg；FPR 中两者是 554 与 491 kg。
    - USPA 完全不用 x.5，THSPA 的体重 89% 记到两位小数。

    同一个数值（例如 25 岁）在不同记录习惯下含义不同，这些差异部分来自不同的选手群体。模型需要显式的精度特征才能利用它们。
    '''),
    md('''
    ## 6. 训练集里的「双胞胎」

    约 1% 的训练行在训练集中有 16 个特征完全相同的另一行（exact），其中约四分之三的 TotalKg 也完全相同：同一次成绩被登记在多个组别里。测试集里也有这样的行。树模型不会记住单独的一行，所以 04 在模型之后把这些行的预测拉向双胞胎的真实成绩（双胞胎后处理），公开榜上稳定改善约 0.2 kg。core 表示只有核心 9 列（年龄、体重、年份、性别、联盟、比赛国、比赛州、国籍、州）相同。
    '''),
    code(TWINS),
    md('''
    ## 7. 数据质量

    - **对抗验证**：训练一个模型区分训练行和测试行，AUC ≈ 0.5 表示两者分布一致，交叉验证的结果可以代表测试集。
    - **id 结构**：如果 id 保留了原始数据的顺序（例如按比赛排列），相邻 id 会更常来自同一场比赛。
    - **疑似错误或不完整的成绩**：总成绩远低于体重的记录多为只完成单项或比赛失败。是否在训练时删除，05 用实验回答。
    '''),
    code(QUALITY),
    md('''
    **结论**：
    - **训练集与测试集分布一致**：对抗验证 AUC 0.502，02 中每个特征块的训练 / 测试均值也几乎相同，所以交叉验证可以代表测试集。
    - **id 的顺序是随机的**：
      - 按 id 排序后，相邻两行同一场比赛、同一联盟、同一年份、同一国家的比例都是随机顺序的 0.97–1.02 倍；
      - id 与 TotalKg 的相关为 −0.003；训练和测试的 id 交错分布。
      - 因此不作为特征。
    - **疑似行保留**：共 265 行（0.25%），05 的实验表明删掉它们反而变差。
    '''),
    md('''
    ## 8. 固定五折

    `KFold(5, shuffle=True, random_state=42)`，折号 1–5。所有实验都用这同一套折，比较时看同一折上的配对差。最终集成另用种子 7、2024 的两套五折做多种子平均。

    每个外层折内再按 `train_test_split(..., test_size=0.1, random_state=种子)` 留出 10% 做早停，外层验证只用来评分。早期的 v1、v2 直接用外层验证折早停，交叉验证偏乐观：v2 的公开榜比交叉验证高约 0.5 kg，改用内层早停后只高 0.13–0.25。
    '''),
    code(FOLDS_TABLE),
]

# ============================================================================= 02
R1_CONFIG = '''
# 实验 07 的设置（结果目录：原 artifacts/feature_round1_v1）
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / 'round1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_PARAM_CHECK = True                 # CatBoost C2 / C3（在最佳特征集上）
'''

R2_CONFIG = '''
# 实验 09 的设置（结果目录：原 artifacts/feature_round2_v1）
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / 'round2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
'''

R3_CONFIG = '''
# 实验 10 的设置（结果目录：原 artifacts/feature_round3_v1）
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / 'round3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
'''

FINAL_SETTINGS = '''
SEEDS = [42, 7, 2024]                      # 每个种子：不同的五折划分、内层早停划分和模型随机性
FINAL_MODELS = ['lgb', 'xgb', 'lgb_rel', 'catboost', 'cb_rel', 'tabm']   # 每折内的运行顺序：快的先跑
FULL_REFIT = True                          # 树模型在全部训练行上重训
REFIT_FACTOR = 1.2                         # 全量重训轮数 = CV 平均最佳轮数 × 1.2（数据多约 39%）
'''

R4_CONFIG = '''
# 实验 14 的设置（结果目录：原 artifacts/feature_round4_v1）；基线 R3 = 实验 11 的最终分模型特征
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / 'round4'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
''' + FINAL_SETTINGS + '''
SCREEN_MODELS = ['lgb', 'xgb', 'cb_rel']        # 筛选用的三个模型（cb_rel 是最强的单模型）
TARGET_MODELS = ['cb_allo', 'cb_gl']            # 目标变量工程：两种力量公式
INNER_FEATURE_FOLDS = 5                         # 基于成绩的特征：外层训练部分内的嵌套折数
'''

SHOW_R1 = '''
def show_round1():
    directory = ARTIFACTS / '02_feature_engineering' / 'round1'
    single = read_summary(directory / 'summary_single.csv', index_col=[0, 1])
    if single is not None:
        print('单模型五折 OOF RMSE（kg）：')
        display(single['oof_rmse'].unstack(0).reindex(['F0', 'T', 'P', 'PT', 'M', 'ALL']).round(3))
    blend = read_summary(directory / 'summary_blend.csv')
    print('三模型 NNLS 融合（主要判读依据）：')
    show(blend, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cb'])
    plot_fold_deltas(blend, 'F0', 'Round 1: three-model blend vs F0')
    print('双胞胎后处理（alpha 按核心键分两半交叉拟合）：')
    show(read_summary(directory / 'summary_twins.csv'))


show_round1()
'''

SHOW_R2 = '''
def show_round2():
    directory = ARTIFACTS / '02_feature_engineering' / 'round2'
    singles = {m: read_summary(directory / f'summary_single_{m}.csv') for m in ['lgb', 'xgb', 'catboost']}
    singles = {m: t for m, t in singles.items() if t is not None}
    if singles:
        print('单模型：相对 R1 的平均 ΔRMSE（kg，负数 = 更好；C 只给 CatBoost）：')
        display(pd.DataFrame({m: t['mean_delta'] for m, t in singles.items()})
                .reindex(['M2', 'D', 'H', 'S', 'C']).round(3))
    screen = read_summary(directory / 'summary_blend_screen.csv')
    print('三模型 NNLS 融合：')
    show(screen, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cb'])
    plot_fold_deltas(screen, 'R1', 'Round 2: three-model blend vs R1')
    print('达标的块合成 ALL2：')
    show(read_summary(directory / 'summary_blend_all2.csv'), ['oof_rmse', 'mean_delta', 'folds_better', 'keep'])


show_round2()
'''

SHOW_R3 = '''
def show_round3():
    directory = ARTIFACTS / '02_feature_engineering' / 'round3'
    singles = {m: read_summary(directory / f'summary_single_{m}.csv') for m in ['lgb', 'xgb', 'catboost']}
    singles = {m: t for m, t in singles.items() if t is not None}
    if singles:
        print('单模型：相对 R2 的平均 ΔRMSE（kg；PM 不改变 LightGBM 的特征）：')
        display(pd.DataFrame({m: t['mean_delta'] for m, t in singles.items()}).reindex(['H2', 'E', 'PM']).round(3))
    screen = read_summary(directory / 'summary_blend_screen.csv')
    print('三模型 NNLS 融合：')
    show(screen, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cb'])
    plot_fold_deltas(screen, 'R2', 'Round 3: three-model blend vs R2')
    print('三个块都达标，合成 ALL3：')
    show(read_summary(directory / 'summary_blend_all3.csv'), ['oof_rmse', 'mean_delta', 'folds_better', 'keep'])


show_round3()
'''

SHOW_R4 = '''
def show_round4():
    directory = ARTIFACTS / '02_feature_engineering' / 'round4'
    screen = read_summary(directory / 'summary_blend_screen.csv')
    print('三模型 NNLS 融合（LightGBM、XGBoost、cb_rel）：')
    show(screen, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cbrel'])
    plot_fold_deltas(screen, 'R3', 'Round 4 (label-based features): three-model blend vs R3')
    net = read_summary(directory / 'summary_net_after_stage2.csv')
    if net is not None:
        print('再加双胞胎 + S3 平滑（交叉拟合）之后，以及公开榜：')
        display(net.assign(公开榜=pd.Series({'R3': 76.592, 'AS': 76.802})).round(3))


show_round4()
'''

nb02 = [
    md('''
    # 02 · 特征工程

    **原则**：
    - 前三轮只用逐行变换和 train + test 的**无标签**统计（计数、比例、去重计数），不使用 TotalKg。
    - 第四轮尝试基于成绩的特征，一律用嵌套折外计算，防止泄漏。

    | 轮次 | 特征块 | 结果 |
    |---|---|---|
    | 1（实验 07） | `P` 记录精度、`T` 跨地参赛、`M` 比赛背景 | P、M 有效，合计 −0.67 kg |
    | 2（实验 09） | `M2` 比赛背景加强、`D` 组别背景、`H` 运动员历史、`S` DOTS 系数、`C` 比赛 / 运动员 ID | 只有 H 达标（−0.13） |
    | 3（实验 10） | `H2` 运动员生涯、`E` 比赛经验结构、`PM` 按模型挑特征 | 三块都达标，合计 −0.20 |
    | 4（实验 14） | `AS` 运动员实力、`MS` 比赛实力（基于成绩，嵌套折外） | 都变差，不用 |

    **消融协议**：
    - 固定五折，外层训练部分内留 10% 早停，直接用早停模型给外层验证打分。
    - 每个变体训练 LightGBM、XGBoost、CatBoost 三个模型，再做三模型 NNLS 融合。
    - 只比较同一折上相对基线的 RMSE 差：融合的平均 ΔRMSE ≤ −0.05 kg 且至少 4/5 折更好才保留。

    {NOTEBOOK_MAP}
    '''),
    md(TRAIN_NOTE),
    code('''
    TRAIN = {
        'round1': False,   # 实验 07：第一轮特征 6 个变体 × 3 个模型 × 5 折 + CatBoost 参数 + 双胞胎（GPU，约 1.5–2 小时）
        'round2': False,   # 实验 09：第二轮特征（GPU，约 1.5–2 小时）
        'round3': False,   # 实验 10：第三轮特征 + 第二层平滑比较（GPU，约 1–1.5 小时）
        'round4': False,   # 实验 14：基于成绩的特征 + id 检查 + 目标变量工程（GPU，约 1–1.5 小时）
    }
    '''),
    code(SETUP),
    md('## 0. 读取数据与固定五折'),
    raw_code(DATA),
    md('''
    ## 1. 第一轮特征（实验 07）

    **为什么做**：v2 融合（OOF RMSE 77.302）的残差在下面这些分组上明显不为 0，说明模型没有学到这些信号。

    | 信号 | 残差均值（kg） | 解释 |
    |---|---|---|
    | 年龄整数 / x.5 | +2.6 / −3.6；USAPL 内 **+19.0 / −5.5** | x.5 表示只知道出生年份 |
    | 体重记录精度 | 整数 −2.9，两位小数 +1.3 | 称重与录入精度对应赛事级别 |
    | 跨州参赛 / 同州 | +3.6 / −1.8 | 去外州比赛的多是更高级别的赛事 |
    | 出国参赛 / 本国 | +1.6 / −0.5 | 国际赛事 |
    | 比赛规模最小的 1/8 | −3.6 | 小型地方赛 |

    「比赛」近似为 联盟 × 年份 × 比赛国 × 比赛州。

    | 块 | 新列 |
    |---|---|
    | `F0` 基线 | v2 的 19 列：3 数值 + 13 类别 + 体重级别 3 列（`WeightClass_num`、`Is_SHW`、`BW_minus_Class`） |
    | `P` 记录精度 | `age_half`（1 = x.5）、`bw_precision`（0 整数 / 1 x.5 / 2 其他一位小数 / 3 两位小数）、`bw_lb_int`（体重恰为整磅） |
    | `T` 跨地参赛 | `abroad`、`other_state`（居住地未知时为 NaN） |
    | `M` 比赛背景 | 比赛人数、同性别同装备人数、联盟当年人数；比赛内跨州、出国、x.5 年龄、装备、男性的比例；比赛平均年龄 |
    '''),
    raw_code(cell('features_round1')),
    md('''
    ## 2. 第二轮特征（实验 09）

    基线 `R1` = F0 + P + T + M（33 列）。

    | 块 | 内容 |
    |---|---|
    | `M2` 比赛背景加强 | 比赛内的州数、国家数、级别数；药检、无装备、多层装备、两位小数体重、青年、大师的比例；年龄标准差；同性别平均体重；全国赛规模 |
    | `D` 组别背景 | 组别（比赛 × 性别 × 装备 × 级别）人数，组内体重和年龄排名；完全重复行与核心列重复行的个数 |
    | `H` 运动员历史 | 近似同一运动员（性别 + 国籍 + 州 + 联盟 + 出生年）的出现次数、出现年数、距第一次 / 最后一次的年数、用过几种装备、体重偏差 |
    | `S` DOTS 系数 | 力量举通用的体重-性别系数（TotalKg = DOTS 分 × 系数） |
    | `C` ID（只给 CatBoost） | 比赛 ID、运动员 ID，用 CatBoost 的有序目标统计编码 |

    出生年 = `floor(MeetYear − Age)`：x.5 年龄时恰好是出生年，整数年龄可能差 1。
    '''),
    raw_code(cell('features_round2')),
    md('''
    ## 3. 第三轮特征（实验 10）

    基线 `R2` = 第二轮的 ALL2（R1 + H，41 列）。

    | 块 | 内容 | 为什么 |
    |---|---|---|
    | `H2` 运动员生涯 | 参加过的最大比赛规模、最多来自几个州的比赛、是否出国比赛过、换过几个级别、体重变化范围、第一次出现的年龄、同一年出现次数、药检赛比例、出生年 ±1 的模糊匹配计数 | `H` 有效说明「这个人是谁」重要；去过多高级别的比赛把 M 和 H 结合起来 |
    | `E` 比赛经验结构 | 选手平均出现次数、平均经验年数、老将（经验 ≥ 3 年）比例、新人比例 | 经验丰富的选手多的比赛级别更高 |
    | `PM` 按模型挑特征 | CatBoost 加 `M2` + `C`；XGBoost 加 `M2` + `S` + `D`；LightGBM 不加 | 第二轮中各模型单独有效的块不同，也增加融合多样性 |
    '''),
    raw_code(cell('features_round3')),
    md('''
    ## 4. 第四轮：基于成绩的特征（实验 14）

    前三轮都是**不看成绩**的描述，已接近饱和。效果最大的一步（04 的同一运动员平滑）用的是同一个人在训练集里的真实成绩，但只在模型之后做线性修正。第四轮把这类信息直接做成特征：

    - `AS` 运动员实力：近似同一个人其他记录的相对力量（DOTS 分）的个数、均值、最高值、时间衰减均值、年份差、体重变化，以及双胞胎的总成绩。
    - `MS` 比赛实力：同一场比赛、同一组别其他选手的相对力量。

    为了不泄漏，全部嵌套折外计算：外层每一折的训练部分再分 5 个内层折，每行只用其他内层折的成绩；外层验证行和测试行用整个训练部分。这些特征依赖折，所以只在训练开关打开时计算（见第 5.4 节）。
    '''),
    md('''
    ## 5. 消融实验

    下表中 `mean_delta` 是五折平均的 RMSE 变化（kg，负数 = 更好），`folds_better` 是 5 折中优于基线的折数，`keep` 表示是否达到保留标准。图中柱为五折平均，黑点为单折，虚线为保留门槛 −0.05。
    '''),
    raw_code(RESULT_HELPERS),
    md('''
    ### 5.1 第一轮（实验 07，基线 F0）

    三个模型都用 v2 的参数：LightGBM（CPU）、XGBoost（GPU）、CatBoost（GPU，`C1`）。实验 07 同时在最佳特征集上比较了 CatBoost 的正则化参数（结果在 03），并做了双胞胎后处理。
    '''),
    gated("TRAIN['round1']", cells('r1_models', 'r1_run', 'r1_compare', 'r1_residuals', 'r1_catboost_params',
                                   'r1_twins'), setup=R1_CONFIG),
    code(SHOW_R1),
    md('''
    **结论**：
    - `P` 记录精度 −0.33、`M` 比赛背景 −0.36，都是 5/5 折；`T` 跨地参赛 −0.04，未达标，但它的列被 M、H2 使用。
    - **ALL = P + T + M 合计 −0.67（5/5 折）**，三个模型方向一致。
    - 双胞胎后处理再降约 0.25。
    - 公开榜：F0 + 双胞胎 77.947 → ALL + CatBoost C2 + 双胞胎 77.158，与交叉验证一致。
    '''),
    md('### 5.2 第二轮（实验 09，基线 R1 = F0 + P + T + M）\n\nCatBoost 改用 03 选出的 C2 参数（L2 10、Bernoulli 0.8）。'),
    gated("TRAIN['round2']", cells('r2_models', 'r2_run', 'r2_compare', 'r2_all2_twins', 'r2_diagnostics'),
          setup=R2_CONFIG),
    code(SHOW_R2),
    md('''
    **结论**：
    - 只有 **`H` 运动员历史达标（−0.13，5/5 折）**，ALL2 = R1 + H。
    - `C` 让 CatBoost 单独 −0.12，但融合只 −0.05。
    - `M2` 对 CatBoost、XGBoost 各约 −0.06，对 LightGBM +0.07；`D`、`S` 只对 XGBoost 有效。
    - 这些「对单个模型有效」的块在第三轮的 `PM` 中按模型加入。
    '''),
    md('### 5.3 第三轮（实验 10，基线 R2 = ALL2）\n\n实验 10 还在 ALL3 融合上比较了 5 种第二层平滑的做法，结果在 05。'),
    gated("TRAIN['round3']", cells('r3_models', 'r3_run', 'r3_compare', 'r3_all3_twins', 'r3_stage2_variants',
                                   'r3_diagnostics'), setup=R3_CONFIG),
    code(SHOW_R3),
    md('''
    **结论**：
    - `H2` −0.09（5/5 折）、`E` −0.07（4/5）、`PM` −0.11（5/5），**ALL3 合计 −0.20（5/5 折）**，三模型融合 76.876。
    - 残差诊断显示剩下的分组偏差多在 2 kg 以内，不看成绩的特征到此基本饱和。
    '''),
    md('''
    ### 5.4 第四轮（实验 14，基线 R3 = 实验 11 的最终分模型特征）

    三个模型改为 LightGBM、XGBoost 和最强的 cb_rel（DOTS 相对力量目标的 CatBoost，见 03）。新特征和 04 的第二层平滑有重叠，所以还要看加平滑之后的净收益。

    同一个训练开关还包括 `id` 结构检查（结论见 01）和目标变量工程（结果在 03）。
    '''),
    gated("TRAIN['round4']", cells('final_frames', 'r4_id_check', 'r4_label_features', 'final_model_defs',
                                   'r4_models', 'r4_run', 'r4_compare', 'r4_stage2_net', 'r4_targets'),
          setup=R4_CONFIG),
    code(SHOW_R4),
    md('''
    **结论**：
    - 运动员实力 `AS` +0.14（0/5 折），比赛实力 `MS` +0.52（0/5 折），**都变差**。
    - 加平滑之后，以及在公开榜上（76.802 对 76.592），结论都一样。
    - 可能的原因有两个：训练行的特征只参考 80% 的数据，验证行和测试行参考全部训练部分，两者分布不一致；而且这类特征噪声大，容易过拟合。
    - 同样的信息放在模型之后由 S3 平滑使用，效果更好（04、05）。
    '''),
    md('''
    ## 6. 最终使用的特征

    最终特征 = 第三轮的 ALL3，按模型不同（实验 11 起所有最终模型都用它）：

    | 模型 | 特征块 | 列数 |
    |---|---|---:|
    | LightGBM、lgb_rel、TabM、sklearn HGB | F0 + P + T + M + H + H2 + E | 57 |
    | XGBoost | 同上 + M2 + S + D | 78 |
    | CatBoost、cb_rel、cb_gl | 同上 + M2 + C（比赛 / 运动员 ID） | 73 |

    全部特征都由本 notebook 第 1–3 节的代码生成。03、04 打开训练开关时，会用同样的代码重新生成。
    '''),
]

# ============================================================================= 03
ZOO_CONFIG = '''
# 实验 08 的设置（结果目录：原 artifacts/model_zoo_v1_ALL）
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / 'model_zoo'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_SET = 'ALL'        # 实验 07 融合最好的特征变体（F0 + P + T + M）
ZOO = ['lgb_rel', 'cb_rel', 'realmlp', 'tabm']   # 每折内的运行顺序：快的先跑
RUN_ONLY = None            # 例如 ['realmlp']；None 表示全部
STAGE2_MIN_GAIN = 0.02     # kg：运动员平滑在交叉拟合下至少改善这么多才用于提交文件
R1_DIR = ARTIFACTS / '02_feature_engineering' / 'round1'      # 实验 07 的结果
V2_DIR = ARTIFACTS / '03_model_selection' / 'baseline_v2'     # v2 的逐折预测（原 outputs_v2）
'''

COMPARE_CONFIG = '''
# 实验 13 的设置（结果目录：原 artifacts/model_compare_v1）
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / 'model_compare'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
''' + FINAL_SETTINGS + '''
PREV_DIR = ARTIFACTS / '04_final_model' / 'v1_six_models'     # 实验 11 的结果
BOOST_DIR = ARTIFACTS / '04_final_model' / 'v2_more_seeds'    # 实验 12 的结果
PREV_SEEDS = [42, 7, 2024]
COMPARE = ['ridge', 'hgb', 'et', 'rf', 'tabpfn', 'mlp_plr', 'resnet', 'ebm']   # 运行顺序：快的先跑
OPTIONAL = []                         # 想多试可以加 'ftt'（很慢）、'tabr'（需要 faiss）
TABPFN_CONTEXT_ROWS = 10000           # TabPFN v2 的预训练上限是 10,000 行
ADD_DELTA = -0.02                     # 加入融合的门槛：嵌套 ΔRMSE ≤ −0.02 且 ≥ 4/5 折
TABPFN_TOKEN = os.environ.get('TABPFN_TOKEN')
if not TABPFN_TOKEN and drive is not None:
    try:
        from google.colab import userdata
        TABPFN_TOKEN = userdata.get('TABPFN_TOKEN')
    except Exception:
        TABPFN_TOKEN = None
if TABPFN_TOKEN:
    os.environ['TABPFN_TOKEN'] = TABPFN_TOKEN
print('TabPFN：', '最新版（已提供 TABPFN_TOKEN）' if TABPFN_TOKEN else 'v2（无需登录）')
if not (PREV_DIR / 'summary_weights.csv').is_file():
    raise FileNotFoundError(f'缺少实验 11 的结果：{PREV_DIR}')
'''

BASELINES = '''
baselines = pd.DataFrame([
    ('sklearn GradientBoosting（序号编码类别）', '5 折，折均值', 82.60, '约 80'),
    ('LightGBM / CatBoost 融合（本地 CPU）', '5 折，外层折早停', 77.766, '—'),
    ('v1：XGBoost + CatBoost + MLP，NNLS（Colab GPU）', '5 折，外层折早停', 77.775, 78.0),
    ('v2：LightGBM ×1 + XGBoost ×3 + CatBoost ×3 + MLP ×3 种子，NNLS', '10 折，外层折早停', 77.302, 77.799),
    ('v2 单模型：LightGBM / XGBoost / CatBoost / MLP', '同上', '78.189 / 78.432 / 78.093 / 78.395', '—'),
], columns=['模型', '协议', 'CV RMSE', '公开榜'])
display(baselines.set_index('模型'))
'''

SHOW_CB_PARAMS = '''
print('CatBoost 参数，特征 ALL，五折配对（基线 C1 = v2 参数）：')
show(read_summary(ARTIFACTS / '02_feature_engineering' / 'round1' / 'summary_catboost_params.csv'),
     ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'd_fold1', 'd_fold2', 'd_fold3', 'd_fold4', 'd_fold5'])
'''

SHOW_ZOO = '''
def show_zoo():
    directory = ARTIFACTS / '03_model_selection' / 'model_zoo'
    models = read_summary(directory / 'summary_models.csv')
    print('单独的五折 OOF RMSE，以及加入实验 07 三模型融合后的变化（07_ 开头的是实验 07 的模型）：')
    show(models, ['oof_rmse', 'corr_with_07_blend', 'blend_with_07', 'delta_vs_07_blend', 'folds_better'])
    if models is not None:
        ordered = models.sort_values('oof_rmse', ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(range(len(ordered)), ordered['oof_rmse'],
                color=['#999999' if n.startswith('07_') else '#4c72b0' for n in ordered.index])
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered.index.tolist())
        ax.set_xlim(ordered['oof_rmse'].min() - 0.5, ordered['oof_rmse'].max() + 0.2)
        ax.set_xlabel('5-fold OOF RMSE (kg), features ALL; grey = experiment 07 models')
        plt.tight_layout()
        plt.show()
    print('三种堆叠方式（五折嵌套，诚实分数）：')
    show(read_summary(directory / 'summary_stacking.csv'))


show_zoo()
'''

SHOW_TARGETS = '''
print('目标变量工程：R3 特征，种子 42 五折；增益为加入 LightGBM + XGBoost + cb_rel 融合后的嵌套 ΔRMSE')
show(read_summary(ARTIFACTS / '02_feature_engineering' / 'round4' / 'summary_target_engineering.csv'))
'''

SHOW_COMPARE = '''
def show_compare():
    compare = read_summary(ARTIFACTS / '03_model_selection' / 'model_compare' / 'summary_compare.csv')
    show(compare, ['来源', 'oof_seed42', 'corr_with_blend', 'blend_nested', 'delta_vs_blend', 'folds_better',
                   'weight_in_blend'])
    if compare is not None:
        ordered = compare.sort_values('oof_seed42', ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(range(len(ordered)), ordered['oof_seed42'],
                color=['#4c72b0' if s == 11 or str(s) == '11' else '#dd8452' for s in ordered['来源']])
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered.index.tolist())
        ax.set_xlim(76, ordered['oof_seed42'].max() + 0.5)
        ax.set_xlabel('seed-42 5-fold OOF RMSE (kg); blue = final models, orange = compared in experiment 13')
        plt.tight_layout()
        plt.show()


show_compare()
'''

SHOW_LR = '''
lr_check = read_summary(ARTIFACTS / '04_final_model' / 'v2_more_seeds' / 'summary_lr_check.csv')
show(lr_check)
if lr_check is not None:
    print(f"学习率 0.03 相对 0.05：平均 {lr_check['delta'].mean():+.3f} kg，{int((lr_check['delta'] < 0).sum())}/5 折更好；"
          f"最佳轮数约为 {lr_check['best_iter_0.03'].mean() / lr_check['best_iter_0.05'].mean():.2f} 倍")
'''

nb03 = [
    md('''
    # 03 · 模型选择

    回答三个问题：**用哪些模型**、**预测什么目标**、**怎样融合**。所有比较都用 01 的固定五折和同样的内层 10% 早停，看同一折上的配对差。

    1. 早期基线：sklearn、LightGBM / CatBoost、v1、v2
    2. CatBoost 的正则化参数（实验 07）
    3. 其他模型、相对力量目标与三种堆叠方式（实验 08）
    4. 目标变量工程：DOTS、IPF GL、异速生长（实验 08、14）
    5. 其他模型大比较：Ridge、sklearn HGB、随机森林、ExtraTrees、TabPFN、EBM 等（实验 13）
    6. cb_rel 的学习率（实验 12）

    {NOTEBOOK_MAP}
    '''),
    md(TRAIN_NOTE),
    code('''
    TRAIN = {
        'model_zoo': False,       # 实验 08：RealMLP、TabM、cb_rel、lgb_rel + 堆叠 + 运动员平滑（GPU，约 1–1.5 小时）
        'model_compare': False,   # 实验 13：其他模型大比较 + 加入 HGB 的提交文件（GPU，约 1.5 小时）
    }
    '''),
    md('实验 13 需要额外的包（EBM、TabPFN），要在导入 numpy 之前安装，所以这一格放在环境设置之前。'),
    gated("TRAIN['model_compare']", cell('compare_install'), env=False),
    code(SETUP),
    md('## 0. 读取数据；打开训练开关时重新生成特征（与 02 相同的代码）'),
    raw_code(DATA),
    gated('TRAINING', FEATURES, env=False, off='没有打开训练开关：不需要重新生成特征，下面只读取已保存的结果。'),
    raw_code(RESULT_HELPERS),
    md('''
    ## 1. 早期基线

    项目开始时的几次尝试（代码在 Git 历史第一个提交的 `modeling.ipynb`、`pipeline.ipynb`、`colab_gpu_pipeline*.ipynb` 中）：

    - 四类梯度提升 / 神经网络模型单独都在 78.1–78.4，OOF 相关 0.990–0.995，瓶颈在特征的信息量。
    - 多种子平均对神经网络帮助很大：v2 中 MLP 单种子 80.05 → 3 种子 78.40。
    - v1、v2 用外层验证折早停，交叉验证偏乐观：v2 的 CV→公开榜差约 +0.5；实验 07 起改用内层早停，差缩小到 +0.13 到 +0.25。
    '''),
    code(BASELINES),
    md('''
    ## 2. CatBoost 的正则化参数（实验 07）

    在第一轮最佳特征 ALL 上比较：
    - `C1`：v2 参数，depth 8、L2 3；
    - `C2`：depth 8、L2 10、Bernoulli 抽样 0.8；
    - `C3`：depth 6、L2 10、Bernoulli 0.8。

    训练代码在 02 的第一轮训练开关里。
    '''),
    code(SHOW_CB_PARAMS),
    md('**结论**：**C2 −0.22（5/5 折）**，C3 −0.17。之后所有 CatBoost 都用 C2。'),
    md('''
    ## 3. 其他模型、相对力量目标与堆叠（实验 08）

    | 模型 | 为什么可能有用 |
    |---|---|
    | `realmlp`：RealMLP-TD（pytabkit，NeurIPS 2024） | 调好默认参数的 MLP，比 v2 自写的 MLP 强得多 |
    | `tabm`：TabM（pytabkit，ICLR 2025） | 一次训练 32 个共享权重的 MLP，加分段线性数值嵌入，正好解决神经网络方差大的问题 |
    | `cb_rel`、`lgb_rel`：相对力量目标 | 先用 DOTS 公式把 TotalKg 换成相对力量，模型预测「这个人相对多强」，再乘回 kg |

    相对力量目标的样本按系数² 加权，所以模型仍然在最小化 kg 的 RMSE。乘法结构与普通模型的加法结构不同，错误也不同。

    堆叠方式在同一五折上嵌套比较：全局 NNLS、按性别 × 装备分段 NNLS、残差元模型（小 LightGBM）。同一运动员平滑的结果在 05。

    **没放进来的**：
    - TabPFN-2.5：面向 ≤ 5 万行，建议用 A100 / H100；
    - TabR：需要 faiss；
    - AutoGluon：在 T4 上内存不足；
    - log 目标：对 kg 的 RMSE 有系统偏差。
    '''),
    gated("TRAIN['model_zoo']", cells('zoo_models', 'zoo_run', 'zoo_compare', 'zoo_stage2'), setup=ZOO_CONFIG),
    code(SHOW_ZOO),
    md('''
    **结论**：
    - **TabM 77.636 是当时最好的单模型**，加入实验 07 融合 −0.15（5/5 折）；cb_rel −0.06，RealMLP −0.07（但每折 15 分钟），lgb_rel −0.04。
    - 三种堆叠方式中全局 NNLS 最好。最终集成用全局 NNLS（非负权重，不加截距）。
    '''),
    md('''
    ## 4. 目标变量工程（实验 14）

    cb_rel 用的 DOTS 系数之外，再试两种力量公式作为目标（预测后乘回 kg，按系数² 加权）：
    - **异速生长**：总成绩 ÷ 体重^(2/3)；
    - **IPF GL 系数**：按性别 × 是否装备分别取系数，Raw 与 Wraps 算无装备。

    训练代码在 02 的第四轮训练开关里。
    '''),
    code(SHOW_TARGETS),
    md('**结论**：**IPF GL 目标的 CatBoost（`cb_gl`）单独 77.162**，与 cb_rel（77.160）一样强，两者相关 0.9975，加入三模型融合 −0.076（5/5 折），进入最终集成。异速生长 −0.043。'),
    md('''
    ## 5. 其他模型大比较（实验 13）

    所有模型用同样的最终特征（LightGBM / TabM 那一套，57 列）和种子 42 的五折。比较两件事：
    - 单独的分数，与实验 11 中最终模型的种子 42 直接可比；
    - 加入实验 11 的 6 模型融合后，五折嵌套 NNLS 的变化和权重。融合只看「错误是否不同」，单独弱的模型也可能有用。

    MLP-PLR 与 ResNet 因为缺少依赖包 skorch 没有跑成。
    '''),
    gated("TRAIN['model_compare']", cells('final_frames', 'final_model_defs', 'compare_models', 'compare_run',
                                          'compare_results', 'compare_blend', 'compare_stage2'),
          setup=COMPARE_CONFIG),
    code(SHOW_COMPARE),
    md('''
    **结论**：
    - 梯度提升树（CatBoost、LightGBM、XGBoost、sklearn HGB）和 TabM 明显领先。
    - TabPFN、随机森林、ExtraTrees、EBM、Ridge 单独差 3–7 kg，加入融合后权重为 0。
    - 只有 **sklearn HGB** 有用：单独 77.881，加入融合 −0.034（5/5 折），权重 0.15，进入最终集成。
    - 加入 HGB 的提交文件 v3_plus 在公开榜上得 76.334（04）。
    '''),
    md('''
    ## 6. cb_rel 的学习率（实验 12）

    种子 42 五折，与实验 11 的 cb_rel（学习率 0.05）同一划分配对比较。训练代码在 04 的「更多种子」训练开关里。
    '''),
    code(SHOW_LR),
    md('''
    **结论**：学习率 0.03 平均 −0.07（4/5 折），之后 cb_rel、cb_gl 的全量重训都用 0.03，轮数相应增加约 1.44 倍。

    ## 7. 小结：最终集成的 8 个模型

    | 模型 | 目标 | 说明 |
    |---|---|---|
    | LightGBM | kg | v2 参数 |
    | XGBoost | kg | GPU，v2 参数 |
    | CatBoost | kg | C2 参数 |
    | lgb_rel | DOTS 相对力量 | LightGBM |
    | cb_rel | DOTS 相对力量 | CatBoost C2，最强的单模型之一 |
    | cb_gl | IPF GL 相对力量 | CatBoost C2，与 cb_rel 一样强 |
    | TabM | kg（标准化） | pytabkit，分段线性嵌入 |
    | sklearn HGB | kg | HistGradientBoosting，原生类别 |

    融合用全局 NNLS，权重见 04。
    '''),
]

# ============================================================================= 04
FINAL_SHARED = FINAL_SETTINGS + '\n' + cells('final_frames', 'final_model_defs')

V1_CONFIG = '''
# 实验 11 的设置（结果目录：原 artifacts/final_v1）
OUTPUT_DIR = ARTIFACTS / '04_final_model' / 'v1_six_models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_CLEANING_TEST = True                   # 数据清洗实验（只作报告证据，结果在 05）
'''

V2_CONFIG = '''
# 实验 12 的设置（结果目录：原 artifacts/final_v2）
OUTPUT_DIR = ARTIFACTS / '04_final_model' / 'v2_more_seeds'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR = ARTIFACTS / '04_final_model' / 'v1_six_models'     # 实验 11 的结果
PREV_SEEDS = [42, 7, 2024]
EXTRA_SEEDS = [11, 23, 57, 99, 123]                          # 全量重训新增的种子
EXTRA_MODELS = ['lgb', 'xgb', 'lgb_rel', 'cb_rel']           # 快的先跑
TABM_FULL_SEEDS = [42, 7, 2024]
TABM_HOLDOUT = 0.05                                          # TabM 近全量：留 5% 早停
RUN_LR_CHECK = True
if not (PREV_DIR / 'summary_weights.csv').is_file() or not (PREV_DIR / 'oof_nested_blend.npy').is_file():
    raise FileNotFoundError(f'缺少实验 11 的结果：{PREV_DIR}')
'''

V4_CONFIG = '''
# 实验 15 的设置（结果目录：原 artifacts/final_v3）
OUTPUT_DIR = ARTIFACTS / '04_final_model' / 'v4_eight_models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_V1 = ARTIFACTS / '04_final_model' / 'v1_six_models'          # 实验 11
FINAL_V2 = ARTIFACTS / '04_final_model' / 'v2_more_seeds'          # 实验 12
COMPARE_DIR = ARTIFACTS / '03_model_selection' / 'model_compare'   # 实验 13
ROUND4_DIR = ARTIFACTS / '02_feature_engineering' / 'round4'       # 实验 14
PREV_SEEDS = [42, 7, 2024]
GL_FULL_SEEDS = [42, 7, 2024, 11, 23]
HGB_FULL_SEEDS = [42, 7, 2024]
TABM_EXTRA_SEEDS = [11, 23]
TABM_HOLDOUT = 0.05
for directory in (FINAL_V1, FINAL_V2):
    if not directory.is_dir():
        raise FileNotFoundError(f'缺少之前的结果目录：{directory}')
'''

SHOW_V1 = '''
def show_v1():
    directory = ARTIFACTS / '04_final_model' / 'v1_six_models'
    print('每个模型：三个种子各自的五折 OOF RMSE 与三种子平均：')
    show(read_summary(directory / 'summary_models.csv'))
    weights = read_summary(directory / 'summary_weights.csv')
    if weights is not None:
        print('NNLS 权重：', weights['weight'].round(3).to_dict())
    print('第二层（交叉拟合）：')
    show(read_summary(directory / 'summary_stage2.csv'))
    display(pd.DataFrame([('折平均：每个模型 15 个折模型的测试预测平均', 76.520),
                          ('混合：全量与折平均各半', 76.421),
                          ('全量重训：树模型在全部 105,000 行上重训 × 3 个种子，TabM 用折平均', 76.360)],
                         columns=['测试预测（都加双胞胎 + S3）', '公开榜']).set_index('测试预测（都加双胞胎 + S3）'))


show_v1()
'''

SHOW_V2 = '''
def show_v2():
    v1 = ARTIFACTS / '04_final_model' / 'v1_six_models'
    v2 = ARTIFACTS / '04_final_model' / 'v2_more_seeds'
    counts = {m: len(list(v1.glob(f'{m}__s*__full.npz'))) + len(list(v2.glob(f'{m}__s*__full.npz')))
              for m in ['lgb', 'xgb', 'lgb_rel', 'catboost', 'cb_rel']}
    counts['tabm'] = len(list(v2.glob('tabm__s*__nearfull.npz')))
    print('每个模型用于测试预测的全量（TabM：95% 数据）重训模型个数：', counts)
    display(pd.DataFrame([('v2_seeds：全量重训共 8 个种子，TabM 折平均', 76.359),
                          ('v2_seeds_tabmfull：再加 TabM 95% 重训 × 3 个种子', 76.337)],
                         columns=['提交（权重与第二层沿用实验 11）', '公开榜']).set_index('提交（权重与第二层沿用实验 11）'))


show_v2()
'''

SHOW_V3 = '''
print('实验 13：实验 12 的全部改进 + sklearn HGB（五折平均），融合与第二层（交叉拟合）：')
show(read_summary(ARTIFACTS / '03_model_selection' / 'model_compare' / 'summary_stage2.csv'))
'''

SHOW_V4 = '''
def show_v4():
    """从保存的逐折预测重新计算实验 15 的融合，并与保存的提交文件核对（只读，不训练）。"""
    v1 = ARTIFACTS / '04_final_model' / 'v1_six_models'
    v2 = ARTIFACTS / '04_final_model' / 'v2_more_seeds'
    v4 = ARTIFACTS / '04_final_model' / 'v4_eight_models'
    six = ['lgb', 'xgb', 'lgb_rel', 'catboost', 'cb_rel', 'tabm']
    seeds = [42, 7, 2024]

    def refit_files(model):
        if model in ('lgb', 'xgb', 'lgb_rel', 'catboost', 'cb_rel'):
            return sorted(v1.glob(f'{model}__s*__full.npz')) + sorted(v2.glob(f'{model}__s*__full.npz'))
        if model == 'tabm':
            return sorted(v2.glob('tabm__s*__nearfull.npz')) + sorted(v4.glob('tabm__s*__nearfull.npz'))
        if model == 'hgb':
            return sorted(v4.glob('hgb__s*__nearfull.npz'))
        return sorted(v4.glob(f'{model}__s*__full.npz'))

    per, rows = {}, []
    for model in six + ['hgb', 'cb_gl']:
        loaded = {s: read_folds(v1 if model in six else v4, f'{model}__s{s}') for s in seeds}
        loaded = {s: v for s, v in loaded.items() if v is not None}
        if not loaded:
            print(f'{model}：缺少五折结果，跳过')
            continue
        oof = np.mean([o for o, _ in loaded.values()], axis=0)
        files = refit_files(model)
        full = (np.mean([np.load(p)['test'] for p in files], axis=0) if files
                else np.mean([t for _, t in loaded.values()], axis=0))
        per[model] = {'oof': oof, 'full': full}
        rows.append({'model': model, **{f'种子 {s}': rmse(y, o) for s, (o, _) in loaded.items()},
                     '三种子平均 OOF': rmse(y, oof),
                     '测试预测': f'{len(files)} 个全量 / 95% 重训' if files else '折平均'})
    if not per:
        return
    names = list(per)
    weights, _ = nnls(np.column_stack([per[n]['oof'] for n in names]), y)
    table = pd.DataFrame(rows).set_index('model')
    table['NNLS 权重'] = pd.Series(dict(zip(names, weights)))
    display(table.round(3))

    def nested(columns):
        P_sub = np.column_stack([per[n]['oof'] for n in columns])
        out = np.zeros(n_train)
        for k in range(1, N_FOLDS + 1):
            fit, va = FOLD_IDS != k, FOLD_IDS == k
            w, _ = nnls(P_sub[fit], y[fit])
            out[va] = P_sub[va] @ w
        return out

    steps = {'实验 11：6 个模型': [m for m in names if m in six],
             '+ HGB（实验 13）': [m for m in names if m in six + ['hgb']],
             '+ HGB + cb_gl（最终，实验 15）': names}
    display(pd.DataFrame([{'融合': k, '模型数': len(v), '嵌套 OOF RMSE': rmse(y, nested(v))}
                          for k, v in steps.items() if v]).set_index('融合').round(3))
    print('第二层（交叉拟合）：')
    show(read_summary(v4 / 'summary_stage2.csv'))

    blend_test = np.clip(np.column_stack([per[n]['full'] for n in names]) @ weights, 0, None)
    saved_path = v4 / 'submission_final_v4_base.csv'
    if saved_path.is_file() and len(names) == 8:
        saved = pd.read_csv(saved_path)
        require(saved['id'].equals(test['id']), f'{saved_path.name} 的 id 顺序与 test.csv 不同')
        print(f'重新计算的融合测试预测与保存的 {saved_path.name} 最大相差 '
              f'{np.abs(blend_test - saved[TARGET].to_numpy()).max():.2e} kg')


show_v4()
'''

FINAL_FILE_CHECK = '''
final_file = ARTIFACTS / '04_final_model' / 'v4_eight_models' / 'submission_final_v4_final.csv'
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
if final_file.is_file():
    final_sub = pd.read_csv(final_file)
    require(final_sub.columns.tolist() == ['id', TARGET], '列名应为 id, TotalKg')
    require(len(final_sub) == len(sample_submission) == 45000 and final_sub['id'].is_unique
            and set(final_sub['id']) == set(sample_submission['id']), 'id 与 sample_submission 不一致')
    require(np.isfinite(final_sub[TARGET]).all() and (final_sub[TARGET] > 0).all(), '预测含无效值')
    print(f'检查通过：{final_file.relative_to(PROJECT_DIR)}，{len(final_sub)} 行；预测均值 {final_sub[TARGET].mean():.1f}，'
          f'范围 {final_sub[TARGET].min():.1f}–{final_sub[TARGET].max():.1f}（训练集均值 {y.mean():.1f}）')
else:
    print(f'缺少 {final_file.relative_to(PROJECT_DIR)}')

leaderboard = pd.DataFrame([
    ('11', '6 个模型 × 3 种子，折平均 + 双胞胎 + S3', 76.044, 76.520),
    ('11', '同上，树模型全量重训（混合版 76.421）', np.nan, 76.360),
    ('12', '全量重训共 8 个种子', np.nan, 76.359),
    ('12', '+ TabM 95% 重训 × 3 种子', np.nan, 76.337),
    ('13', 'v3_plus：+ sklearn HGB', 76.010, 76.334),
    ('15', 'v4：8 个模型（+ cb_gl、HGB 三种子，TabM 重训 5 个种子）', 75.966, 76.298),
], columns=['实验', '提交', 'CV（嵌套融合 + 交叉拟合第二层）', '公开榜'])
display(leaderboard.set_index('实验'))
'''

nb04 = [
    md('''
    # 04 · 最终模型与提交

    **最终方案**（实验 15，提交文件 `submission_final_v4_final.csv`）：五折交叉验证 **75.966**，公开榜 **76.298**，两者都是全部提交中最好的；2026-10-17 时公开榜第 1。

    | 步骤 | 内容 |
    |---|---|
    | 特征 | 02 的 ALL3，按模型不同（57 / 73 / 78 列） |
    | 模型 | LightGBM、XGBoost、CatBoost、lgb_rel、cb_rel、cb_gl、TabM、sklearn HGB（03） |
    | 验证 | 5 折 × 3 个种子（42、7、2024，每个种子一套不同的五折），外层训练部分内留 10% 早停；每个模型的 OOF 取种子平均 |
    | 测试预测 | 树模型在全部 105,000 行上重训：轮数 = CV 平均最佳轮数 × 1.2，cb_rel、cb_gl 用学习率 0.03。TabM 和 HGB 在 95% / 全部数据上重训（内部早停）。多个种子平均 |
    | 融合 | 全局 NNLS；用五折嵌套给出诚实的分数 |
    | 第二层 | 双胞胎（01）+ S3 同一运动员残差平滑（05），按「性别 + 国籍 + 州」分两半交叉拟合 |

    它由三个实验逐步完成，后一个复用前一个的结果：
    - 实验 11：6 个模型；
    - 实验 12：更多全量重训种子；
    - 实验 15：加入 cb_gl 与 HGB 的三种子版本。

    {NOTEBOOK_MAP}
    '''),
    md(TRAIN_NOTE),
    code('''
    TRAIN = {
        'six_models': False,     # 实验 11：6 个模型 × 5 折 × 3 种子 + 全量重训 + 数据清洗实验（GPU，约 2.5 小时）
        'more_seeds': False,     # 实验 12：cb_rel 学习率检查 + 全量重训再加 5 个种子 + TabM 95% 重训（GPU，约 1 小时）
        'eight_models': False,   # 实验 15：cb_gl、HGB 补齐三种子 + 全量重训 → 8 个模型的最终提交（GPU，约 1 小时）
    }
    '''),
    code(SETUP),
    md('## 0. 读取数据；打开训练开关时重新生成特征与模型定义（与 02、实验 11 相同的代码）'),
    raw_code(DATA),
    gated('TRAINING', FEATURES, env=False, off='没有打开训练开关：不需要重新生成特征，下面只读取已保存的结果。'),
    gated('TRAINING', FINAL_SHARED, off='没有打开训练开关：不需要模型定义。'),
    raw_code(RESULT_HELPERS),
    md('''
    ## 1. 6 个模型 × 5 折 × 3 种子 + 全量重训（实验 11）

    - LightGBM、XGBoost 用 v2 参数，CatBoost 用 C2。
    - `cb_rel`、`lgb_rel` 的目标 = TotalKg ÷ DOTS 系数，样本权重 = 系数²，预测后乘回 kg。
    - TabM：pytabkit 默认参数 + 分段线性数值嵌入；数值缺失用外层训练部分的中位数填补并加缺失标记，目标先标准化。

    测试预测有三种：
    - 折平均：每个模型 15 个折模型平均；
    - 全量重训：树模型在全部训练行上重训，每个种子一次；
    - 混合：两者各半。

    同一个训练开关还包括数据清洗实验，结果在 05。
    '''),
    gated("TRAIN['six_models']", cells('v1_run', 'v1_blend', 'stage2_twins_s3', 'v1_cleaning_test'), setup=V1_CONFIG),
    code(SHOW_V1),
    md('''
    **结论**：
    - 三个种子平均让每个模型降约 0.4–0.8 kg；cb_rel 是最强的单模型（三种子 76.784，权重 0.45）。
    - 融合 76.395 → 双胞胎 76.172 → S3 76.044。
    - **全量重训比折平均好 0.16**（公开榜 76.360 对 76.520）：训练数据越多越好。之后的提交都用全量重训。
    '''),
    md('''
    ## 2. 更多全量重训种子 + TabM 近全量（实验 12）

    - cb_rel、XGBoost、lgb_rel、LightGBM 的全量重训各加 5 个种子（共 8 个）。cb_rel 的新种子用学习率 0.03，学习率检查见 03。
    - TabM 没有固定轮数可以照搬，改为在 95% 的数据上训练（留 5% 早停）× 3 个种子。
    - 融合权重和第二层系数沿用实验 11（全量模型无法交叉验证）。
    '''),
    gated("TRAIN['more_seeds']", cells('v2_setup', 'v2_lr_check', 'v2_refit', 'v2_assemble', 'stage2_twins_s3'),
          setup=V2_CONFIG),
    code(SHOW_V2),
    md('''
    ## 3. 加入 sklearn HGB（实验 13 的融合部分）

    03 的模型大比较中只有 HGB 达到加入门槛。在实验 12 的全部改进之上加入 HGB（五折平均），得到 `v3_plus`，公开榜 **76.334**。训练代码在 03 的模型比较开关里。
    '''),
    code(SHOW_V3),
    md('''
    ## 4. 最终版：8 个模型（实验 15）

    复用实验 11–14 的全部结果，只补缺的部分：

    | 步骤 | 内容 |
    |---|---|
    | 1 | `cb_gl` 补种子 7、2024 的五折（种子 42 复用实验 14） |
    | 2 | sklearn HGB 补种子 7、2024 的五折（种子 42 复用实验 13） |
    | 3 | `cb_gl` 全量重训 × 5 个种子（学习率 0.03） |
    | 4 | HGB 在全部数据上重训（内部留 10% 早停）× 3 个种子 |
    | 5 | TabM 95% 重训再加 2 个种子（共 5 个） |
    | 6 | 8 个模型 NNLS 融合 → 双胞胎 + S3 → 导出 → 自动检查 |

    下面的结果表不是读取保存的数字，而是从逐折预测文件**重新计算**的；最后与保存的提交文件逐行核对。
    '''),
    gated("TRAIN['eight_models']", cells('v4_setup', 'v4_cv', 'v4_refit', 'v4_assemble', 'stage2_twins_s3',
                                         'v4_final_check'), setup=V4_CONFIG),
    code(SHOW_V4),
    md('''
    **结论**：
    - cb_gl（三种子 76.781）与 cb_rel（76.784）一样强，权重最大（0.29）。
    - HGB 单独较弱，但错误与其他模型不同，权重 0.20。
    - LightGBM、lgb_rel、CatBoost 的权重变为 0，它们的信息已被 cb_rel / cb_gl / HGB 覆盖。
    - 嵌套融合 76.395 → 76.351（+ HGB）→ **76.319**（+ cb_gl）；加第二层后 **75.966**，是全部实验中最好的交叉验证分数。
    '''),
    md('''
    ## 5. 提交文件与公开榜

    公开榜只用约 30% 的测试数据（约 13,500 行），0.05 kg 以内的差异基本是噪声（05），所以选模以交叉验证为准。全量重训的版本没有对应的交叉验证分数。
    '''),
    code(FINAL_FILE_CHECK),
    md('''
    **Kaggle 最终选择两个提交**（私榜按另外 70% 的测试数据计算）：

    1. `artifacts/04_final_model/v4_eight_models/submission_final_v4_final.csv`：交叉验证与公开榜都最好（75.966 / 76.298），主选；
    2. `artifacts/03_model_selection/model_compare/submission_final_v3_plus_final.csv`：公开榜第二好（76.334），作保险。

    v4 在公开榜上比 v3_plus 好 0.036，与交叉验证上的 0.045 一致。05 的模拟显示，这样的差异在公开榜上反过来的概率只有约 4%。两个文件的副本都在 `submissions/`。
    '''),
]

# ============================================================================= 05
RESIDUALS = '''
def athlete_key(frame, with_federation=True, extra=()):
    birth = np.floor(frame['MeetYear'] - frame['Age'] + 1e-9).astype(int).astype(str)
    key = frame['Sex'] + '|' + frame['Country'] + '|' + frame['State']
    if with_federation:
        key = key + '|' + frame['Federation']
    key = key + '|' + birth
    for column in extra:
        key = key + '|' + frame[column].astype(str)
    return key.reset_index(drop=True)


def final_residuals():
    v4 = ARTIFACTS / '04_final_model' / 'v4_eight_models'
    before = np.load(v4 / 'oof_nested_blend.npy')        # 8 个模型的嵌套 NNLS 融合
    after = np.load(v4 / 'oof_final_crossfit.npy')       # 再加双胞胎 + S3（交叉拟合）
    require(len(before) == n_train and len(after) == n_train, 'OOF 长度不符')
    r0, r1 = y - before, y - after
    print(f'OOF RMSE：融合 {rmse(y, before):.3f} → 加第二层 {rmse(y, after):.3f}')
    keys_all = pd.concat([athlete_key(train), athlete_key(test)], ignore_index=True)
    appearances = keys_all.map(keys_all.value_counts()).iloc[:n_train].to_numpy()
    appearances = np.where(train['Country'].to_numpy() == 'Unknown', 0, appearances)
    age_half = ((train['Age'] * 2).round() % 2).astype(int).to_numpy()
    largest = train['Federation'].value_counts().head(8).index
    groups = {
        '性别 × 装备': (train['Sex'] + '|' + train['Equipment']).to_numpy(),
        '是否药检赛': train['Tested'].to_numpy(),
        '年龄段': pd.cut(train['Age'], [0, 18, 24, 35, 40, 50, 60, np.inf], right=False).astype(str).to_numpy(),
        '体重段': pd.cut(train['BodyweightKg'], [0, 60, 75, 90, 105, 120, 140, np.inf], right=False).astype(str).to_numpy(),
        '比赛年份': pd.cut(train['MeetYear'], [-np.inf, 2000, 2010, 2015, 2020, np.inf], right=False).astype(str).to_numpy(),
        '联盟（最大的 8 个）': np.where(train['Federation'].isin(largest), train['Federation'], '其他'),
        '年龄是否 x.5': np.where(age_half == 1, 'x.5', '整数'),
        '同一运动员出现次数（train+test）': pd.cut(appearances, [-1, 0, 1, 2, 5, np.inf],
                                                labels=['国籍未知', '1', '2', '3-5', '6+']).astype(str),
    }
    parts = []
    for name, key in groups.items():
        frame = pd.DataFrame({'k': np.asarray(key).astype(str), 'r0': r0, 'r1': r1})
        part = frame.groupby('k').agg(行数=('r1', 'size'), 残差均值_融合=('r0', 'mean'), 残差均值_加第二层=('r1', 'mean'),
                                      RMSE_加第二层=('r1', lambda v: float(np.sqrt(np.mean(np.square(v))))))
        part.index = pd.MultiIndex.from_product([[name], part.index], names=['分组', '取值'])
        parts.append(part)
    display(pd.concat(parts).round(2))

    big = np.abs(r1) > 150
    print(f'|残差| > 150 kg 的行占 {big.mean():.1%}，却占总平方误差的 {np.sum(r1[big] ** 2) / np.sum(r1 ** 2):.0%}')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(np.clip(r1, -400, 400), bins=120, color='#4c72b0')
    axes[0].set_title('Final OOF residual (true - predicted), clipped at +-400 kg')
    deciles = pd.qcut(after, 10, labels=False, duplicates='drop')
    binned = pd.DataFrame({'d': deciles, 'r': r1}).groupby('d')['r'].agg(
        ['mean', lambda v: float(np.sqrt(np.mean(np.square(v))))])
    binned.columns = ['mean residual', 'RMSE']
    binned.plot.bar(ax=axes[1], color=['#999999', '#dd8452'])
    axes[1].set_xlabel('decile of the prediction')
    axes[1].set_title('Residual mean and RMSE by prediction decile')
    plt.tight_layout()
    plt.show()


final_residuals()
'''

ATHLETE_CORR = '''
def athlete_residual_correlation():
    """同一键内「其他行」的平均残差与本行残差的相关（留一）：> 0 说明残差里还有可借用的个人 / 比赛效应。"""
    v4 = ARTIFACTS / '04_final_model' / 'v4_eight_models'
    before = y - np.load(v4 / 'oof_nested_blend.npy')
    after = y - np.load(v4 / 'oof_final_crossfit.npy')
    known = (train['Country'] != 'Unknown').to_numpy()
    keys = {
        'A0 性别 + 国籍 + 州 + 出生年': (athlete_key(train, with_federation=False), known),
        'A1 再加联盟': (athlete_key(train), known),
        'A2 再加装备': (athlete_key(train, extra=['Equipment']), known),
        '比赛 × 性别 × 装备': ((train['Federation'] + '|' + train['MeetYear'].astype(str) + '|' + train['MeetCountry']
                             + '|' + train['MeetState'] + '|' + train['Sex'] + '|' + train['Equipment'])
                            .reset_index(drop=True), np.ones(n_train, dtype=bool)),
    }

    def loo_corr(residual, key, valid):
        frame = pd.DataFrame({'k': key.to_numpy(), 'r': residual})
        total = frame.groupby('k')['r'].transform('sum').to_numpy()
        count = frame.groupby('k')['r'].transform('size').to_numpy()
        mask = valid & (count > 1)
        others = (total - residual) / np.maximum(count - 1, 1)
        return float(np.corrcoef(residual[mask], others[mask])[0, 1]), int(mask.sum())

    rows = []
    for name, (key, valid) in keys.items():
        c0, n0 = loo_corr(before, key, valid)
        c1, _ = loo_corr(after, key, valid)
        rows.append({'键': name, '有同键其他行的训练行': n0, '残差相关：融合后': c0, '残差相关：加第二层后': c1})
    display(pd.DataFrame(rows).set_index('键').round(3))


athlete_residual_correlation()
'''

STAGE2_HISTORY = '''
def stage2_history():
    round1 = ARTIFACTS / '02_feature_engineering' / 'round1'
    zoo = ARTIFACTS / '03_model_selection' / 'model_zoo'
    round3 = ARTIFACTS / '02_feature_engineering' / 'round3'
    final = ARTIFACTS / '04_final_model' / 'v4_eight_models'
    print('1) 双胞胎后处理（实验 07，三模型融合）：')
    show(read_summary(round1 / 'summary_twins.csv'))
    print('2) 同一运动员平滑 S0（实验 08）：在实验 08 的堆叠和 v2 上')
    show(read_summary(zoo / 'summary_stage2.csv'),
         ['oof_rmse', 'twins_crossfit', 'twins_athlete_crossfit', 'athlete_gain', 'beta_A0', 'beta_A1', 'beta_A2',
          'beta_A3', 'beta_M2'])
    print('3) 5 种第二层做法（实验 10，ALL3 三模型融合 + 双胞胎之后）：')
    show(read_summary(round3 / 'summary_stage2_variants.csv'))
    print('4) 最终版（实验 15）：')
    show(read_summary(final / 'summary_stage2.csv'))


stage2_history()
'''

CLEANING = '''
def cleaning_test():
    v1 = ARTIFACTS / '04_final_model' / 'v1_six_models'
    ratio = y / train['BodyweightKg'].to_numpy(dtype='float64')
    suspicious = (np.where(train['Sex'].to_numpy() == 'M', ratio < 2.5, ratio < 1.8)
                  | (train['Age'].to_numpy(dtype='float64') < 5))
    print(f'疑似行：{int(suspicious.sum())}（{suspicious.mean():.2%}）')
    rows = []
    for model in ['lgb', 'xgb', 'catboost']:
        base_all, clean_all, base_normal, clean_normal = [], [], [], []
        for k in range(1, N_FOLDS + 1):
            base_path = v1 / f'{model}__s{SEED}__fold{k}.npz'
            clean_path = v1 / f'clean_{model}__s{SEED}__fold{k}.npz'
            if not (base_path.is_file() and clean_path.is_file() and clean_path.with_suffix('.json').is_file()):
                continue                                   # 与实验 11 相同：npz 和 json 都在才算完成
            base, clean = np.load(base_path), np.load(clean_path)
            idx = base['valid_idx']
            require(np.array_equal(idx, clean['valid_idx']), '清洗实验与基线的验证行不一致')
            normal = ~suspicious[idx]
            base_all.append(rmse(y[idx], base['val']))
            clean_all.append(rmse(y[idx], clean['val']))
            base_normal.append(rmse(y[idx][normal], base['val'][normal]))
            clean_normal.append(rmse(y[idx][normal], clean['val'][normal]))
        if base_all:
            delta = np.array(clean_all) - np.array(base_all)
            rows.append({'模型': model, '完成折数': len(base_all), '基线：全部验证行': np.mean(base_all),
                         '去掉疑似行训练：全部验证行': np.mean(clean_all), 'Δ': delta.mean(),
                         '更好折数': int((delta < 0).sum()), '基线：只看正常行': np.mean(base_normal),
                         '去掉疑似行训练：只看正常行': np.mean(clean_normal)})
    if rows:
        display(pd.DataFrame(rows).set_index('模型').round(3))


cleaning_test()
'''

LEADERBOARD = '''
def leaderboard_analysis():
    history = pd.DataFrame([
        ('v1', 77.775, 78.0), ('v2', 77.302, 77.799), ('v2 + twins', 77.060, 77.604),
        ('07 F0 + twins', 77.701, 77.947), ('07 ALL + C2', 77.210, 77.337), ('07 ALL + C2 + twins', 76.966, 77.158),
        ('08 stack', 77.013, 77.207), ('08 + twins', 76.757, 77.018), ('08 + twins + athlete', 76.521, 76.731),
        ('v2 + twins + athlete', 76.774, 77.125), ('10 ALL3 + twins', 76.649, 76.884),
        ('10 + S0', 76.539, 76.775), ('10 + S3', 76.510, 76.716), ('11 fold average', 76.044, 76.520),
        ('13 v3_plus', 76.010, 76.334), ('14 R3', 76.361, 76.592), ('14 AS', 76.508, 76.802),
        ('15 v4 (final)', 75.966, 76.298),
    ], columns=['submission', 'cv', 'public'])
    history['public - cv'] = history['public'] - history['cv']
    display(history.set_index('submission').round(3))
    early = history['submission'].isin(['v1', 'v2', 'v2 + twins', 'v2 + twins + athlete'])   # 外层折早停
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(history.loc[~early, 'cv'], history.loc[~early, 'public'], color='#4c72b0', label='inner early stopping (07+)')
    ax.scatter(history.loc[early, 'cv'], history.loc[early, 'public'], color='#c44e52', label='outer-fold early stopping (v1, v2-based)')
    for _, row in history.iterrows():
        ax.annotate(row['submission'], (row['cv'], row['public']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    lims = [history[['cv', 'public']].min().min() - 0.1, history[['cv', 'public']].max().max() + 0.1]
    ax.plot(lims, lims, color='black', lw=0.8, ls='--', label='public = CV')
    ax.set_xlabel('CV RMSE (kg)')
    ax.set_ylabel('public leaderboard RMSE (kg)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    print(f"实验 07 起（内层早停）公开榜与 CV 的 Spearman 相关："
          f"{history.loc[~early, ['cv', 'public']].corr(method='spearman').iloc[0, 1]:.2f}")


def public_noise():
    """用 OOF 模拟公开榜：随机抽 13,500 行（≈ 测试集的 30%），比较 v4 与 v3_plus。"""
    v3 = np.load(ARTIFACTS / '03_model_selection' / 'model_compare' / 'oof_final_crossfit.npy')
    v4 = np.load(ARTIFACTS / '04_final_model' / 'v4_eight_models' / 'oof_final_crossfit.npy')
    rng = np.random.default_rng(SEED)
    single, diff = [], []
    for _ in range(500):
        idx = rng.choice(n_train, 13500, replace=False)
        a, b = rmse(y[idx], v3[idx]), rmse(y[idx], v4[idx])
        single.append(b)
        diff.append(b - a)
    diff = np.array(diff)
    public_gap = 76.298 - 76.334                       # 实际公开榜：v4 76.298，v3_plus 76.334
    print(f'全部 105,000 行：v4 {rmse(y, v4):.3f}，v3_plus {rmse(y, v3):.3f}，差 {rmse(y, v4) - rmse(y, v3):+.3f}')
    print(f'随机 13,500 行：单个提交的 RMSE 标准差 {np.std(single):.3f}；两者之差的标准差 {np.std(diff):.3f}；'
          f'公开榜上 v4 反而显示更差的概率 {np.mean(diff > 0):.0%}')
    print(f'实际公开榜：v4 比 v3_plus {public_gap:+.3f}')
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(diff, bins=40, color='#4c72b0')
    ax.axvline(rmse(y, v4) - rmse(y, v3), color='black', ls='--', label='difference on all rows')
    ax.axvline(public_gap, color='#dd8452', lw=1.5, label='actual public leaderboard')
    ax.axvline(0, color='#c44e52', lw=1)
    ax.set_xlabel('RMSE(v4) - RMSE(v3_plus) on 13.5k random rows (kg)')
    ax.set_title('How noisy is a 30% public leaderboard?')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


leaderboard_analysis()
public_noise()
'''

nb05 = [
    md('''
    # 05 · 误差分析

    最终模型之后，检查误差还剩在哪里，并记录试过但没有采用的方法：

    1. **最终模型的残差**：哪些分组仍有系统偏差？大误差来自哪里？
    2. **残差的结构 → 第二层**：同一运动员、同一场比赛的残差相关，双胞胎；5 种平滑做法的比较。
    3. **数据清洗实验**：训练时去掉疑似错误的成绩，会不会更好？
    4. **没有成功的方向**。
    5. **交叉验证与公开榜**：两者是否一致，公开榜有多大噪声。

    本 notebook 只读取 02–04 保存的结果，不训练，也不需要 GPU。

    {NOTEBOOK_MAP}
    '''),
    code(SETUP),
    md('## 0. 读取数据'),
    raw_code(DATA),
    raw_code(RESULT_HELPERS),
    md('''
    ## 1. 最终模型的残差

    残差 = 真实 − 预测（kg），用 04 最终版（实验 15）的五折 OOF：「融合」为 8 个模型的嵌套 NNLS，「加第二层」再加双胞胎和 S3 平滑（交叉拟合）。
    '''),
    code(RESIDUALS),
    md('''
    **怎么读**：
    - **大多数分组的残差均值在 ±1.5 kg 以内**，模型在这些方向上已经校准。偏差较大的都是小分组或极端分组：
      - 女子多层装备（380 行）+4.3；
      - 体重 ≥ 140 kg +3.9；
      - 60 岁以上 −2.5；
      - 只出现一次的运动员 −1.3。

      Mx 只有 9 行。
    - **误差大小差别很大**：女子无装备 RMSE 54 kg，男子多层装备 111 kg；体重越大越难预测。
    - **大误差集中在少数极端行**：|残差| > 150 kg 的 5.3% 行占总平方误差的 36%，多为失败或不完整的比赛记录，例如 216 kg 体重的选手总成绩 155 kg。它们无法从特征识别，也不能删掉（第 3 节）。
    '''),
    md('''
    ## 2. 残差的结构：同一运动员、双胞胎 → 第二层

    特征描述不了个人训练水平，但训练集里常有**同一个人**的其他成绩。下表是「同一键内其他行的平均残差」与本行残差的相关（留一）。相关大于 0，说明可以借用这个人其他记录的残差来修正预测。

    - 早期 v2 的残差上，运动员键的相关约 0.095。
    - 最终融合中，H、H2 特征已经吸收了一部分，降到约 0.045。
    - 加第二层之后降到约 0，同一运动员的信息基本用完。
    - 比赛键本来就接近 0（−0.008）。加第二层后为 0.037：第二层里的比赛项在交叉拟合下带来了少量同场比赛的相关，影响很小。
    '''),
    code(ATHLETE_CORR),
    md('''
    第二层就是利用这一点的后处理，全部只用训练标签和 OOF。为了得到诚实的分数，按「性别 + 国籍 + 州」的哈希奇偶分两半交叉拟合：同一个人的所有记录在同一半，一半拟合、另一半评分。

    1. **双胞胎**：训练集中 16 列完全相同（exact）或核心 9 列相同（core）的行，把预测往其真实成绩拉：`pred + alpha × (双胞胎均值 − pred)`，exact 的 alpha 约 0.85，core 约 0.5。
    2. **同一运动员平滑**：近似同一运动员的其他行的残差，收缩后取平均，线性组合后加到预测上。
       - 键：A0 = 性别 + 国籍 + 州 + 出生年，A1 加联盟，A2 加装备，A3 加级别；另加比赛键。
       - S0 只用精确出生年；S1 允许出生年 ±1；S2 按比赛年份差做时间衰减；S3 三者合并，线性组合；S4 三者合并，用小 LightGBM 组合。
    '''),
    code(STAGE2_HISTORY),
    md('''
    **结论**：
    - 双胞胎在交叉验证上稳定降约 0.22–0.26，公开榜上两次分别 −0.18、−0.19。
    - 同一运动员平滑是最有效的一步：实验 08 中 −0.24，公开榜 −0.29 / −0.48。
    - 5 种做法中 **S3（精确 + 出生年 ±1 + 时间衰减，线性组合）最好**，S4 的小 LightGBM 过拟合（+0.21）。最终集成用 S3。
    - 特征里加入运动员历史（H、H2）之后，平滑的增益从 −0.24 降到 −0.13，两者有重叠，但仍然互补。
    '''),
    md('''
    ## 3. 数据清洗实验（实验 11）

    - 疑似行：男子总成绩 < 2.5 × 体重、女子 < 1.8 × 体重（多为只完成单项或失败的比赛），或年龄 < 5 岁，约 0.25%。
    - 做法：种子 42、同样的五折和内层划分，只把疑似行从训练和早停集合中去掉；外层验证仍然用全部行评分，因为测试集里同样有这类行。
    - 与 04 中同模型、同种子的结果逐折配对比较；另外单独看「正常行」（非疑似的验证行）上的分数。
    '''),
    code(CLEANING),
    md('''
    **结论**：去掉疑似行后，三个模型在全部验证行上都**变差**（+0.04 到 +0.08，最多 2/5 折更好）；只看正常行时略好 0.01–0.03 kg。测试集里同样有这类失败记录，按全部行评分时删掉它们得不偿失，所以全部保留，不删除也不截尾。
    '''),
    md('''
    ## 4. 没有成功的方向

    | 方向 | 结果 | 在哪里 |
    |---|---|---|
    | 基于成绩的嵌套特征：运动员实力、比赛实力 | +0.14 / +0.52，公开榜也变差 | 02 第四轮 |
    | 比赛背景加强、组别背景、DOTS 系数（对三模型融合） | +0.00 到 +0.05 | 02 第二轮 |
    | 跨地参赛 `T` 单独 | −0.04，未达标 | 02 第一轮 |
    | 粗粒度目标编码（性别 × 装备 × 级别等） | +0.27，变差 | 早期 pipeline |
    | 频次特征、性别 × 装备交互、近似出生年 | ±0.05 以内 | 早期 pipeline |
    | TabPFN、随机森林、ExtraTrees、EBM、Ridge | 单独差 3–7 kg，融合权重 0 | 03 第 5 节 |
    | 分段 NNLS、残差元模型（堆叠） | 不如全局 NNLS | 03 第 3 节 |
    | 第二层用小 LightGBM（S4） | 过拟合 +0.21 | 本 notebook 第 2 节 |
    | 训练时删除疑似错误的成绩 | +0.04 到 +0.08 | 本 notebook 第 3 节 |
    | id 作为特征 | id 顺序随机，没有信息 | 01 第 7 节 |
    | log 目标 | 预测几何均值，对 kg 的 RMSE 系统偏低，没有采用 | — |
    | 外部数据（OpenPowerlifting 原始库） | 测试行就在其中，等于标签泄漏，禁止使用 | — |
    '''),
    md('''
    ## 5. 交叉验证与公开榜

    下面的图比较所有有交叉验证分数的提交。实验 07 起改用内层早停后，公开榜与交叉验证的顺序基本一致；v1、v2 用外层验证折早停，交叉验证偏乐观。

    第二部分用 OOF 模拟公开榜：随机抽 13,500 行，看最终两个候选 v4 与 v3_plus 的差异有多大可能在公开榜上反过来。
    '''),
    code(LEADERBOARD),
    md('''
    **结论**：
    - **排序可信**：实验 07 起（内层早停），公开榜与交叉验证的排序相关 0.97。公开榜一致地比交叉验证高，大多高 0.2–0.3 kg，实验 11 的折平均版高 0.48；用外层折早停的 v2 系列高 0.35–0.54。
    - **偏移来自公开榜那 30% 行**：在随机 13,500 行上，单个提交的 RMSE 标准差约 0.59 kg，而两个相近提交之差的标准差只有约 0.03 kg。公开榜对所有提交都用同一批行，所以它们共享同一个偏移。比较两个提交时，看差值是可靠的，看绝对值则不可靠。
    - **最终两个候选**：v4 在交叉验证上比 v3_plus 好 0.045，模拟中在公开榜上反过来的概率约 4%；实际公开榜上 v4 好 0.036，与预期一致。
    '''),
    md('''
    ## 6. 结论

    - **最有效的方向**：
      - 发现数据结构：双胞胎、同一运动员的其他成绩、记录精度；
      - 让模型看到「比赛级别」和「这个人是谁」的特征；
      - 多种子与全量重训。
    - **相对力量目标**：DOTS、IPF GL 的 CatBoost 是最强的单模型，与 kg 目标的模型互补。
    - **饱和**：不看成绩的特征到第三轮基本饱和；基于成绩的特征放进模型反而变差，放在模型之后的平滑里效果更好。
    - **选模**：公开榜只有 30% 的测试数据，所以选模以五折交叉验证和同协议的配对比较为准；公开榜只作确认，两者的排序一致。
    - **最终结果**：v4 的五折交叉验证 75.966，公开榜 76.298（2026-10-17 公开榜第 1）。起点 v1 的交叉验证为 77.775，公开榜 78.0。
    '''),
]

save(nb01, '01_data_exploration.ipynb')
save(nb02, '02_feature_engineering.ipynb')
save(nb03, '03_model_selection.ipynb')
save(nb04, '04_final_model.ipynb')
save(nb05, '05_error_analysis.ipynb')
