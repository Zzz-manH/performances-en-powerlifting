"""生成 13_pl_colab_gpu_model_compare.ipynb（只生成文件并做语法检查，不在本地执行）。

环境、数据、特征、模型定义、第二层单元原样取自 11，读取 11（以及 12，如已运行）的结果。
用法：/opt/anaconda3/bin/python3 tools/build_13_model_compare.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '13_pl_colab_gpu_model_compare.ipynb'
NB11 = nbf.read(ROOT / '11_pl_colab_gpu_final.ipynb', as_version=4)
ENV_CELL = NB11.cells[2].source.replace("EXPERIMENT_ID = 'final_v1'", "EXPERIMENT_ID = 'model_compare_v1'")
DATA_CELL, FEATURE_CELL, ROUND2_CELL, FINAL_FEATURE_CELL = (NB11.cells[i].source for i in (4, 6, 8, 10))
MODEL_DEFS = NB11.cells[12].source.split('\n# ---- 预检')[0]
STAGE2_CELL = NB11.cells[16].source
assert "EXPERIMENT_ID = 'model_compare_v1'" in ENV_CELL, '11 的环境单元位置变了'
assert 'FOLD_IDS' in DATA_CELL and "BLOCKS = {'P': P" in FEATURE_CELL and "BLOCKS2 = {'M2': M2" in ROUND2_CELL
assert "FRAMES = {m: frames3('ALL3'" in FINAL_FEATURE_CELL, '11 的最终特征单元位置变了'
assert 'def nn_frames' in MODEL_DEFS and 'probe' not in MODEL_DEFS, '11 的模型单元位置变了'
assert 'NESTED' in STAGE2_CELL and 'BLEND_TEST' in STAGE2_CELL


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md('''
# 13 · Powerlifting：其他模型大比较（Colab GPU，五折）

**目的**：回答「别的模型能不能用」。所有模型用**同样的最终特征**（11 中 LightGBM / TabM 用的 R2 + H2 + E，57 列）、**同样的五折**（种子 42）和同样的内层划分，比较两件事：
1. **单独的分数**：和 11 中各模型种子 42 的分数直接可比（单种子对单种子）。
2. **能不能帮到最终融合**：加到 11 的 6 模型融合里，五折嵌套 NNLS 的分数变化和权重。融合只看「错误是否不同」，单独弱的模型也可能有用。

|模型|类型|说明|
|---|---|---|
|`ridge`|线性|数值中位数填补 + 标准化，类别 one-hot（稀有类别合并），Ridge|
|`hgb`|梯度提升树（sklearn）|HistGradientBoosting，原生类别（最多 250 类，其余合并）|
|`et`|随机化树集成|ExtraTrees，200 棵|
|`rf`|随机森林|RandomForest，200 棵，每棵用一半样本|
|`tabpfn`|表格基础模型|TabPFN：默认 v2（无需登录，上下文 10,000 行）；在 Colab「密钥」里设置 `TABPFN_TOKEN` 时用最新版|
|`mlp_plr`|神经网络|MLP + PLR 数值嵌入（pytabkit）|
|`resnet`|神经网络|ResNet（pytabkit）|
|`ebm`|可解释提升（GA2M）|InterpretML 的 EBM，可画出每个特征的形状函数，适合写进报告|
|`ftt`、`tabr`|Transformer / 检索式网络|默认不跑：FT-Transformer 很慢；TabR 需要 faiss|

已经比较过、这里不再重复的：v1/v2 的自写 MLP（单种子 80.05）、RealMLP（78.48，每折 15 分钟）、决策树（88.5）、早期 sklearn GB（82.6）。

**判读**：加入后嵌套 NNLS 平均 ΔRMSE ≤ −0.02 且 ≥ 4/5 折更好，才在最后写一个加入新模型的提交文件。

**操作**：Colab 选 **T4 GPU**，从上到下运行；约 1.5 小时（EBM、两个神经网络、TabPFN 较慢）。某个模型安装或运行失败会自动跳过，不影响其他模型。按模型逐个跑完，中途中断后从第 8 节开始运行可以用已经跑完的模型。需要 11 的结果目录 `artifacts/final_v1/`。
'''),
    md('## 0. 安装额外的包（在导入 numpy 之前）'),
    code('''
import importlib
import importlib.metadata
import subprocess
import sys

EXTRA_PACKAGES = ['interpret-core', 'tabpfn']        # EBM、TabPFN
missing = []
for package in EXTRA_PACKAGES:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        missing.append(package)
if missing:
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], text=True,
                            capture_output=True, check=False)
    print('pip 安装：', missing, '返回码', result.returncode)
    if result.returncode != 0:
        print(result.stderr[-2000:])
    importlib.invalidate_caches()
'''),
    md('## 1. 环境与配置（与 11 相同，实验名不同）'),
    code(ENV_CELL + '''

PREV_DIR = PROJECT_DIR / 'artifacts' / 'final_v1'          # 11 的结果
BOOST_DIR = PROJECT_DIR / 'artifacts' / 'final_v2'         # 12 的结果（如已运行）
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
    raise FileNotFoundError(f'缺少 11 的结果：{PREV_DIR}；请先完整运行 11')
'''),
    md('## 2. 读取并核对数据（与 07 相同）'),
    code(DATA_CELL),
    md('## 3. 第一轮特征（与 07 相同）'),
    code(FEATURE_CELL),
    md('## 4. 第二轮特征（与 09 相同）'),
    code(ROUND2_CELL),
    md('## 5. 第三轮特征与各模型的最终特征表（与 11 相同）'),
    code(FINAL_FEATURE_CELL),
    md('## 6. 新模型的定义与预检'),
    code(MODEL_DEFS + '''

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

X_BASE, XT_BASE, CATS = FRAMES['lgb']            # R2 + H2 + E，类别为 category（与 TabM 相同）
NUMS = [c for c in X_BASE.columns if c not in CATS]
PYTABKIT_CLASSES = {'mlp_plr': 'MLP_PLR_D_Regressor', 'resnet': 'Resnet_RTDL_D_Regressor',
                    'ftt': 'FTT_D_Regressor', 'tabr': 'TabR_S_D_Regressor'}


def numeric_with_codes(enc, X, fill=None):
    num = X[NUMS].to_numpy(dtype='float64')
    if fill is not None:
        num = np.where(np.isnan(num), fill, num)
    return np.hstack([num, enc.transform(X[CATS])])


def fit_ridge(seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), NUMS),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), CATS)])
    model = make_pipeline(pre, Ridge(alpha=10.0)).fit(X_BASE.iloc[rows], y[rows])
    return model.predict(X_BASE.iloc[v]), model.predict(Xt)


def fit_hgb(seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan,
                         max_categories=250).fit(X_BASE.iloc[rows][CATS])
    mask = [False] * len(NUMS) + [True] * len(CATS)
    model = HistGradientBoostingRegressor(
        categorical_features=mask, learning_rate=0.05, max_iter=20 if small else 3000, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=100, random_state=seed)
    model.fit(numeric_with_codes(enc, X_BASE.iloc[rows]), y[rows])           # 内部留 10% 早停
    return model.predict(numeric_with_codes(enc, X_BASE.iloc[v])), model.predict(numeric_with_codes(enc, Xt))


def fit_forest(kind, seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_BASE.iloc[rows][CATS])
    common = dict(n_estimators=10 if small else 200, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    model = (RandomForestRegressor(max_features=0.3, max_samples=0.5, **common) if kind == 'rf'
             else ExtraTreesRegressor(max_features=0.5, **common))
    model.fit(numeric_with_codes(enc, X_BASE.iloc[rows], fill=-999.0), y[rows])
    return (model.predict(numeric_with_codes(enc, X_BASE.iloc[v], fill=-999.0)),
            model.predict(numeric_with_codes(enc, Xt, fill=-999.0)))


def fit_ebm(seed, a, s, v, Xt, small=False):
    from interpret.glassbox import ExplainableBoostingRegressor
    rows = np.concatenate([a, s])
    model = ExplainableBoostingRegressor(interactions=0 if small else 5, outer_bags=1 if small else 2,
                                         max_rounds=50 if small else 5000, n_jobs=-1, random_state=seed)
    model.fit(X_BASE.iloc[rows], y[rows])                                   # 内部留出验证集早停
    return np.asarray(model.predict(X_BASE.iloc[v])), np.asarray(model.predict(Xt))


def make_tabpfn():
    from tabpfn import TabPFNRegressor
    if TABPFN_TOKEN:
        return TabPFNRegressor(device='cuda', ignore_pretraining_limits=True)
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2)


def fit_tabpfn(seed, a, s, v, Xt, small=False):
    rows = np.concatenate([a, s])
    context = np.random.default_rng(seed).choice(rows, size=min(500 if small else TABPFN_CONTEXT_ROWS, len(rows)),
                                                 replace=False)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                         encoded_missing_value=np.nan).fit(X_BASE.iloc[context][CATS])
    model = make_tabpfn()
    model.fit(numeric_with_codes(enc, X_BASE.iloc[context]), y[context])
    p_val = np.asarray(model.predict(numeric_with_codes(enc, X_BASE.iloc[v])), dtype='float64')
    p_test = np.asarray(model.predict(numeric_with_codes(enc, Xt)), dtype='float64')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test


def fit_pytabkit(name, seed, a, s, v, Xt, small=False):
    if name == 'tabr':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'faiss-cpu'], check=False)
    import pytabkit
    cls = getattr(pytabkit, PYTABKIT_CLASSES[name])
    params = dict(device='cuda', random_state=seed, n_cv=1, n_refit=0, verbosity=0)
    if small:
        params['max_epochs'] = 2
    unknown = set(params) - set(inspect.signature(cls.__init__).parameters)
    require(not unknown, f'{name} 不支持参数 {sorted(unknown)}')
    Za, Zs, Zv, Zt = nn_frames(X_NN.iloc[a], X_NN.iloc[s], X_NN.iloc[v], Xt)    # 与 TabM 相同的填补
    mu, sd = y[a].mean(), y[a].std()
    m = cls(**params)
    m.fit(Za, (y[a] - mu) / sd, Zs, (y[s] - mu) / sd, cat_col_names=[c for c in Za.columns if c in CATS_NN])
    p_val = np.asarray(m.predict(Zv), dtype='float64').reshape(-1) * sd + mu
    p_test = np.asarray(m.predict(Zt), dtype='float64').reshape(-1) * sd + mu
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_test


def fit_new(name, seed, a, s, v, Xt, small=False):
    if name == 'ridge':
        return fit_ridge(seed, a, s, v, Xt, small)
    if name == 'hgb':
        return fit_hgb(seed, a, s, v, Xt, small)
    if name in ('rf', 'et'):
        return fit_forest(name, seed, a, s, v, Xt, small)
    if name == 'ebm':
        return fit_ebm(seed, a, s, v, Xt, small)
    if name == 'tabpfn':
        return fit_tabpfn(seed, a, s, v, Xt, small)
    return fit_pytabkit(name, seed, a, s, v, Xt, small)


def save_result(stem, arrays, info):
    np.savez(OUTPUT_DIR / f'{stem}.npz', **arrays)
    info = {**info, 'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUTPUT_DIR / f'{stem}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


# 预检：每个模型在 3,000 行上用很小的预算跑一次（几秒到一两分钟），失败的自动跳过
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
failed = {}
for name in COMPARE + OPTIONAL:
    started = time.perf_counter()
    try:
        p_val, p_test = fit_new(name, SEED, pa, ps, pv, XT_BASE.iloc[:300], small=True)
        p_val, p_test = np.asarray(p_val, dtype='float64'), np.asarray(p_test, dtype='float64')
        require(len(p_val) == len(pv) and len(p_test) == 300, '预测长度不符')
        require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测含非有限值')
        print(f'预检通过：{name:8s} RMSE（小预算）{rmse(y[pv], p_val):.1f}，用时 {time.perf_counter() - started:.0f}s')
    except Exception as exc:
        failed[name] = repr(exc)
        print(f'预检失败，跳过：{name} → {exc!r}')
RUN_NEW = [n for n in COMPARE + OPTIONAL if n not in failed]
print('将运行：', RUN_NEW)
'''),
    md('## 7. 五折训练（按模型逐个跑完，可续跑）'),
    code('''
def run_new(name, fold):
    stem = f'{name}__s{SEED}__fold{fold}'
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        return json.loads(json_path.read_text(encoding='utf-8')), 'cache'
    fit_idx, valid_idx = np.flatnonzero(FOLD_IDS != fold), np.flatnonzero(FOLD_IDS == fold)
    inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 11
    started = time.perf_counter()
    p_val, p_test = fit_new(name, SEED, inner_fit, inner_stop, valid_idx, XT_BASE)
    p_val, p_test = np.asarray(p_val, dtype='float64'), np.asarray(p_test, dtype='float64')
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
    info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val, 'test': p_test},
                       {'model': name, 'fold': fold, 'rmse': rmse(y[valid_idx], p_val),
                        'seconds': time.perf_counter() - started})
    return info, f'{info["seconds"]:.0f}s'


for name in RUN_NEW:
    started = time.perf_counter()
    try:
        for fold in range(1, N_FOLDS + 1):
            info, how = run_new(name, fold)
            print(f'{name:8s} fold {fold}  RMSE {info["rmse"]:.3f}  ({how})')
        print(f'—— {name} 完成，本次运行用时 {(time.perf_counter() - started) / 60:.1f} 分钟')
    except Exception as exc:
        print(f'!! {name} 运行出错，跳过剩下的折：{exc!r}')
'''),
    md('## 8. 比较：单独的分数（种子 42）与对最终融合的贡献（五折嵌套 NNLS）'),
    code('''
def load_folds(directory, stem_prefix):
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = directory / f'{stem_prefix}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    return oof, test_pred


def fold_rmse(pred):
    return np.array([rmse(y[FOLD_IDS == k], pred[FOLD_IDS == k]) for k in range(1, N_FOLDS + 1)])


def nested_nnls(P):
    out = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        w, _ = nnls(P[fit], y[fit])
        out[va] = P[va] @ w
    return out


# 11 的 6 个模型：种子 42 的 OOF（单独比较用）、三种子平均 OOF（融合用）、最好的测试预测
BASE = {}
for model in FINAL_MODELS:
    per_seed = {s: load_folds(PREV_DIR, f'{model}__s{s}') for s in PREV_SEEDS}
    require(all(v is not None for v in per_seed.values()), f'11 缺少 {model} 的结果')
    fulls = [np.load(p)['test'] for d in (PREV_DIR, BOOST_DIR) if d.is_dir() for p in sorted(d.glob(f'{model}__s*__full.npz'))]
    near = [np.load(p)['test'] for p in sorted(BOOST_DIR.glob('tabm__s*__nearfull.npz'))] if BOOST_DIR.is_dir() else []
    fold_avg = np.mean([t for _, t in per_seed.values()], axis=0)
    best_test = np.mean(fulls, axis=0) if fulls else (np.mean(near, axis=0) if (model == 'tabm' and near) else fold_avg)
    BASE[model] = {'oof42': per_seed[42][0], 'oof': np.mean([o for o, _ in per_seed.values()], axis=0),
                   'test': best_test}
NEW = {}
for name in COMPARE + OPTIONAL:
    loaded = load_folds(OUTPUT_DIR, f'{name}__s{SEED}')
    if loaded is not None:
        NEW[name] = {'oof42': loaded[0], 'oof': loaded[0], 'test': loaded[1]}

base_names = list(BASE)
P_base = np.column_stack([BASE[m]['oof'] for m in base_names])
base_nested = nested_nnls(P_base)
base_folds = fold_rmse(base_nested)
rows = []
for name, item in {**BASE, **NEW}.items():
    row = {'model': name, '来源': '11' if name in BASE else '13', 'oof_seed42': rmse(y, item['oof42']),
           'corr_with_blend': float(np.corrcoef(item['oof42'], base_nested)[0, 1])}
    if name in NEW:
        P = np.column_stack([P_base, item['oof']])
        nested = nested_nnls(P)
        delta = fold_rmse(nested) - base_folds
        w, _ = nnls(P, y)
        row.update({'blend_nested': rmse(y, nested), 'delta_vs_blend': rmse(y, nested) - rmse(y, base_nested),
                    'folds_better': int((delta < 0).sum()), 'weight_in_blend': float(w[-1])})
    rows.append(row)
compare = pd.DataFrame(rows).set_index('model').sort_values('oof_seed42')
print(f'11 的 6 模型融合（嵌套）：{rmse(y, base_nested):.3f}')
display(compare.round(3))
compare.to_csv(OUTPUT_DIR / 'summary_compare.csv')
seconds = {}
for p in OUTPUT_DIR.glob('*__fold*.json'):
    info = json.loads(p.read_text(encoding='utf-8'))
    seconds.setdefault(info['model'], []).append(info['seconds'])
print('每折平均用时（秒）：', {k: round(float(np.mean(v))) for k, v in seconds.items()})

KEEP_NEW = [n for n in NEW if compare.loc[n, 'delta_vs_blend'] <= ADD_DELTA and compare.loc[n, 'folds_better'] >= 4]
print('达到加入门槛的新模型：', KEEP_NEW or '无')
'''),
    md('## 9. 如有达标的新模型：加入融合 → 双胞胎 + S3 → 导出提交文件（与 11 相同的第二层）'),
    code('''
if KEEP_NEW:
    names = base_names + KEEP_NEW
    ITEMS = {**BASE, **{n: NEW[n] for n in KEEP_NEW}}
    P_all = np.column_stack([ITEMS[n]['oof'] for n in names])
    WEIGHTS_PLUS, _ = nnls(P_all, y)
    NESTED = nested_nnls(P_all)
    BLEND_TEST = {'v3_plus': np.column_stack([ITEMS[n]['test'] for n in names]) @ WEIGHTS_PLUS}
    print('权重：', dict(zip(names, WEIGHTS_PLUS.round(3))))
    print(f'嵌套融合：{rmse(y, base_nested):.3f} → {rmse(y, NESTED):.3f}')
else:
    NESTED, BLEND_TEST = None, None
    print('没有新模型达到门槛，最终提交保持 11 / 12 的版本。')
'''),
    code('''
if BLEND_TEST is not None:
''' + '\n'.join('    ' + line if line.strip() else line for line in STAGE2_CELL.splitlines()) + '''
print('完成。跑完告诉 Claude，本地会同步 OUTPUT_DIR 的结果。')
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
