"""生成 12_pl_colab_gpu_final_boost.ipynb（只生成文件并做语法检查，不在本地执行）。

环境、数据、特征、模型定义、第二层单元原样取自 11，读取 11 的结果（artifacts/final_v1）。
用法：/opt/anaconda3/bin/python3 tools/build_12_final_boost.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '12_pl_colab_gpu_final_boost.ipynb'
NB11 = nbf.read(ROOT / '11_pl_colab_gpu_final.ipynb', as_version=4)
ENV_CELL = NB11.cells[2].source.replace("EXPERIMENT_ID = 'final_v1'", "EXPERIMENT_ID = 'final_v2'")
DATA_CELL, FEATURE_CELL, ROUND2_CELL, FINAL_FEATURE_CELL = (NB11.cells[i].source for i in (4, 6, 8, 10))
MODEL_DEFS = NB11.cells[12].source.split('\n# ---- 预检')[0]
STAGE2_CELL = NB11.cells[16].source
assert "EXPERIMENT_ID = 'final_v2'" in ENV_CELL and 'REFIT_FACTOR' in ENV_CELL, '11 的环境单元位置变了'
assert 'FOLD_IDS' in DATA_CELL and "BLOCKS = {'P': P" in FEATURE_CELL and "BLOCKS2 = {'M2': M2" in ROUND2_CELL
assert "FRAMES = {m: frames3('ALL3'" in FINAL_FEATURE_CELL, '11 的最终特征单元位置变了'
assert 'def fit_full' in MODEL_DEFS and 'def tabm_fit_predict' in MODEL_DEFS and 'probe' not in MODEL_DEFS
assert 'NESTED' in STAGE2_CELL and 'BLEND_TEST' in STAGE2_CELL and 'submission_final_' in STAGE2_CELL


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md('''
# 12 · Powerlifting：最终冲刺——更多全量重训 + TabM 近全量（Colab GPU）

**11 的结果**

|模型（5 折 × 3 种子）|三种子平均 OOF|NNLS 权重|
|---|---:|---:|
|cb_rel（DOTS 相对力量 CatBoost）|**76.784**|0.453|
|TabM|77.050|0.217|
|CatBoost|77.097|0.042|
|lgb_rel|77.167|0.057|
|LightGBM|77.200|0.031|
|XGBoost|77.693|0.201|

融合（嵌套）76.395 → 双胞胎 76.172 → S3 平滑 76.044（交叉拟合）。公开榜：**全量重训 76.360**、混合 76.421、折平均 76.520。全量重训比折平均好 0.16，说明**训练数据越多越好**。11 的数据清洗实验：训练时去掉疑似行，LightGBM +0.043、XGBoost +0.063、CatBoost +0.071（都变差），保持不删。

**本 notebook 做三件事**（不重跑交叉验证，融合权重和第二层系数沿用 11）：

|步骤|内容|为什么|
|---|---|---|
|1. 学习率检查|cb_rel 用学习率 0.03（原 0.05），种子 42 五折，与 11 同折配对比较；平均 ≤ −0.02 且 ≥ 4/5 折更好才在后面的全量重训里使用|cb_rel 占融合 45%；更小学习率通常更稳|
|2. 更多全量重训种子|cb_rel、XGBoost、lgb_rel、LightGBM 各再加 5 个种子（共 8 个）|全量重训的测试预测目前只平均了 3 个种子|
|3. TabM 近全量|TabM 在 95% 训练行上训练、5% 早停，3 个种子|11 的「全量版」里 TabM 只有折模型（每个只用 72% 的数据），而 TabM 占融合 22%|

**输出**（`artifacts/final_v2/`）：
- `submission_final_v2_seeds_final.csv`：只加更多种子（TabM 仍用折平均）
- `submission_final_v2_seeds_tabmfull_final.csv`：再加 TabM 近全量（**预计最好**）

两个都交，和 11 的 76.360 配对比较。

**操作**：Colab 选 **T4 GPU**，从上到下运行；约 50–60 分钟，可续跑。需要 11 的结果目录 `artifacts/final_v1/` 完整。
'''),
    md('## 1. 环境与配置（与 11 相同，实验名不同）'),
    code(ENV_CELL + '''

PREV_DIR = PROJECT_DIR / 'artifacts' / 'final_v1'          # 11 的结果
PREV_SEEDS = [42, 7, 2024]
EXTRA_SEEDS = [11, 23, 57, 99, 123]                         # 全量重训新增的种子
EXTRA_MODELS = ['lgb', 'xgb', 'lgb_rel', 'cb_rel']          # 快的先跑
TABM_FULL_SEEDS = [42, 7, 2024]
TABM_HOLDOUT = 0.05                                         # TabM 近全量：留 5% 早停
RUN_LR_CHECK = True
if not (PREV_DIR / 'summary_weights.csv').is_file() or not (PREV_DIR / 'oof_nested_blend.npy').is_file():
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
    md('## 6. 模型定义（与 11 相同）与预检'),
    code(MODEL_DEFS + '''

CB_PARAMS_BASE = dict(CB_PARAMS)
CB_PARAMS_LR03 = dict(CB_PARAMS_BASE, learning_rate=0.03)


def with_cb(params, fn, *args, **kwargs):
    """临时替换 CatBoost 参数（gbdt_fit 在调用时读取全局 CB_PARAMS）。"""
    global CB_PARAMS
    saved = CB_PARAMS
    CB_PARAMS = params
    try:
        return fn(*args, **kwargs)
    finally:
        CB_PARAMS = saved


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


# 11 的结果必须完整：每个模型 3 个种子 × 5 折，树模型 3 个全量重训
for model in FINAL_MODELS:
    for s in PREV_SEEDS:
        for k in range(1, N_FOLDS + 1):
            require((PREV_DIR / f'{model}__s{s}__fold{k}.npz').is_file(), f'11 缺少 {model} 种子 {s} 第 {k} 折')
        if model != 'tabm':
            require((PREV_DIR / f'{model}__s{s}__full.npz').is_file(), f'11 缺少 {model} 种子 {s} 的全量重训')

# 预检：几秒内确认 cb_rel（学习率 0.03）和 TabM 近全量的代码路径可用
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
p_val, p_test, _ = with_cb(CB_PARAMS_LR03, fit_fold, 'cb_rel', SEED, pa, ps, pv, test_rows=np.arange(500), rounds=30)
require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'cb_rel 预检失败')
p_full = with_cb(CB_PARAMS_LR03, fit_full, 'cb_rel', SEED, 30, test_rows=np.arange(500))
require(np.isfinite(p_full).all(), 'cb_rel 全量重训预检失败')
TABM_OK = True
try:
    p_val, p_test = tabm_fit_predict(SEED, pa, ps, pv, XT_NN.iloc[:500], overrides={'n_epochs': 3})
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), 'TabM 预测无效')
except Exception as exc:
    try:                                               # 与 11 相同：分段线性嵌入不可用时去掉
        TABM_PARAMS = dict(TABM_FALLBACK)
        p_val, p_test = tabm_fit_predict(SEED, pa, ps, pv, XT_NN.iloc[:500], overrides={'n_epochs': 3})
        print('TabM 改用不带嵌入的版本：', repr(exc))
    except Exception as exc2:
        TABM_OK = False
        print('TabM 预检失败，跳过 TabM 近全量：', repr(exc2))
print('预检完成；TabM 近全量：', '可用' if TABM_OK else '跳过')
'''),
    md('## 7. 学习率检查：cb_rel 学习率 0.03 vs 0.05（种子 42 五折，与 11 同一划分配对）'),
    code('''
USE_LR03 = False
if RUN_LR_CHECK:
    lr_rows = []
    for fold in range(1, N_FOLDS + 1):
        stem = f'cb_rel_lr03__s{SEED}__fold{fold}'
        spec = {'model': 'cb_rel', 'params': dict(CB_PARAMS_LR03), 'columns': list(FRAMES['cb_rel'][0].columns)}
        info = load_cached(stem, spec)
        if info is None:
            fit_idx, valid_idx = np.flatnonzero(FOLD_IDS != fold), np.flatnonzero(FOLD_IDS == fold)
            inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=SEED)  # = 11
            started = time.perf_counter()
            p_val, _, best = with_cb(CB_PARAMS_LR03, fit_fold, 'cb_rel', SEED, inner_fit, inner_stop, valid_idx,
                                     test_rows=np.arange(10))
            info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val},
                               {'spec': spec, 'fold': fold, 'best_iteration': best,
                                'rmse': rmse(y[valid_idx], p_val), 'seconds': time.perf_counter() - started})
        base = json.loads((PREV_DIR / f'cb_rel__s{SEED}__fold{fold}.json').read_text(encoding='utf-8'))
        lr_rows.append({'fold': fold, 'lr_0.05': base['rmse'], 'lr_0.03': info['rmse'],
                        'delta': info['rmse'] - base['rmse'], 'best_iter_0.05': base['best_iteration'],
                        'best_iter_0.03': info['best_iteration'], 'minutes_0.03': info['seconds'] / 60})
        print(f'fold {fold}: lr 0.05 {base["rmse"]:.3f} → lr 0.03 {info["rmse"]:.3f}')
    lr_table = pd.DataFrame(lr_rows).set_index('fold')
    display(lr_table.round(3))
    lr_table.to_csv(OUTPUT_DIR / 'summary_lr_check.csv')
    USE_LR03 = bool(lr_table['delta'].mean() <= -0.02 and (lr_table['delta'] < 0).sum() >= 4)
print('cb_rel 全量重训使用学习率', 0.03 if USE_LR03 else 0.05)
'''),
    md('## 8. 更多全量重训种子 + TabM 近全量（可续跑）'),
    code('''
def prev_best_iterations(model):
    return [json.loads((PREV_DIR / f'{model}__s{s}__fold{k}.json').read_text(encoding='utf-8'))['best_iteration']
            for s in PREV_SEEDS for k in range(1, N_FOLDS + 1)]


def refit_plan(model):
    """(参数, 轮数)：轮数 = CV 平均最佳轮数 × REFIT_FACTOR。"""
    if model == 'cb_rel' and USE_LR03:
        bests = [json.loads((OUTPUT_DIR / f'cb_rel_lr03__s{SEED}__fold{k}.json').read_text(encoding='utf-8'))
                 ['best_iteration'] for k in range(1, N_FOLDS + 1)]
        return CB_PARAMS_LR03, int(round(np.mean(bests) * REFIT_FACTOR))
    return CB_PARAMS_BASE, int(round(np.mean(prev_best_iterations(model)) * REFIT_FACTOR))


for seed in EXTRA_SEEDS:
    for model in EXTRA_MODELS:
        params, rounds = refit_plan(model)
        stem = f'{model}__s{seed}__full'
        spec = {'model': model, 'rounds': rounds, 'columns': list(FRAMES[model][0].columns),
                'cb_params': dict(params) if model in ('catboost', 'cb_rel') else None}
        info = load_cached(stem, spec)
        if info is None:
            started = time.perf_counter()
            p_test = with_cb(params, fit_full, model, seed, rounds)
            require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
            info = save_result(stem, {'test': p_test}, {'spec': spec, 'seed': seed,
                                                        'seconds': time.perf_counter() - started})
            how = f'{info["seconds"]:.0f}s'
        else:
            how = 'cache'
        print(f'种子 {seed:3d} 全量重训 {model:8s} 轮数 {rounds}  ({how})')

if TABM_OK:
    for seed in TABM_FULL_SEEDS:
        stem = f'tabm__s{seed}__nearfull'
        spec = {'model': 'tabm', 'params': dict(TABM_PARAMS), 'holdout': TABM_HOLDOUT,
                'columns': list(X_NN.columns)}
        info = load_cached(stem, spec)
        if info is None:
            fit_rows, stop_rows = train_test_split(np.arange(n_train), test_size=TABM_HOLDOUT, random_state=seed)
            started = time.perf_counter()
            p_stop, p_test = tabm_fit_predict(seed, fit_rows, stop_rows, stop_rows, XT_NN)
            require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
            info = save_result(stem, {'test': p_test}, {'spec': spec, 'seed': seed,
                                                        'holdout_rmse': rmse(y[stop_rows], p_stop),
                                                        'seconds': time.perf_counter() - started})
        print(f'TabM 近全量 种子 {seed}：早停集 RMSE {info["holdout_rmse"]:.3f}（仅供参考，早停用过它）')
'''),
    md('## 9. 组装测试预测（权重沿用 11）'),
    code('''
weights = pd.read_csv(PREV_DIR / 'summary_weights.csv')
NAMES, W = weights['model'].tolist(), weights['weight'].to_numpy(dtype='float64')
NESTED = np.load(PREV_DIR / 'oof_nested_blend.npy')
require(len(NESTED) == n_train and set(NAMES) <= set(FINAL_MODELS), '11 的权重或 OOF 不符')


def fold_average(model):
    return np.mean([np.load(PREV_DIR / f'{model}__s{s}__fold{k}.npz')['test']
                    for s in PREV_SEEDS for k in range(1, N_FOLDS + 1)], axis=0)


rows, PER_MODEL = [], {}
for model in NAMES:
    if model == 'tabm':
        fold_avg = fold_average('tabm')
        near = [np.load(p)['test'] for p in sorted(OUTPUT_DIR.glob('tabm__s*__nearfull.npz'))]
        PER_MODEL[model] = {'seeds': fold_avg, 'seeds_tabmfull': np.mean(near, axis=0) if near else fold_avg}
        rows.append({'model': model, 'test_models': f'{len(near)} 个近全量（另一版用 15 个折模型）'})
    else:
        files = sorted(PREV_DIR.glob(f'{model}__s*__full.npz')) + sorted(OUTPUT_DIR.glob(f'{model}__s*__full.npz'))
        full = np.mean([np.load(p)['test'] for p in files], axis=0)
        PER_MODEL[model] = {'seeds': full, 'seeds_tabmfull': full}
        rows.append({'model': model, 'test_models': f'{len(files)} 个全量重训'})
display(pd.DataFrame(rows).assign(weight=W).set_index('model'))

BLEND_TEST = {f'v2_{kind}': np.column_stack([PER_MODEL[m][kind] for m in NAMES]) @ W
              for kind in ['seeds', 'seeds_tabmfull']}
prev_full = pd.read_csv(PREV_DIR / 'submission_final_full_base.csv')[TARGET].to_numpy()
for kind, pred in BLEND_TEST.items():
    print(f'{kind}：与 11 全量版（未加第二层）的平均绝对差 {np.abs(np.clip(pred, 0, None) - prev_full).mean():.2f} kg')
'''),
    md('## 10. 第二层：双胞胎 + S3 平滑（与 11 相同，系数由 11 的嵌套 OOF 拟合）→ 导出提交文件'),
    code(STAGE2_CELL + '''
print('完成。请提交 submission_final_v2_seeds_final.csv 和 submission_final_v2_seeds_tabmfull_final.csv，'
      '与 11 的 76.360 比较；跑完告诉 Claude。')
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
