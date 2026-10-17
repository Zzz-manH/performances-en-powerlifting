"""生成 15_pl_colab_gpu_final_last.ipynb（最终一次运行；只生成文件并做语法检查，不在本地执行）。

环境、数据、特征、模型定义、第二层单元原样取自 11；复用 11–14 已保存的全部结果，只补训练缺的部分。
用法：/opt/anaconda3/bin/python3 tools/build_15_final.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = '15_pl_colab_gpu_final_last.ipynb'
NB11 = nbf.read(ROOT / '11_pl_colab_gpu_final.ipynb', as_version=4)
ENV_CELL = NB11.cells[2].source.replace("EXPERIMENT_ID = 'final_v1'", "EXPERIMENT_ID = 'final_v3'")
DATA_CELL, FEATURE_CELL, ROUND2_CELL, FINAL_FEATURE_CELL = (NB11.cells[i].source for i in (4, 6, 8, 10))
MODEL_DEFS = NB11.cells[12].source.split('\n# ---- 预检')[0]
STAGE2_CELL = NB11.cells[16].source
assert "EXPERIMENT_ID = 'final_v3'" in ENV_CELL and 'REFIT_FACTOR' in ENV_CELL, '11 的环境单元位置变了'
assert 'FOLD_IDS' in DATA_CELL and "BLOCKS = {'P': P" in FEATURE_CELL and "BLOCKS2 = {'M2': M2" in ROUND2_CELL
assert "FRAMES = {m: frames3('ALL3'" in FINAL_FEATURE_CELL, '11 的最终特征单元位置变了'
assert 'def gbdt_fit' in MODEL_DEFS and 'def tabm_fit_predict' in MODEL_DEFS and 'probe' not in MODEL_DEFS
assert 'NESTED' in STAGE2_CELL and 'BLEND_TEST' in STAGE2_CELL and 'submission_final_' in STAGE2_CELL


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md('''
# 15 · Powerlifting：最终版（最后一次运行）

**到目前为止**（公开榜，越低越好）

|提交|内容|公开榜|
|---|---|---:|
|**13 `v3_plus`**|11 的 6 模型 + 12 的改进（全量重训 8 个种子、cb_rel 学习率 0.03、TabM 95% 重训）+ sklearn HGB|**76.334**|
|12 `v2_seeds_tabmfull`|同上，不含 HGB|76.337|
|12 `v2_seeds`|不含 TabM 95% 重训|76.359|
|11 `full`|全量重训 3 个种子|76.360|
|14 `R3` / `AS`|三模型融合；加入「运动员实力」嵌套特征后变差|76.592 / 76.802|

14 的结论：基于成绩的嵌套特征（运动员实力 +0.14、比赛实力 +0.52）在 CV 和公开榜上都变差，不用；`id` 顺序是随机的；**IPF GL 目标的 CatBoost（`cb_gl`）单独 77.16，与最强的 cb_rel 一样，加入三模型融合 −0.076（5/5 折）**。

**本 notebook 做什么**（只补训练缺的部分，其余全部复用 11–14 已保存的结果）

|步骤|内容|约用时|
|---|---|---:|
|1|`cb_gl` 补种子 7、2024 的五折（种子 42 复用 14），得到与其他模型可比的三种子 OOF|17 分钟|
|2|sklearn HGB 补种子 7、2024 的五折（种子 42 复用 13）|5 分钟|
|3|`cb_gl` 全量重训 × 5 个种子（学习率 0.03，轮数按 12 的经验换算）|15 分钟|
|4|HGB 在全部数据上重训（内部留 10% 早停）× 3 个种子|3 分钟|
|5|TabM 95% 重训再加 2 个种子（共 5 个）|16 分钟|
|6|8 个模型 NNLS 融合（五折嵌套给出诚实分数）→ 双胞胎 + S3 平滑 → 导出 → 自动检查|2 分钟|

每一步都有保护：某一步失败会打印原因，后面用已有的结果继续（例如 `cb_gl` 失败就用 14 的种子 42 结果），一定会写出提交文件。

**最后在 Kaggle 上选两个最终提交**（私榜按另外 70% 的测试数据算）：
1. `artifacts/final_v3/submission_final_v4_final.csv`（本 notebook，CV 最好）
2. `artifacts/model_compare_v1/submission_final_v3_plus_final.csv`（目前公开榜最好的 76.334）

**操作**：Colab 选 **T4 GPU**，「全部运行」，约 1 小时；断线后重新「全部运行」会跳过已完成的部分。
'''),
    md('## 1. 环境与配置（与 11 相同，实验名不同）'),
    code(ENV_CELL + '''

FINAL_V1 = PROJECT_DIR / 'artifacts' / 'final_v1'                 # 11
FINAL_V2 = PROJECT_DIR / 'artifacts' / 'final_v2'                 # 12
COMPARE_DIR = PROJECT_DIR / 'artifacts' / 'model_compare_v1'      # 13
ROUND4_DIR = PROJECT_DIR / 'artifacts' / 'feature_round4_v1'      # 14
PREV_SEEDS = [42, 7, 2024]
GL_FULL_SEEDS = [42, 7, 2024, 11, 23]
HGB_FULL_SEEDS = [42, 7, 2024]
TABM_EXTRA_SEEDS = [11, 23]
TABM_HOLDOUT = 0.05
for directory in (FINAL_V1, FINAL_V2):
    if not directory.is_dir():
        raise FileNotFoundError(f'缺少之前的结果目录：{directory}')
print('之前的结果：', {d.name: d.is_dir() for d in (FINAL_V1, FINAL_V2, COMPARE_DIR, ROUND4_DIR)})
'''),
    md('## 2. 读取并核对数据（与 07 相同）'),
    code(DATA_CELL),
    md('## 3. 第一轮特征（与 07 相同）'),
    code(FEATURE_CELL),
    md('## 4. 第二轮特征（与 09 相同）'),
    code(ROUND2_CELL),
    md('## 5. 第三轮特征与各模型的最终特征表（与 11 相同）'),
    code(FINAL_FEATURE_CELL),
    md('## 6. 模型定义（与 11 相同）+ GL 目标、HGB + 预检'),
    code(MODEL_DEFS + '''

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


def gl_scale(sex_values, equipment_values, bw_values):
    """IPF GL 系数的分母 / 100：总成绩 = GL 分 × gl_scale。Raw / Wraps 用无装备系数，其余用装备系数。"""
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


GL = gl_scale(train['Sex'], train['Equipment'], train['BodyweightKg'])
GL_TEST = gl_scale(test['Sex'], test['Equipment'], test['BodyweightKg'])
require(min(GL.min(), GL_TEST.min()) > 0, 'GL 系数异常')
SCALE_OF = {'cb_rel': (SCALE, SCALE_TEST), 'cb_gl': (GL, GL_TEST)}
FRAMES['cb_gl'] = FRAMES['cb_rel']                    # 与 14 相同：CatBoost 族的 ALL3 特征
PATIENCE['cb_gl'] = PATIENCE['cb_rel']
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


def fit_matrix(model, a, s, v, seed, rounds=MAX_ROUNDS, test_rows=None):
    """与 14 相同：相对力量目标（按系数² 加权），返回 (外层验证预测, 测试预测, 最佳轮数)。"""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale, scale_test = SCALE_OF[model]
    scale_t = scale_test if test_rows is None else scale_test[test_rows]
    norm = (scale[a] ** 2).mean()
    m, best = gbdt_fit(model, seed, Xtr.iloc[a], y[a] / scale[a], scale[a] ** 2 / norm,
                       Xtr.iloc[s], y[s] / scale[s], scale[s] ** 2 / norm, rounds)
    p_val = np.asarray(gbdt_predict(model, m, best, Xtr.iloc[v]), dtype='float64') * scale[v]
    p_test = np.asarray(gbdt_predict(model, m, best, Xt), dtype='float64') * scale_t
    return p_val, p_test, best


def fit_full_scaled(model, seed, rounds, test_rows=None):
    """全部训练行、固定轮数、不早停的相对力量目标模型。返回测试预测。"""
    Xtr, Xte, _ = FRAMES[model]
    Xt = Xte if test_rows is None else Xte.iloc[test_rows]
    scale, scale_test = SCALE_OF[model]
    scale_t = scale_test if test_rows is None else scale_test[test_rows]
    m, _ = gbdt_fit(model, seed, Xtr, y / scale, scale ** 2 / (scale ** 2).mean(), rounds=rounds)
    return np.asarray(gbdt_predict(model, m, rounds, Xt), dtype='float64') * scale_t


X_BASE, XT_BASE, CATS = FRAMES['lgb']                # 与 13 相同：LightGBM / TabM 的 ALL3 特征
NUMS = [c for c in X_BASE.columns if c not in CATS]


def numeric_with_codes(enc, X):
    return np.hstack([X[NUMS].to_numpy(dtype='float64'), enc.transform(X[CATS])])


def fit_hgb(seed, rows, v, Xt, small=False):
    """与 13 相同：rows 上训练（内部留 10% 早停）。返回 (v 的预测或 None, 测试预测, 实际轮数)。"""
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan,
                         max_categories=250).fit(X_BASE.iloc[rows][CATS])
    model = HistGradientBoostingRegressor(
        categorical_features=[False] * len(NUMS) + [True] * len(CATS), learning_rate=0.05,
        max_iter=20 if small else 3000, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=100, random_state=seed)
    model.fit(numeric_with_codes(enc, X_BASE.iloc[rows]), y[rows])
    p_val = None if v is None else np.asarray(model.predict(numeric_with_codes(enc, X_BASE.iloc[v])), dtype='float64')
    return p_val, np.asarray(model.predict(numeric_with_codes(enc, Xt)), dtype='float64'), int(model.n_iter_)


def folds_for(seed):
    ids = np.zeros(n_train, dtype=int)
    for k, (_, valid) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n_train)), start=1):
        ids[valid] = k
    return ids


def save_result(stem, arrays, info):
    np.savez(OUTPUT_DIR / f'{stem}.npz', **arrays)
    info = {**info, 'gpu': GPU_NAME, 'versions': VERSIONS, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUTPUT_DIR / f'{stem}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info


def cached(stem):
    json_path, npz_path = OUTPUT_DIR / f'{stem}.json', OUTPUT_DIR / f'{stem}.npz'
    if json_path.is_file() and npz_path.is_file():
        return json.loads(json_path.read_text(encoding='utf-8'))
    return None


require(np.array_equal(folds_for(SEED), FOLD_IDS), '种子 42 的五折与之前不一致')
rng = np.random.default_rng(SEED)
probe = rng.choice(n_train, 3000, replace=False)
pa, ps, pv = probe[:2000], probe[2000:2500], probe[2500:]
OK = {}
try:
    p_val, p_test, _ = fit_matrix('cb_gl', pa, ps, pv, SEED, rounds=30, test_rows=np.arange(300))
    p_full = with_cb(CB_PARAMS_LR03, fit_full_scaled, 'cb_gl', SEED, 30, test_rows=np.arange(300))
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all() and np.isfinite(p_full).all(), '预测无效')
    OK['cb_gl'] = True
except Exception as exc:
    OK['cb_gl'] = False
    print('cb_gl 预检失败，之后只用 14 的种子 42 结果：', repr(exc))
try:
    p_val, p_test, _ = fit_hgb(SEED, np.concatenate([pa, ps]), pv, XT_BASE.iloc[:300], small=True)
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测无效')
    OK['hgb'] = True
except Exception as exc:
    OK['hgb'] = False
    print('HGB 预检失败，之后只用 13 的种子 42 结果：', repr(exc))
try:
    p_val, p_test = tabm_fit_predict(SEED, pa, ps, pv, XT_NN.iloc[:300], overrides={'n_epochs': 3})
    require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), '预测无效')
    OK['tabm'] = True
except Exception as exc:
    OK['tabm'] = False
    print('TabM 预检失败，TabM 只用 12 的 3 个 95% 重训：', repr(exc))
print('预检：', OK)
'''),
    md('## 7. 步骤 1–2：cb_gl 与 HGB 补齐三个种子的五折（种子 42 复用 14 / 13）'),
    code('''
def adopt_previous(stem_prev_dir, prev_stem, new_stem):
    """把之前保存的种子 42 五折结果复制到本目录（统一命名）。"""
    for k in range(1, N_FOLDS + 1):
        src = stem_prev_dir / f'{prev_stem}__fold{k}.npz'
        if cached(f'{new_stem}__fold{k}') is None and src.is_file():
            saved = np.load(src)
            src_json = src.with_suffix('.json')
            info = json.loads(src_json.read_text(encoding='utf-8')) if src_json.is_file() else {}
            save_result(f'{new_stem}__fold{k}', {'valid_idx': saved['valid_idx'], 'val': saved['val'],
                                                 'test': saved['test']},
                        {'source': str(src), 'rmse': rmse(y[saved['valid_idx']], saved['val']),
                         'best_iteration': info.get('best_iteration')})


adopt_previous(ROUND4_DIR, 'cb_gl__R3', 'cb_gl__s42')
adopt_previous(COMPARE_DIR, 'hgb__s42', 'hgb__s42')


def run_cv(model, seed):
    ids = folds_for(seed)
    for k in range(1, N_FOLDS + 1):
        stem = f'{model}__s{seed}__fold{k}'
        info = cached(stem)
        if info is None:
            fit_idx, valid_idx = np.flatnonzero(ids != k), np.flatnonzero(ids == k)
            inner_fit, inner_stop = train_test_split(fit_idx, test_size=INNER_STOP_FRACTION, random_state=seed)
            started = time.perf_counter()
            if model == 'cb_gl':
                p_val, p_test, best = fit_matrix('cb_gl', inner_fit, inner_stop, valid_idx, seed)
            else:
                p_val, p_test, best = fit_hgb(seed, fit_idx, valid_idx, XT_BASE)
            require(np.isfinite(p_val).all() and np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
            info = save_result(stem, {'valid_idx': valid_idx, 'val': p_val, 'test': p_test},
                               {'model': model, 'seed': seed, 'fold': k, 'rmse': rmse(y[valid_idx], p_val),
                                'best_iteration': best, 'seconds': time.perf_counter() - started})
            how = f'{info["seconds"]:.0f}s'
        else:
            how = 'cache'
        print(f'{model:6s} 种子 {seed:4d} fold {k}  RMSE {info["rmse"]:.3f}  ({how})')


for model in ['hgb', 'cb_gl']:                       # HGB 在 CPU 上很快，先跑
    if not OK.get(model):
        continue
    for seed in PREV_SEEDS:
        try:
            run_cv(model, seed)
        except Exception as exc:
            print(f'!! {model} 种子 {seed} 出错，跳过：{exc!r}')
'''),
    md('## 8. 步骤 3–5：全量重训（cb_gl × 5 个种子、HGB × 3 个种子、TabM 95% × 2 个新种子）'),
    code('''
def mean_best_iterations(model):
    bests = []
    for seed in PREV_SEEDS:
        for k in range(1, N_FOLDS + 1):
            info = cached(f'{model}__s{seed}__fold{k}')
            if info is not None and info.get('best_iteration'):
                bests.append(info['best_iteration'])
    return float(np.mean(bests)) if bests else None


try:                                             # 12 中 cb_rel：学习率 0.03 的最佳轮数约为 0.05 的 1.4 倍
    lr_check = pd.read_csv(FINAL_V2 / 'summary_lr_check.csv')
    LR03_RATIO = float(lr_check['best_iter_0.03'].mean() / lr_check['best_iter_0.05'].mean())
except Exception:
    LR03_RATIO = 1.44
print(f'学习率 0.03 / 0.05 的轮数比：{LR03_RATIO:.2f}')

if OK.get('cb_gl') and mean_best_iterations('cb_gl'):
    gl_rounds = int(round(mean_best_iterations('cb_gl') * LR03_RATIO * REFIT_FACTOR))
    for seed in GL_FULL_SEEDS:
        stem = f'cb_gl__s{seed}__full'
        try:
            info = cached(stem)
            if info is None:
                started = time.perf_counter()
                p_test = with_cb(CB_PARAMS_LR03, fit_full_scaled, 'cb_gl', seed, gl_rounds)
                require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
                info = save_result(stem, {'test': p_test}, {'model': 'cb_gl', 'seed': seed, 'rounds': gl_rounds,
                                                            'learning_rate': 0.03,
                                                            'seconds': time.perf_counter() - started})
            print(f'cb_gl 全量重训 种子 {seed}：轮数 {info["rounds"]}，用时 {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! cb_gl 全量重训 种子 {seed} 出错，跳过：{exc!r}')

if OK.get('hgb'):
    for seed in HGB_FULL_SEEDS:
        stem = f'hgb__s{seed}__nearfull'
        try:
            info = cached(stem)
            if info is None:
                started = time.perf_counter()
                _, p_test, n_iter = fit_hgb(seed, np.arange(n_train), None, XT_BASE)
                info = save_result(stem, {'test': p_test}, {'model': 'hgb', 'seed': seed, 'n_iter': n_iter,
                                                            'seconds': time.perf_counter() - started})
            print(f'HGB 全部数据（内部 10% 早停）种子 {seed}：轮数 {info["n_iter"]}，用时 {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! HGB 全量 种子 {seed} 出错，跳过：{exc!r}')

if OK.get('tabm'):
    for seed in TABM_EXTRA_SEEDS:
        stem = f'tabm__s{seed}__nearfull'
        try:
            info = cached(stem)
            if info is None:
                fit_rows, stop_rows = train_test_split(np.arange(n_train), test_size=TABM_HOLDOUT, random_state=seed)
                started = time.perf_counter()
                p_stop, p_test = tabm_fit_predict(seed, fit_rows, stop_rows, stop_rows, XT_NN)
                require(np.isfinite(p_test).all(), f'{stem} 预测含非有限值')
                info = save_result(stem, {'test': p_test}, {'model': 'tabm', 'seed': seed,
                                                            'holdout_rmse': rmse(y[stop_rows], p_stop),
                                                            'seconds': time.perf_counter() - started})
            print(f'TabM 95% 重训 种子 {seed}：用时 {info["seconds"]:.0f}s')
        except Exception as exc:
            print(f'!! TabM 种子 {seed} 出错，跳过：{exc!r}')
'''),
    md('## 9. 步骤 6：8 个模型的 NNLS 融合（五折嵌套给出诚实分数）'),
    code('''
def load_cv(directory, stem):
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for k in range(1, N_FOLDS + 1):
        path = directory / f'{stem}__fold{k}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    return (oof, test_pred) if not np.isnan(oof).any() else None


def full_files(model):
    if model in ('lgb', 'xgb', 'lgb_rel', 'catboost', 'cb_rel'):
        return sorted(FINAL_V1.glob(f'{model}__s*__full.npz')) + sorted(FINAL_V2.glob(f'{model}__s*__full.npz'))
    if model == 'tabm':
        return sorted(FINAL_V2.glob('tabm__s*__nearfull.npz')) + sorted(OUTPUT_DIR.glob('tabm__s*__nearfull.npz'))
    if model == 'hgb':
        return sorted(OUTPUT_DIR.glob('hgb__s*__nearfull.npz'))
    return sorted(OUTPUT_DIR.glob(f'{model}__s*__full.npz'))


ALL_MODELS = ['lgb', 'xgb', 'lgb_rel', 'catboost', 'cb_rel', 'tabm', 'hgb', 'cb_gl']
PER, rows = {}, []
for model in ALL_MODELS:
    directory = FINAL_V1 if model in FINAL_MODELS else OUTPUT_DIR
    loaded = {s: load_cv(directory, f'{model}__s{s}') for s in PREV_SEEDS}
    loaded = {s: v for s, v in loaded.items() if v is not None}
    if not loaded:
        print(f'{model}：没有可用的交叉验证结果，跳过')
        continue
    oof = np.mean([o for o, _ in loaded.values()], axis=0)
    fold_avg = np.mean([t for _, t in loaded.values()], axis=0)
    files = full_files(model)
    full = np.mean([np.load(p)['test'] for p in files], axis=0) if files else fold_avg
    PER[model] = {'oof': oof, 'full': full}
    rows.append({'model': model, 'cv_seeds': list(loaded), 'oof_seed_avg': rmse(y, oof),
                 'test_from': f'{len(files)} 个全量 / 95% 重训' if files else '折平均'})


def nested_nnls(P):
    out = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        w, _ = nnls(P[fit], y[fit])
        out[va] = P[va] @ w
    return out


NAMES = list(PER)
P = np.column_stack([PER[n]['oof'] for n in NAMES])
WEIGHTS, _ = nnls(P, y)
NESTED = nested_nnls(P)
model_table = pd.DataFrame(rows).set_index('model')
model_table['weight'] = pd.Series(dict(zip(NAMES, WEIGHTS)))
display(model_table.round(3))
model_table.to_csv(OUTPUT_DIR / 'summary_models.csv')

steps = {'11 的 6 个模型': [m for m in NAMES if m in FINAL_MODELS],
         '+ HGB': [m for m in NAMES if m in FINAL_MODELS + ['hgb']],
         '+ HGB + cb_gl（最终）': NAMES}
step_rows = [{'融合': k, '模型数': len(v), '嵌套 OOF RMSE': rmse(y, nested_nnls(np.column_stack([PER[n]['oof'] for n in v])))}
             for k, v in steps.items() if v]
display(pd.DataFrame(step_rows).set_index('融合').round(3))
BLEND_TEST = {'v4': np.column_stack([PER[n]['full'] for n in NAMES]) @ WEIGHTS}
print('NNLS 权重：', dict(zip(NAMES, WEIGHTS.round(3))))
'''),
    md('## 10. 第二层：双胞胎 + S3 平滑（与 11 相同）→ 导出提交文件'),
    code(STAGE2_CELL),
    md('## 11. 最终检查与提交建议'),
    code('''
sample = pd.read_csv(PROJECT_DIR / 'sample_submission.csv')
final_path = OUTPUT_DIR / 'submission_final_v4_final.csv'
final = pd.read_csv(final_path)
require(final.columns.tolist() == ['id', TARGET], '列名应为 id, TotalKg')
require(len(final) == len(sample) == 45000 and final['id'].is_unique
        and set(final['id']) == set(sample['id']), 'id 与 sample_submission 不一致')   # Kaggle 按 id 匹配，顺序无关
require(np.isfinite(final[TARGET]).all() and (final[TARGET] > 0).all(), '预测含无效值')
print(f'检查通过：{final_path.name}，{len(final)} 行；预测均值 {final[TARGET].mean():.1f}，'
      f'范围 {final[TARGET].min():.1f}–{final[TARGET].max():.1f}（训练集均值 {y.mean():.1f}）')
previous = COMPARE_DIR / 'submission_final_v3_plus_final.csv'
if previous.is_file():
    prev = pd.read_csv(previous)
    diff = final[TARGET].to_numpy() - prev[TARGET].to_numpy()
    print(f'与当前公开榜最好的 v3_plus（76.334）相比：相关 {np.corrcoef(final[TARGET], prev[TARGET])[0, 1]:.5f}，'
          f'平均绝对差 {np.abs(diff).mean():.2f} kg，最大 {np.abs(diff).max():.1f} kg')
print()
print('==== 最终在 Kaggle 上选择这两个提交 ====')
print('1.', final_path.relative_to(PROJECT_DIR), '（本 notebook：8 个模型 + 双胞胎 + S3，CV 最好）')
print('2.', previous.relative_to(PROJECT_DIR) if previous.is_file() else '目前公开榜最好的提交',
      '（目前公开榜最好的 76.334，作为保险）')
print('如果第 1 个在公开榜上明显变差（比 76.334 差 0.05 以上），改选 v3_plus 和 final_v2/submission_final_v2_seeds_tabmfull_final.csv。')
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
