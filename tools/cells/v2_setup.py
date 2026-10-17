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
