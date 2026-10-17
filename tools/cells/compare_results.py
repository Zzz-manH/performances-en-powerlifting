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
