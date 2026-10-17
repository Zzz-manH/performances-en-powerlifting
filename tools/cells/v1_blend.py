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
