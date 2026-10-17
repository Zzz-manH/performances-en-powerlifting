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
