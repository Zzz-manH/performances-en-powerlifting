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
