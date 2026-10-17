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
        print(f'\n=== {tag}')
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
print('\n=== 三模型 NNLS 融合（主要判读依据）')
display(blend_table.round(3))
BEST = blend_table.index[0]
print(f'融合 OOF 最好的变体：{BEST}；达到保留标准的变体：{blend_table.index[blend_table["keep"]].tolist() or "无"}')

pd.concat({tag: t for tag, t in SINGLE.items()}, names=['model']).to_csv(OUTPUT_DIR / 'summary_single.csv')
blend_table.to_csv(OUTPUT_DIR / 'summary_blend.csv')
