if RUN_PARAM_CHECK:
    for fold in range(1, N_FOLDS + 1):
        for key in ['C2', 'C3']:
            if ('catboost', 'F0', key) in failed:
                continue
            result, how = run_one('catboost', BEST, key, fold)
            print(f'fold {fold}  catboost_{key}  {BEST:4s}  RMSE {result["rmse"]:.3f}  '
                  f'最佳轮数 {result["best_iteration"]:5d}  ({how})')
    preds = {}
    for key in ['C1', 'C2', 'C3']:
        oof, test_pred = load_oof(f'catboost_{key}', BEST)
        if oof is not None:
            preds[key], TESTS[(f'catboost_{key}', BEST)] = oof, test_pred
    param_table = paired_table(preds, base='C1')
    display(param_table.round(3))
    param_table.to_csv(OUTPUT_DIR / 'summary_catboost_params.csv')
    BEST_CB = param_table.index[0] if param_table.loc[param_table.index[0], 'keep'] else 'C1'
else:
    BEST_CB = 'C1'
tags = ['lgb', 'xgb', f'catboost_{BEST_CB}']
P_oof = np.column_stack([load_oof(tag, BEST)[0] for tag in tags])
P_test = np.column_stack([TESTS[(tag, BEST)] for tag in tags])
w, _ = nnls(P_oof, y)
FINAL = {'name': f'{BEST}_blend3_{BEST_CB}', 'oof': P_oof @ w, 'test': P_test @ w}
print(f'最终组合：特征 {BEST}，CatBoost {BEST_CB}，权重 {dict(zip(tags, w.round(3)))}，'
      f'OOF RMSE {rmse(y, FINAL["oof"]):.3f}（F0 融合 {rmse(y, BLENDS["F0"]["oof"]):.3f}）')
