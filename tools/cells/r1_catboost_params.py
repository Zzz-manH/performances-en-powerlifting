if RUN_PARAM_CHECK:
    for fold in range(1, N_FOLDS + 1):
        for key in [CATBOOST_DEPTH8_REGULARIZED, CATBOOST_DEPTH6_REGULARIZED]:
            if (MODEL_CATBOOST_ABSOLUTE, VARIANT_WEIGHT_CLASS_BASELINE, key) in failed:
                continue
            result, how = run_one(MODEL_CATBOOST_ABSOLUTE, BEST, key, fold)
            print(f'fold {fold}  catboost_{key}  {BEST:4s}  RMSE {result["rmse"]:.3f}  '
                  f'best iteration {result["best_iteration"]:5d}  ({how})')
    preds = {}
    for key in [CATBOOST_DEPTH8_BASELINE, CATBOOST_DEPTH8_REGULARIZED, CATBOOST_DEPTH6_REGULARIZED]:
        oof, test_pred = load_oof(f'{MODEL_CATBOOST_ABSOLUTE}_{key}', BEST)
        if oof is not None:
            preds[key], TESTS[(f'{MODEL_CATBOOST_ABSOLUTE}_{key}', BEST)] = oof, test_pred
    param_table = paired_table(preds, base=CATBOOST_DEPTH8_BASELINE)
    display(param_table.round(3))
    param_table.to_csv(OUTPUT_DIR / 'summary_catboost_params.csv')
    BEST_CB = param_table.index[0] if param_table.loc[param_table.index[0], 'keep'] else CATBOOST_DEPTH8_BASELINE
else:
    BEST_CB = CATBOOST_DEPTH8_BASELINE
tags = [MODEL_LIGHTGBM, MODEL_XGBOOST, f'{MODEL_CATBOOST_ABSOLUTE}_{BEST_CB}']
P_oof = np.column_stack([load_oof(tag, BEST)[0] for tag in tags])
P_test = np.column_stack([TESTS[(tag, BEST)] for tag in tags])
w, _ = nnls(P_oof, y)
FINAL = {'name': f'{BEST}_blend3_{BEST_CB}', 'oof': P_oof @ w, 'test': P_test @ w}
print(f'Final blend: features {BEST}, CatBoost {BEST_CB}, weights {dict(zip(tags, w.round(3)))}, '
      f'OOF RMSE {rmse(y, FINAL["oof"]):.3f}(F0 blend {rmse(y, BLENDS[VARIANT_WEIGHT_CLASS_BASELINE]["oof"]):.3f})')
