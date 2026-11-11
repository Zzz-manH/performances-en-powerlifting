for model in TARGET_MODELS:
    if (model, VARIANT_FINAL_BASELINE) in failed:
        continue
    for fold in range(1, N_FOLDS + 1):
        info, how = run_one(model, VARIANT_FINAL_BASELINE, fold)
        print(f'fold {fold}  {model:8s}  RMSE {info["rmse"]:.3f}  best iteration {info["best_iteration"]}  ({how})')

base_models = SCREEN_MODELS
P_base = np.column_stack([load_oof(m, VARIANT_FINAL_BASELINE)[0] for m in base_models])


def nested_nnls(oof_prediction_matrix):
    out = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        w, _ = nnls(oof_prediction_matrix[fit], y[fit])
        out[va] = oof_prediction_matrix[va] @ w
    return out


base_nested = nested_nnls(P_base)
rows = [{'model': m, 'oof': rmse(y, load_oof(m, VARIANT_FINAL_BASELINE)[0])} for m in base_models]
for model in TARGET_MODELS:
    loaded = load_oof(model, VARIANT_FINAL_BASELINE)
    if loaded is None:
        continue
    nested = nested_nnls(np.column_stack([P_base, loaded[0]]))
    delta = fold_rmse(nested) - fold_rmse(base_nested)
    rows.append({'model': model, 'oof': rmse(y, loaded[0]),
                 'corr_with_cb_rel': float(np.corrcoef(loaded[0], load_oof(MODEL_CATBOOST_DOTS, VARIANT_FINAL_BASELINE)[0])[0, 1]),
                 'blend_delta_nested': rmse(y, nested) - rmse(y, base_nested), 'folds_better': int((delta < 0).sum())})
target_table = pd.DataFrame(rows).set_index('model')
print(f'Combinaison à trois modèles avec variables finales (poids croisés) : {rmse(y, base_nested):.3f}')
display(target_table.round(3))
target_table.to_csv(OUTPUT_DIR / 'summary_target_engineering.csv')
print('Done.')
