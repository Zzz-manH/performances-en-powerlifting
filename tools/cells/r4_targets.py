for model in TARGET_MODELS:
    if (model, 'R3') in failed:
        continue
    for fold in range(1, N_FOLDS + 1):
        info, how = run_one(model, 'R3', fold)
        print(f'fold {fold}  {model:8s}  RMSE {info["rmse"]:.3f}  最佳轮数 {info["best_iteration"]}  ({how})')

base_models = SCREEN_MODELS
P_base = np.column_stack([load_oof(m, 'R3')[0] for m in base_models])


def nested_nnls(P):
    out = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        w, _ = nnls(P[fit], y[fit])
        out[va] = P[va] @ w
    return out


base_nested = nested_nnls(P_base)
rows = [{'model': m, 'oof': rmse(y, load_oof(m, 'R3')[0])} for m in base_models]
for model in TARGET_MODELS:
    loaded = load_oof(model, 'R3')
    if loaded is None:
        continue
    nested = nested_nnls(np.column_stack([P_base, loaded[0]]))
    delta = fold_rmse(nested) - fold_rmse(base_nested)
    rows.append({'model': model, 'oof': rmse(y, loaded[0]),
                 'corr_with_cb_rel': float(np.corrcoef(loaded[0], load_oof('cb_rel', 'R3')[0])[0, 1]),
                 'blend_delta_nested': rmse(y, nested) - rmse(y, base_nested), 'folds_better': int((delta < 0).sum())})
target_table = pd.DataFrame(rows).set_index('model')
print(f'R3 三模型融合（嵌套）：{rmse(y, base_nested):.3f}')
display(target_table.round(3))
target_table.to_csv(OUTPUT_DIR / 'summary_target_engineering.csv')
print('完成。')
