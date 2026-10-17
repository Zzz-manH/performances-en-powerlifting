# 提交记录

公开榜只用约 30% 测试数据（约 13,500 行）。选模以 CV 为准，公开榜只作确认。

|日期|文件|模型|CV RMSE|公开榜|
|---|---|---|---:|---:|
|2026-10-02|`../submission_baseline.csv`|sklearn GradientBoosting|82.60（折均值）|~80|
|2026-10-02|`../submission_3_blend.csv`|pipeline：LGBM + CatBoost|77.766|—|
|2026-10-02|`../outputs/submission_gpu_blend.csv`|v1：XGB + CatBoost + MLP|77.775|78.0|
|2026-10-16|`../outputs_v2/submission_FINAL_blend.csv`|v2：4 模型 × 多种子，10 折|77.302|77.799|
|2026-10-16|`20261016_v2_blend_twinfix.csv`|v2 + 训练集双胞胎后处理（519 行改动）|77.060|77.604|
|2026-10-16|`../artifacts/feature_round1_v1/submission_ALL_blend3_C2.csv`|07：ALL 特征 + CatBoost C2，三模型，5 折单种子|77.210|77.337|
|2026-10-16|`../artifacts/feature_round1_v1/submission_ALL_blend3_C2_twinfix.csv`|同上 + 双胞胎|76.966|77.158|
|2026-10-16|`../artifacts/feature_round1_v1/submission_F0_blend3_twinfix.csv`|07：F0 特征（对照）+ 双胞胎|77.701|77.947|
|2026-10-16|`../artifacts/model_zoo_v1_ALL/submission_zoo5f_base.csv`|08：07 模型 + TabM、cb_rel、lgb_rel、RealMLP，全局 NNLS（5 折单种子）|77.013|77.207|
|2026-10-16|`../artifacts/model_zoo_v1_ALL/submission_zoo5f_twins.csv`|同上 + 双胞胎|76.757|77.018|
|2026-10-16|`../artifacts/model_zoo_v1_ALL/submission_zoo5f_final.csv`|同上 + 同一运动员平滑|76.521|76.731|
|2026-10-16|`../artifacts/model_zoo_v1_ALL/submission_v2_final.csv`|v2 + 双胞胎 + 同一运动员平滑|76.774|77.125|
|2026-10-16|`../artifacts/feature_round3_v1/submission_ALL3_blend3_twins_S0_exact.csv`|10：ALL3 特征，三模型 + 双胞胎 + S0 平滑（5 折单种子）|76.539|76.775|
|2026-10-16|`../artifacts/feature_round3_v1/submission_ALL3_blend3_twins_S3_all_linear.csv`|10：同上，S3 合并平滑|76.510|76.716|
|2026-10-16|`../artifacts/feature_round3_v1/submission_ALL3_blend3_twinfix.csv`|10：ALL3 三模型 + 双胞胎（无运动员平滑）|76.649|76.884|
|2026-10-16|`../artifacts/final_v1/submission_final_fold_avg_final.csv`|11：6 模型 × 5 折 × 3 种子，折平均 + 双胞胎 + S3|76.044|76.520|
|2026-10-16|`../artifacts/final_v1/submission_final_hybrid_final.csv`|11：全量与折平均各半 + 双胞胎 + S3|—|76.421|
|2026-10-16|`../artifacts/final_v1/submission_final_full_final.csv`|11：树模型全量重训 × 3 种子（TabM 折平均）+ 双胞胎 + S3|—|76.360|
|2026-10-17|`../artifacts/final_v2/submission_final_v2_seeds_final.csv`|12：全量重训共 8 个种子（cb_rel 新种子用学习率 0.03），TabM 折平均|—|76.359|
|2026-10-17|`../artifacts/final_v2/submission_final_v2_seeds_tabmfull_final.csv`|12：同上 + TabM 近全量（95%）× 3 种子|—|76.337|
|2026-10-17|`../artifacts/model_compare_v1/submission_final_v3_plus_final.csv`|13：12 的全部 + sklearn HGB（权重 0.15）|76.010（交叉拟合，折平均口径）|**76.334**|
|2026-10-17|`../artifacts/feature_round4_v1/submission_R3_blend3_stage2.csv`|14：三模型 R3 + 双胞胎 + S3（对照）|76.361|76.592|
|2026-10-17|`../artifacts/feature_round4_v1/submission_AS_blend3_stage2.csv`|14：加运动员实力嵌套特征（变差）|76.508|76.802|
|2026-10-17|`../artifacts/final_v3/submission_final_v4_final.csv`|15 最终版：8 个模型（+ cb_gl、HGB 三种子）+ 双胞胎 + S3|待运行|待提交|

「—」：早期文件表示不确定是否提交过；11 的全量 / 混合版没有对应的 CV（全量重训无法交叉验证）。新提交后在这里补上公开榜分数。

**Kaggle 最终选择（两个）**：`../artifacts/final_v3/submission_final_v4_final.csv`（15，CV 最好）+ `../artifacts/model_compare_v1/submission_final_v3_plus_final.csv`（公开榜 76.334）。如果 15 在公开榜上比 76.334 差 0.05 以上，改选 v3_plus + `../artifacts/final_v2/submission_final_v2_seeds_tabmfull_final.csv`。
