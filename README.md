# Powerlifting：预测 TotalKg

**结果见 [RESULTS.md](RESULTS.md)，后续方案见 [PLAN.md](PLAN.md)**，提交记录在 [submissions/](submissions/)。

## 工作方式

本地（Mac）只生成 notebook、读结果、做语法检查，**不在本地训练**。notebook 由 `tools/build_*.py` 生成，复制到 Drive 的 `我的云端硬盘/Colab Notebooks/performances-en-powerlifting/`，在 Colab T4 GPU 上从上到下运行。结果写到 Drive 上同一项目的 `artifacts/<实验名>/`，跑完后同步回本地。

## 文件

|文件 / 目录|内容|
|---|---|
|`train.csv` `test.csv` `sample_submission.csv`|原始数据，只读（与 Drive 逐字节一致）|
|`eda.ipynb`|探索性分析|
|`modeling.ipynb`|sklearn 基线（GB 82.60）|
|`feature_engineering.ipynb`|早期 FE；目标编码有泄漏，80.40 不可信|
|`pipeline.ipynb`|LGBM / CatBoost + 特征组消融（本地 CPU，77.766）|
|`colab_gpu_pipeline.ipynb`|v1：XGB + CatBoost + MLP（Colab GPU，77.775，公开榜 78.0）；Drive 上带运行输出的版本|
|`colab_gpu_pipeline_v2.ipynb`|v2：10 折 × 3 种子，4 模型融合（77.302）|
|`07_pl_colab_gpu_feature_round1.ipynb`|记录精度 / 跨地参赛 / 比赛背景特征块 + CatBoost 正则化参数（五折配对）；带运行输出|
|`08_pl_colab_gpu_model_zoo.ipynb`|07 之后：RealMLP、TabM、相对力量目标 GBDT、堆叠比较、同一运动员残差平滑（同时用于 v2）|
|`09_pl_colab_gpu_feature_round2.ipynb`|第二轮特征：比赛背景加强、组别背景、运动员历史、DOTS 系数、CatBoost 专用比赛 / 运动员 ID|
|`10_pl_colab_gpu_feature_round3.ipynb`|第三轮特征：运动员生涯、比赛经验结构、按模型挑特征；第二层平滑 5 种做法比较|
|`11_pl_colab_gpu_final.ipynb`|最终集成：6 个模型 × 5 折 × 3 种子 + 全量重训，NNLS + 双胞胎 + S3 平滑；附数据清洗实验；公开榜 76.360|
|`12_pl_colab_gpu_final_boost.ipynb`|最终冲刺：更多全量重训种子、TabM 近全量、cb_rel 学习率检查（读取 11 的结果）|
|`13_pl_colab_gpu_model_compare.ipynb`|其他模型大比较：Ridge、HGB、ExtraTrees、随机森林、TabPFN、MLP-PLR、ResNet、EBM（同一特征和五折），看单独分数与对融合的贡献|
|`14_pl_colab_gpu_feature_round4.ipynb`|第四轮特征：基于成绩的嵌套折外特征（运动员实力 AS、比赛实力 MS）、`id` 结构检查、目标变量工程（异速生长 / IPF GL）|
|`15_pl_colab_gpu_final_last.ipynb`|**最终版**：复用 11–14，补 cb_gl（IPF GL 目标）与 HGB 的三种子和全量重训、TabM 再加 2 个种子；8 模型 NNLS + 双胞胎 + S3|
|`outputs/` `outputs_v2/`|v1、v2 在 Colab 上的 OOF / 测试预测与提交文件（从 Drive 同步）|
|`artifacts/`|07 起的实验结果（从 Drive 同步）|
|`submissions/`|提交文件与记录|
|`tools/`|notebook 生成脚本；`v2_blend_twinfix.py`（双胞胎后处理，只读预测，不训练）|
|根目录 `submission_*.csv`|modeling / feature_engineering / pipeline 的早期提交文件|

## 2026-10-16 与 Drive 的同步

- `outputs/`（v1）和 `outputs_v2/`（v2）从 Drive 复制到本地。
- `colab_gpu_pipeline.ipynb`：Drive 版本只多一个查找文件的 `!find` 单元，并带有 v1 的完整运行输出，已替换本地版本。
- `colab_gpu_pipeline_v2.ipynb`：保留本地版本。Drive 版本把 `TUNE` 改成了 True，带的是一次没有 GPU、只跑了开头几格的会话输出；`outputs_v2/` 的文件名说明完整运行时 `TUNE=False`。
- 其余 notebook 和三个 CSV 两边一致。Drive 上的 `catboost_info/`（训练日志）和根目录下空的 `performances-en-powerlifting/outputs/` 没有同步。
