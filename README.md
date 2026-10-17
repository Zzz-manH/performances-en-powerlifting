# Powerlifting：预测三项总成绩 TotalKg

根据比赛记录预测力量举选手的三项总成绩 `TotalKg`（kg），评估指标 RMSE（越低越好）。训练集 105,000 行、测试集 45,000 行，16 个特征（3 数值 + 13 类别）。

**最终提交**：[submissions/20261017_FINAL_v4_eight_models.csv](submissions/20261017_FINAL_v4_eight_models.csv)，五折交叉验证 **75.966**。公开榜目前最好的是 v3_plus，**76.334**；起点 v1 是 78.0。全部实验数字见 [RESULTS.md](RESULTS.md)，提交记录见 [submissions/README.md](submissions/README.md)。

## 5 本 notebook

|notebook|内容|结果目录|原实验|
|---|---|---|---|
|[01_data_exploration](01_data_exploration.ipynb)|读取与核对原始数据、探索分析、记录精度与「双胞胎」、数据质量检查、固定五折|—|—|
|[02_feature_engineering](02_feature_engineering.ipynb)|四轮特征：记录精度、跨地参赛、比赛背景、运动员历史与生涯、基于成绩的特征，及五折消融|`artifacts/02_feature_engineering/`|07、09、10、14|
|[03_model_selection](03_model_selection.ipynb)|早期基线、CatBoost 参数、其他模型与相对力量目标、堆叠方式、目标变量工程、模型大比较|`artifacts/03_model_selection/`|08、13|
|[04_final_model](04_final_model.ipynb)|8 个模型 × 5 折 × 3 种子 + 全量重训、NNLS 融合、双胞胎 + 同一运动员平滑、提交文件|`artifacts/04_final_model/`|11、12、15|
|[05_error_analysis](05_error_analysis.ipynb)|最终残差、残差结构与第二层、数据清洗实验、未采用的方法、交叉验证与公开榜|—|—|

「实验 07–15」是实验的原始编号。代码注释、RESULTS.md 和提交记录都沿用这些编号，对照见文末。

## 最终方案

|步骤|内容|
|---|---|
|特征|原始 16 列 + 体重级别 + 记录精度 + 跨地参赛 + 比赛背景 + 运动员历史与生涯 + 比赛经验结构；按模型不同（57 / 73 / 78 列）|
|模型|LightGBM、XGBoost、CatBoost、lgb_rel、cb_rel、cb_gl、TabM、sklearn HGB。`_rel` / `_gl` 预测 DOTS / IPF GL 相对力量，再乘回 kg|
|验证|`KFold(5, shuffle=True, random_state=种子)` × 3 个种子（42、7、2024）；外层训练部分内留 10% 早停|
|测试预测|树模型在全部 105,000 行上重训：轮数 = CV 平均最佳轮数 × 1.2，多个种子平均。TabM、HGB 在 95% / 全部数据上重训，内部早停|
|融合|全局 NNLS；五折嵌套得到诚实分数 76.319|
|第二层|双胞胎：训练集中特征完全相同的行，把预测拉向其真实成绩。同一运动员残差平滑（S3）：精确出生年、出生年 ±1、时间衰减三种键，线性组合。两者都交叉拟合，最终 75.966|

## 运行方式

- 整个项目文件夹放在 Google Drive 的 `Colab Notebooks/` 下，在 Colab 打开 notebook，「全部运行」。notebook 会自动找到项目里的 `data/`；本地 Jupyter 也可以运行，依赖见 `requirements.txt`。
- **默认只读**：所有训练开关为 `False`，只读取 `artifacts/` 中保存的结果，生成表格和图。CPU 运行时即可，每本几分钟。
- **重新训练**：
  - 把 notebook 顶部 `TRAIN` 中的一项改为 `True`，运行时类型选 **T4 GPU**；一次只开一个开关。
  - 每个模型训练完立即写入 `artifacts/`，断线后重新运行会跳过已完成的部分。
  - 依赖在 notebook 内自动安装（CatBoost 固定 1.2.10，以及 pytabkit）。
- `artifacts/` 约 400MB，不在 Git 中，完整副本在 Google Drive。
- 不使用外部数据：OpenPowerlifting 原始库里就有测试行，用它等于标签泄漏。

## 目录

```
data/                                     原始数据（只读）：train.csv、test.csv、sample_submission.csv
01–05 *.ipynb                             5 本 notebook
artifacts/                                实验结果（不在 Git 中）
  02_feature_engineering/round1–round4    实验 07、09、10、14
  03_model_selection/baseline_v1、v2      早期 v1、v2 的逐折预测
  03_model_selection/model_zoo            实验 08
  03_model_selection/model_compare        实验 13（含 v3_plus 提交）
  04_final_model/v1_six_models            实验 11
  04_final_model/v2_more_seeds            实验 12
  04_final_model/v4_eight_models          实验 15（最终提交）
submissions/                              最终提交文件与完整提交记录
tools/                                    生成 5 本 notebook 的脚本（见 tools/README.md）
RESULTS.md                                全部实验结果
```

## 原实验编号对照

|原 notebook|内容|现在的位置|原结果目录 → 现在|
|---|---|---|---|
|`eda`、`modeling`、`feature_engineering`、`pipeline`|早期探索、sklearn 基线、本地 LightGBM / CatBoost|01、03 第 1 节|—|
|`colab_gpu_pipeline`、`_v2`|v1、v2：XGBoost / CatBoost / MLP（/ LightGBM）多种子融合|03 第 1 节|`outputs/`、`outputs_v2/` → `03_model_selection/baseline_v1`、`baseline_v2`|
|07|第一轮特征 + CatBoost 参数 + 双胞胎|02 第 5.1 节、03 第 2 节|`feature_round1_v1` → `02_feature_engineering/round1`|
|08|其他模型 + 堆叠 + 同一运动员平滑|03 第 3 节、05 第 2 节|`model_zoo_v1_ALL` → `03_model_selection/model_zoo`|
|09|第二轮特征|02 第 5.2 节|`feature_round2_v1` → `02_feature_engineering/round2`|
|10|第三轮特征 + 第二层平滑比较|02 第 5.3 节、05 第 2 节|`feature_round3_v1` → `02_feature_engineering/round3`|
|11|6 个模型的最终集成 + 数据清洗实验|04 第 1 节、05 第 3 节|`final_v1` → `04_final_model/v1_six_models`|
|12|更多全量重训种子 + TabM 近全量 + 学习率检查|04 第 2 节、03 第 6 节|`final_v2` → `04_final_model/v2_more_seeds`|
|13|其他模型大比较 + v3_plus|03 第 5 节、04 第 3 节|`model_compare_v1` → `03_model_selection/model_compare`|
|14|第四轮特征 + id 检查 + 目标变量工程|02 第 5.4 节、01、03 第 4 节|`feature_round4_v1` → `02_feature_engineering/round4`|
|15|8 个模型的最终版|04 第 4 节|`final_v3` → `04_final_model/v4_eight_models`|

这些原始 notebook（带 Colab 运行输出）和生成它们的脚本保存在 Git 历史的第一个提交中。
