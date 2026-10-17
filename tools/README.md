# tools

- `build_notebooks.py`：生成项目的 5 本 notebook。
- `cells/*.py`：notebook 中的训练代码，每个文件是一个代码单元格。它们是从在 Colab 上跑过的实验 07–15 原样抽取的，只改了数据路径和结束时的提示文字。

用法：

```bash
python tools/build_notebooks.py <输出目录>     # 例如 python tools/build_notebooks.py /tmp/nb
```

`cells/` 的文件名前缀：

|前缀|来源|
|---|---|
|`data_load`、`features_round1`–`3`、`final_frames`、`final_model_defs`、`stage2_twins_s3`|多个实验共用：读数据、三轮特征、最终分模型特征表、最终模型定义、双胞胎 + S3 第二层|
|`r1_`、`r2_`、`r3_`、`r4_`|实验 07、09、10、14（02 的四轮特征消融）|
|`zoo_`、`compare_`|实验 08、13（03 的模型选择）|
|`v1_`、`v2_`、`v4_`|实验 11、12、15（04 的最终模型）|

notebook 中每个实验的结果目录改成了 `artifacts/<notebook>/<子目录>`，其余代码与原实验逐字相同；原始 notebook 和生成它们的脚本在 Git 历史的第一个提交中。

项目根目录里的 notebook 是在 Colab 上跑过、带输出的版本。直接输出到项目根目录会用无输出的版本覆盖它们，所以修改时先输出到其他目录，确认后再复制。
