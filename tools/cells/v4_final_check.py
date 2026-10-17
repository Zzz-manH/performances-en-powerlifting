sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
final_path = OUTPUT_DIR / 'submission_final_v4_final.csv'
final = pd.read_csv(final_path)
require(final.columns.tolist() == ['id', TARGET], '列名应为 id, TotalKg')
require(len(final) == len(sample) == 45000 and final['id'].is_unique
        and set(final['id']) == set(sample['id']), 'id 与 sample_submission 不一致')   # Kaggle 按 id 匹配，顺序无关
require(np.isfinite(final[TARGET]).all() and (final[TARGET] > 0).all(), '预测含无效值')
print(f'检查通过：{final_path.name}，{len(final)} 行；预测均值 {final[TARGET].mean():.1f}，'
      f'范围 {final[TARGET].min():.1f}–{final[TARGET].max():.1f}（训练集均值 {y.mean():.1f}）')
previous = COMPARE_DIR / 'submission_final_v3_plus_final.csv'
if previous.is_file():
    prev = pd.read_csv(previous)
    diff = final[TARGET].to_numpy() - prev[TARGET].to_numpy()
    print(f'与当前公开榜最好的 v3_plus（76.334）相比：相关 {np.corrcoef(final[TARGET], prev[TARGET])[0, 1]:.5f}，'
          f'平均绝对差 {np.abs(diff).mean():.2f} kg，最大 {np.abs(diff).max():.1f} kg')
print()
print('==== 最终在 Kaggle 上选择这两个提交 ====')
print('1.', final_path.relative_to(PROJECT_DIR), '（本 notebook：8 个模型 + 双胞胎 + S3，CV 最好）')
print('2.', previous.relative_to(PROJECT_DIR) if previous.is_file() else '目前公开榜最好的提交',
      '（目前公开榜最好的 76.334，作为保险）')
print('如果第 1 个在公开榜上明显变差（比 76.334 差 0.05 以上），改选 v3_plus 和', (FINAL_V2 / 'submission_final_v2_seeds_tabmfull_final.csv').relative_to(PROJECT_DIR))
