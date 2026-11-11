sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
final_path = OUTPUT_DIR / SUBMISSION_FINAL_FILENAME
final = pd.read_csv(final_path)
require(final.columns.tolist() == ['id', TARGET], 'Expected columns: id, TotalKg')
require(len(final) == len(sample) == 45000 and final['id'].is_unique
        and final['id'].equals(test['id']) and set(final['id']) == set(sample['id']),
        'Les id doivent suivre l’ordre de test.csv et correspondre à ceux de sample_submission.csv')
require(np.isfinite(final[TARGET]).all() and (final[TARGET] > 0).all(), 'Invalid predictions')
print(f'Checks passed: {final_path.name}, {len(final)} rows; prediction mean {final[TARGET].mean():.1f}, '
      f'range {final[TARGET].min():.1f}–{final[TARGET].max():.1f}(training mean {y.mean():.1f})')
previous = COMPARE_DIR / 'submission_final_v3_plus_final.csv'
if previous.is_file():
    prev = pd.read_csv(previous)
    diff = final[TARGET].to_numpy() - prev[TARGET].to_numpy()
    print(f'Compared with archived v3_plus (public RMSE 76.334): correlation {np.corrcoef(final[TARGET], prev[TARGET])[0, 1]:.5f}, '
          f'mean absolute difference {np.abs(diff).mean():.2f} kg, max {np.abs(diff).max():.1f} kg')
print()
print('==== Archived final selection ====')
print('1.', final_path.relative_to(PROJECT_DIR), '(ensemble final : huit candidats, correspondances et lissage combiné ; RMSE publique archivée 76.298)')
print('2.', previous.relative_to(PROJECT_DIR) if previous.is_file() else 'archived v3_plus submission',
      '(alternative: recorded public RMSE 76.334)')
print('Other archived alternative: v3_plus and', (EXPANDED_SEED_ENSEMBLE_DIR / 'submission_final_v2_seeds_tabmfull_final.csv').relative_to(PROJECT_DIR))
