from scipy.stats import spearmanr

ids_all = pd.concat([train['id'], test['id']], ignore_index=True).to_numpy()
order = np.argsort(ids_all)
perm = np.random.default_rng(0).permutation(len(ids_all))
checks = {'同一场比赛': meet.astype(str).to_numpy(), '同一联盟': both['Federation'].astype(str).to_numpy(),
          '同一比赛年份': both['MeetYear'].to_numpy(), '同一国家': both['Country'].astype(str).to_numpy()}
rows = []
for name, values in checks.items():
    by_id = float((values[order][1:] == values[order][:-1]).mean())
    random = float((values[perm][1:] == values[perm][:-1]).mean())
    rows.append({'相邻两行': name, '按 id 排序': by_id, '随机顺序': random, '倍数': by_id / max(random, 1e-12)})
id_table = pd.DataFrame(rows).set_index('相邻两行')
display(id_table.round(4))
rho = float(spearmanr(ids_all, both['MeetYear'].to_numpy())[0])
print(f'id 与比赛年份的 Spearman 相关：{rho:.4f}')
ID_STRUCTURE = bool(id_table.loc['同一场比赛', '倍数'] >= 1.5 or abs(rho) >= 0.1)
print('id 含有比赛结构：', ID_STRUCTURE, '→', '加入 ID 变体' if ID_STRUCTURE else '不测试 id（顺序是随机的）')
id_table.to_csv(OUTPUT_DIR / 'summary_id_check.csv')
