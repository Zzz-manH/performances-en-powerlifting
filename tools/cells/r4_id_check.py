from scipy.stats import spearmanr

ids_all = pd.concat([train['id'], test['id']], ignore_index=True).to_numpy()
order = np.argsort(ids_all)
perm = np.random.default_rng(0).permutation(len(ids_all))
checks = {'same_meet': meet.astype(str).to_numpy(), 'same_federation': both['Federation'].astype(str).to_numpy(),
          'same_year': both['MeetYear'].to_numpy(), 'same_country': both['Country'].astype(str).to_numpy()}
rows = []
for name, values in checks.items():
    by_id = float((values[order][1:] == values[order][:-1]).mean())
    random = float((values[perm][1:] == values[perm][:-1]).mean())
    rows.append({'adjacent_rows': name, 'sorted_by_id': by_id, 'random_order': random, 'ratio': by_id / max(random, 1e-12)})
id_table = pd.DataFrame(rows).set_index('adjacent_rows')
display(id_table.round(4))
rho = float(spearmanr(ids_all, both['MeetYear'].to_numpy())[0])
print(f'Spearman correlation of id and meet year: {rho:.4f}')
ID_STRUCTURE = bool(id_table.loc['same_meet', 'ratio'] >= 1.5 or abs(rho) >= 0.1)
print('Meet structure in id: ', ID_STRUCTURE, '→', 'include the ID variant' if ID_STRUCTURE else 'skip ID: no clear ordering signal')
id_table.to_csv(OUTPUT_DIR / 'summary_id_check.csv')
