# ---- 最终特征：10 的 ALL3（三个块都达标）----
_FRAMES3.clear()
VARIANTS3['ALL3'] = ['H2', 'E', 'PM']
FAMILY = {'lgb': 'lgb', 'lgb_rel': 'lgb', 'tabm': 'lgb', 'xgb': 'xgb', 'catboost': 'catboost', 'cb_rel': 'catboost'}
FRAMES = {m: frames3('ALL3', FAMILY[m]) for m in FINAL_MODELS}
for m, (Xtr_m, Xte_m, cats_m) in FRAMES.items():
    print(f'{m:9s} {Xtr_m.shape[1]} 列（{len(cats_m)} 个类别列），块：{blocks_for("ALL3", FAMILY[m])}')
