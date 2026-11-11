"""实验 16–17：成对相似度平滑。

第三层（实验 16，A_pair）：在 v4 的「融合 → 双胞胎」之后，用一个成对模型估计「训练行 j 的残差对行 i
有多大参考价值」（近似同一运动员：同 Sex × Country × State、出生年相差 ≤ 1），按权重汇总邻居残差。
第四层（实验 17，G/H）：在 A_pair 的残差上，对「同一场比赛 × 性别 × 装备」的行再做一次同样的成对平滑。

只读取已保存的 OOF / 测试预测，不重训第一层模型；只用 CPU，内存峰值约 6 GB。

用法：python pair_smoothing.py <项目根目录> [输出目录]
环境变量：NR（成对模型轮数，默认 300）、BAGS（成对模型种子数，默认 2）
输出：summary_experiments.csv、每个方案一份 submission_v5_<方案>.csv 和 oof_v5_<方案>.npy；
      第三层的特征缓存在 pair_feats_A.npz，重跑时直接读取。
"""
import os, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

ROOT = Path(sys.argv[1])
DATA_DIR = ROOT / 'data'
V4 = ROOT / 'artifacts/04_final_model/v4_eight_models'
CELLS = ROOT / 'tools/cells'
OUTPUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'artifacts/04_final_model/v5_pair_smoothing'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_FOLDS, SEED = 5, 42
NR, BAGS = int(os.environ.get('NR', 300)), int(os.environ.get('BAGS', 2))
MEET_CAP = 300                          # 比赛组内训练行超过这个数就不建候选对（大组里没有信号）
EXPECTED_SHA256 = {}
display = lambda x: print(x)
CORE9 = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']

# ---- 复用项目原有单元：读数据、双胞胎 + S3（得到 p1_cf、r1、F、FT、HALF、alphas …）
exec((CELLS / 'data_load.py').read_text())
NESTED = np.load(V4 / 'oof_nested_blend.npy')
BLEND_TEST = {}
exec((CELLS / 'stage2_twins_s3.py').read_text().replace('np.save(OUTPUT_DIR', 'pass  # np.save(OUTPUT_DIR'))
BASE_TEST = pd.read_csv(V4 / 'submission_final_v4_base.csv')
assert (BASE_TEST['id'].to_numpy() == test['id'].to_numpy()).all()
base_test = BASE_TEST[TARGET].to_numpy(dtype='float64')
REF = rmse(y, p2_cf)

# ---- 每行的属性数组；候选对只保存行号 (i, j)：i 为任意行（train+test），j 为有标签的训练行
ALLF = pd.concat([TR[RAW_FEATURES], TE[RAW_FEATURES]], ignore_index=True)
N = len(ALLF)


def code(cols):
    return pd.factorize(text_key(ALLF, cols))[0].astype('int32')


Q = pd.DataFrame({
    'i': np.arange(N, dtype='int32'),
    'g': code(['Sex', 'Country', 'State']),
    'mk': code(['Federation', 'MeetYear', 'MeetCountry', 'MeetState', 'Sex', 'Equipment']),
    'by': np.floor(ALLF.MeetYear - ALLF.Age + 1e-9).astype('int32'),
    'bw': ALLF.BodyweightKg.astype('float32'), 'yr': ALLF.MeetYear.astype('float32'),
    'age': ALLF.Age.astype('float32'), 'hf': ((ALLF.Age * 2).round() % 2).astype('int8'),
    'fed': code(['Federation']), 'eq': code(['Equipment']), 'wc': code(['WeightClassKg']),
    'tst': code(['Tested']), 'ms': code(['MeetCountry', 'MeetState']), 'ac': code(['AgeClass']),
    'ctry': code(['Country']), 'st': code(['Country', 'State']),
    'sexm': (ALLF.Sex == 'M').astype('int8'), 'st_unk': (ALLF.State == 'Unknown').astype('int8'),
    'usa': (ALLF.Country == 'USA').astype('int8'), 'known': (ALLF.Country != 'Unknown').astype('int8'),
})
QA = {c: Q[c].to_numpy() for c in Q.columns}


def build_athlete():
    """同 Sex|Country|State（国家已知），出生年相差 ≤ 1；两个 .5 年龄的出生年是精确的，必须相同。"""
    left = Q.loc[Q.known == 1, ['i', 'g', 'by']]
    right = left[left.i < n_train].rename(columns={'i': 'j'})
    out = []
    for d in (-1, 0, 1):
        m = left.merge(right.assign(by=right.by - d), on=['g', 'by'])
        i, j = m.i.to_numpy(), m.j.to_numpy()
        keep = (i != j) & ~((QA['hf'][i] == 1) & (QA['hf'][j] == 1) & (d != 0))
        out.append((i[keep], j[keep], np.full(int(keep.sum()), d, dtype='int8')))
    return [np.concatenate([o[k] for o in out]) for k in range(3)]


def build_meet():
    """同一场比赛（联盟 × 年份 × 比赛国 × 比赛州）× 性别 × 装备。"""
    right = Q.loc[Q.i < n_train, ['i', 'mk']].rename(columns={'i': 'j'})
    right = right[right.groupby('mk')['j'].transform('size') <= MEET_CAP]
    m = Q[['i', 'mk']].merge(right, on='mk')
    i, j = m.i.to_numpy(), m.j.to_numpy()
    keep = i != j
    return i[keep], j[keep], np.zeros(int(keep.sum()), dtype='int8')


PF_ATHLETE = ['dbw', 'sdbw', 'dyr', 'dby', 'hf', 'hf_j', 'same_fed', 'same_eq', 'same_wc', 'same_tst', 'same_ms',
              'n_cand', 'n_cand0', 'n_close', 'n_close05', 'rank_dbw', 'age', 'sexm', 'st_unk', 'usa', 'bw']
PF_MEET = ['dbw', 'dage', 'hf', 'hf_j', 'same_wc', 'same_ac', 'same_tst', 'same_ctry', 'same_st', 'st_unk_any',
           'n_cand', 'n_close', 'rank_dbw', 'age', 'sexm', 'usa', 'bw']
PARAMS = dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=2000,
              feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=0)


def pair_features(builder, PF, resid, tag):
    """resid：训练行的残差。返回 (训练特征, 测试特征)，各 5 列。"""
    idx_i, idx_j, dby = builder()
    n_pairs = len(idx_i)
    dbw = np.abs(QA['bw'][idx_i] - QA['bw'][idx_j])
    cnt = lambda w=None: np.bincount(idx_i, weights=w, minlength=N)[idx_i]
    same = lambda c: QA[c][idx_i] == QA[c][idx_j]
    make = {
        'dbw': lambda: dbw,
        'sdbw': lambda: (QA['bw'][idx_i] - QA['bw'][idx_j]) * np.sign(QA['yr'][idx_i] - QA['yr'][idx_j]),
        'dyr': lambda: np.abs(QA['yr'][idx_i] - QA['yr'][idx_j]),
        'dage': lambda: np.abs(QA['age'][idx_i] - QA['age'][idx_j]),
        'dby': lambda: dby, 'hf': lambda: QA['hf'][idx_i], 'hf_j': lambda: QA['hf'][idx_j],
        'same_fed': lambda: same('fed'), 'same_eq': lambda: same('eq'), 'same_wc': lambda: same('wc'),
        'same_tst': lambda: same('tst'), 'same_ms': lambda: same('ms'), 'same_ac': lambda: same('ac'),
        'same_ctry': lambda: same('ctry') & (QA['known'][idx_i] == 1),
        'same_st': lambda: same('st') & (QA['st_unk'][idx_i] == 0),
        'st_unk_any': lambda: (QA['st_unk'][idx_i] == 1) | (QA['st_unk'][idx_j] == 1),
        'n_cand': lambda: cnt(), 'n_cand0': lambda: cnt((dby == 0).astype('float64')),
        'n_close': lambda: cnt((dbw < 2).astype('float64')), 'n_close05': lambda: cnt((dbw < 0.5).astype('float64')),
        'rank_dbw': lambda: pd.Series(dbw).groupby(idx_i).rank(method='average').to_numpy(),
        'age': lambda: QA['age'][idx_i], 'sexm': lambda: QA['sexm'][idx_i], 'st_unk': lambda: QA['st_unk'][idx_i],
        'usa': lambda: QA['usa'][idx_i], 'bw': lambda: QA['bw'][idx_i],
    }
    X = np.empty((n_pairs, len(PF)), dtype='float32')
    for k, name in enumerate(PF):
        X[:, k] = make[name]()
    r_all = np.zeros(N)
    r_all[:n_train] = resid
    is_tr = idx_i < n_train
    rj = r_all[idx_j]
    target = (r_all[idx_i] * rj / float(np.var(resid))).astype('float32')    # 测试行为 0，不参与训练
    half_i = np.where(is_tr, HALF[np.minimum(idx_i, n_train - 1)], -1)
    half_j = HALF[idx_j]
    print(f'[{tag}] 候选对 {n_pairs:,}（train-train {int(is_tr.sum()):,}）', flush=True)

    def fit_predict(fit, ev, p):
        booster = lgb.train(p, lgb.Dataset(X[fit], target[fit], feature_name=PF, free_raw_data=True), NR)
        return booster.predict(X[ev])

    s_hat = np.zeros(n_pairs)
    for b in range(BAGS):
        p = dict(PARAMS, seed=SEED + b)
        for h in (0, 1):                    # 交叉拟合：训练对的两行都不在被预测的半边
            ev = half_i == h
            s_hat[ev] += fit_predict((half_i == 1 - h) & (half_j == 1 - h), ev, p) / BAGS
        s_hat[~is_tr] += fit_predict(is_tr, ~is_tr, p) / BAGS
        print(f'[{tag}] 成对模型 {b + 1}/{BAGS} 完成', flush=True)
    del X
    sp = np.clip(s_hat, 0, None)
    sr, s, s2 = (np.bincount(idx_i, weights=w, minlength=N) for w in (sp * rj, sp, sp * sp))
    order = np.lexsort((sp, idx_i))                          # 每个 i 权重最大的邻居
    last = np.r_[idx_i[order][1:] != idx_i[order][:-1], True]
    top = np.zeros(N)
    top[idx_i[order][last]] = (sp * rj)[order][last]
    feats = np.column_stack([sr, sr / (1 + s), sr / (0.3 + s), top, sr * s2 / np.maximum(s, 1e-9) / (1 + s)])
    return feats[:n_train], feats[n_train:]


def solve(Fm, target, ridge=0.0):
    """最小二乘；ridge > 0 时对标准化后的列加 ridge × 行数 的 L2 惩罚。"""
    if ridge == 0:
        return np.linalg.lstsq(Fm, target, rcond=None)[0]
    scale = Fm.std(axis=0) + 1e-12
    Z = Fm / scale
    return np.linalg.solve(Z.T @ Z + ridge * len(Z) * np.eye(Z.shape[1]), Z.T @ target) / scale


def crossfit(base, resid, Fm, ridge=0.0):
    out = base.copy()
    for h in (0, 1):
        fit, m = HALF != h, HALF == h
        out[m] = base[m] + Fm[m] @ solve(Fm[fit], resid[fit], ridge)
    return out


v4_sub = pd.read_csv(V4 / 'submission_final_v4_final.csv')[TARGET].to_numpy()
rows = [{'方案': 'v4（现有最终方案）', 'oof_rmse': REF}]


def record(name, oof, test_pred, ref_oof):
    final = np.clip(test_pred, 0, None)
    pd.DataFrame({'id': test['id'], TARGET: final}).to_csv(OUTPUT_DIR / f'submission_v5_{name}.csv', index=False)
    np.save(OUTPUT_DIR / f'oof_v5_{name}.npy', oof)
    better = lambda m: rmse(y[m], oof[m]) < rmse(y[m], ref_oof[m])
    rows.append({'方案': name, 'oof_rmse': rmse(y, oof), '相对 v4': rmse(y, oof) - REF,
                 '相对上一层': rmse(y, oof) - rmse(y, ref_oof),
                 '改善的折数': sum(better(FOLD_IDS == k) for k in range(1, 6)),
                 '改善的半边': sum(better(HALF == h) for h in (0, 1)),
                 '与 v4 提交的平均差': float(np.abs(v4_sub - final).mean())})
    print(f'{name}: {rmse(y, oof):.3f}', flush=True)


# ---- 第三层 A_pair：S3 + 近似同一运动员的成对平滑（特征有缓存就直接读）
cache = OUTPUT_DIR / 'pair_feats_A.npz'
if cache.is_file() and int(np.load(cache)['nr']) == NR and int(np.load(cache)['bags']) == BAGS:
    A_TR, A_TE = np.load(cache)['train'], np.load(cache)['test']
    print('读取缓存', cache.name)
else:
    A_TR, A_TE = pair_features(build_athlete, PF_ATHLETE, r1, '运动员')
    np.savez(cache, train=A_TR, test=A_TE, nr=NR, bags=BAGS)
FA, FTA = np.column_stack([F, A_TR]), np.column_stack([FT, A_TE])
oof_A = crossfit(p1_cf, r1, FA)
test_A = apply_twins(base_test, T_MATE, T_LEVEL, alphas) + FTA @ solve(FA, r1)
record('A_pair', oof_A, test_A, p2_cf)

# ---- 第四层：在 A_pair 的残差上，同一场比赛 × 性别 × 装备 的成对平滑
r3 = y - oof_A
M_TR, M_TE = pair_features(build_meet, PF_MEET, r3, '比赛')
for name, ridge in [('G_meet', 0.0), ('H_meet_ridge', 1e-2)]:
    record(name, crossfit(oof_A, r3, M_TR, ridge), test_A + M_TE @ solve(M_TR, r3, ridge), oof_A)

table = pd.DataFrame(rows).set_index('方案')
table.to_csv(OUTPUT_DIR / 'summary_experiments.csv')
pd.set_option('display.width', 250)
print(table.round(3).to_string())
print('结果写入', OUTPUT_DIR)
