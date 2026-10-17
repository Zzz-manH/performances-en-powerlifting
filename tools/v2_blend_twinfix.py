"""v2 融合 + 「训练集双胞胎」后处理。

只读取 outputs_v2/ 中已保存的逐折预测和 train/test，不训练任何模型（本地运行几秒钟）。

背景：约 1% 的行在训练集里有一条 16 个特征完全相同的「双胞胎」，其中 73% 的 TotalKg 也完全相同
（OpenPowerlifting 中同一次成绩登记在多个组别 / 测试与非测试组别）。树模型不会记忆单行，
所以对这些行可以把预测往双胞胎的真实成绩拉：pred + alpha * (双胞胎均值 - pred)。
alpha 按匹配层级（exact = 16 列全同；core = 年龄、体重、年份、性别、联盟、比赛地、居住地相同）
在训练 OOF 上用最小二乘估计，并按 core 键分两半交叉拟合报告诚实的 RMSE。

用法：python3 tools/v2_blend_twinfix.py
输出：submissions/20261016_v2_blend_twinfix.csv 与同名 .json 记录
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / 'outputs_v2'
OUT_CSV = ROOT / 'submissions' / '20261016_v2_blend_twinfix.csv'
OUT_JSON = OUT_CSV.with_suffix('.json')

TARGET = 'TotalKg'
CORE = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState',
        'Country', 'State']
V2_SEEDS = {'lgb': [42], 'xgb': [42, 7, 2024], 'catboost': [42, 7, 2024], 'mlp': [42, 7, 2024]}
N_FOLDS = 10


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def rebuild_v2(y, n_test):
    """按 v2 notebook 的方式重建每个模型的种子平均 OOF / 测试预测，再做 NNLS 融合。"""
    oofs, tests = {}, {}
    for model, seeds in V2_SEEDS.items():
        oof, pred = np.zeros(len(y)), np.zeros(n_test)
        for seed in seeds:
            folds = KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(len(y)))
            for i, (_, valid) in enumerate(folds):
                saved = np.load(V2_DIR / f'{model}_s{seed}_f{i}.npz')
                assert len(saved['val']) == len(valid) and len(saved['test']) == n_test
                oof[valid] += saved['val'] / len(seeds)
                pred += saved['test'] / (N_FOLDS * len(seeds))
        oofs[model], tests[model] = oof, pred
    names = list(V2_SEEDS)
    P, PT = np.column_stack([oofs[n] for n in names]), np.column_stack([tests[n] for n in names])
    w, _ = nnls(P, y)
    return P @ w, PT @ w, dict(zip(names, w.round(4).tolist())), {n: rmse(y, oofs[n]) for n in names}


def row_keys(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def mate_stats(train_keys, y, query_keys=None):
    """query_keys=None：训练行之间留一（不含自己）；否则：查询行在整个训练集中的同键行。"""
    table = pd.DataFrame({'k': train_keys, 'y': y}).groupby('k')['y'].agg(['sum', 'count'])
    if query_keys is None:
        keys = pd.Series(train_keys)
        n = keys.map(table['count']).to_numpy() - 1
        s = keys.map(table['sum']).to_numpy() - y
    else:
        keys = pd.Series(query_keys)
        n = keys.map(table['count']).fillna(0).to_numpy()
        s = keys.map(table['sum']).fillna(0).to_numpy()
    return n, np.where(n > 0, s / np.maximum(n, 1), np.nan)


def twin_level(n_exact, n_core):
    return np.select([n_exact > 0, n_core > 0], ['exact', 'core'], default='none')


def fit_alpha(y, pred, mate):
    d = mate - pred
    return float(np.clip(np.sum((y - pred) * d) / np.sum(d * d), 0.0, 1.0))


def apply_twins(pred, mate, level, alphas):
    out = pred.copy()
    for name, alpha in alphas.items():
        m = level == name
        out[m] = pred[m] + alpha * (mate[m] - pred[m])
    return out


def main():
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    features = [c for c in test.columns if c != 'id']
    assert len(features) == 16 and set(features) <= set(train.columns)
    y = train[TARGET].to_numpy(dtype=float)

    oof, pred_test, weights, single = rebuild_v2(y, len(test))
    saved = pd.read_csv(V2_DIR / 'submission_FINAL_blend.csv')
    assert saved['id'].equals(test['id'])
    gap = float(np.abs(saved[TARGET].to_numpy() - np.clip(pred_test, 0, None)).max())
    assert gap < 1e-3, f'重建的 v2 融合与 submission_FINAL_blend.csv 不一致：{gap}'

    exact_tr, core_tr = row_keys(train, features), row_keys(train, CORE)
    exact_te, core_te = row_keys(test, features), row_keys(test, CORE)

    # 训练集：留一统计（模拟测试行在训练集里找到双胞胎的情形）
    n_ex, m_ex = mate_stats(exact_tr, y)
    n_co, m_co = mate_stats(core_tr, y)
    level = twin_level(n_ex, n_co)
    mate = np.where(level == 'exact', m_ex, m_co)

    alphas = {lv: fit_alpha(y[level == lv], oof[level == lv], mate[level == lv])
              for lv in ['exact', 'core']}
    fixed = apply_twins(oof, mate, level, alphas)

    # 交叉拟合：按 core 键分两半（同一组双胞胎在同一半），一半估 alpha、另一半评分
    half = (core_tr % 2).astype(int)
    crossfit = oof.copy()
    for h in (0, 1):
        fit_rows, score_rows = half != h, half == h
        a = {lv: fit_alpha(y[fit_rows & (level == lv)], oof[fit_rows & (level == lv)],
                           mate[fit_rows & (level == lv)]) for lv in ['exact', 'core']}
        crossfit[score_rows] = apply_twins(oof[score_rows], mate[score_rows], level[score_rows], a)

    # 测试集：在全部训练行里找双胞胎
    tn_ex, tm_ex = mate_stats(exact_tr, y, exact_te)
    tn_co, tm_co = mate_stats(core_tr, y, core_te)
    t_level = twin_level(tn_ex, tn_co)
    t_mate = np.where(t_level == 'exact', tm_ex, tm_co)
    final = np.clip(apply_twins(pred_test, t_mate, t_level, alphas), 0, None)

    OUT_CSV.parent.mkdir(exist_ok=True)
    pd.DataFrame({'id': test['id'], TARGET: final}).to_csv(OUT_CSV, index=False)

    same_y = pd.DataFrame({'k': exact_tr, 'y': y}).groupby('k')['y'].transform(
        lambda v: v.duplicated(keep=False)).to_numpy()
    record = {
        'source': 'outputs_v2（10 折 × 种子，NNLS 融合）+ 训练集双胞胎后处理；无训练',
        'v2_single_model_oof_rmse': {k: round(v, 3) for k, v in single.items()},
        'v2_nnls_weights': weights,
        'v2_blend_oof_rmse': round(rmse(y, oof), 3),
        'alpha': {k: round(v, 3) for k, v in alphas.items()},
        'train_rows_by_level': {lv: int((level == lv).sum()) for lv in ['exact', 'core']},
        'exact_twins_identical_target_share': round(float(same_y[level == 'exact'].mean()), 3),
        'rmse_on_twin_rows_before_after': {
            lv: [round(rmse(y[level == lv], oof[level == lv]), 2),
                 round(rmse(y[level == lv], fixed[level == lv]), 2)] for lv in ['exact', 'core']},
        'oof_rmse_after_in_sample_alpha': round(rmse(y, fixed), 3),
        'oof_rmse_after_crossfit_alpha': round(rmse(y, crossfit), 3),
        'test_rows_by_level': {lv: int((t_level == lv).sum()) for lv in ['exact', 'core']},
        'test_mean_abs_change_on_twin_rows': round(float(
            np.abs(final - np.clip(pred_test, 0, None))[t_level != 'none'].mean()), 2),
        'output': str(OUT_CSV.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
