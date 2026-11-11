"""Expériences 16–17 : lissage par paires.

Premier passage (expérience 16, A_pair) : après la combinaison et la correction des lignes jumelles,
un modèle de paires estime ce que le résidu d'une ligne d'apprentissage j dit de celui d'une ligne i
(même sexe, pays et État ; années de naissance à un an près). Les résidus voisins sont agrégés avec ces poids.
Second passage (expérience 17, G / H) : la même chose sur les résidus de A_pair, pour les lignes
d'une même compétition, de mêmes sexe et équipement.

Le script relit les prédictions sauvegardées, ne réentraîne aucun modèle de base et tourne sur CPU
(environ 6 Go de mémoire).

Usage : python pair_smoothing.py <racine du projet> [dossier de sortie]
Variables d'environnement : NR (itérations du modèle de paires, 300 par défaut), BAGS (graines, 2 par défaut)
Sorties : summary_experiments.csv, puis pour chaque variante submission_v5_<variante>.csv et oof_v5_<variante>.npy.
Les agrégats du premier passage sont mis en cache dans pair_feats_A.npz.
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
MEET_CAP = 300                          # pas de paires dans les groupes de compétition plus grands (pas de signal)
EXPECTED_SHA256 = {}
_display = lambda *values, **kwargs: print(*values)      # utilisé par naming_compatibility hors notebook
CORE9 = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']

# ---- Cellules du projet réutilisées : lecture des données, jumelles et lissage par groupes
#      (elles définissent p1_cf, corrected_residuals, F, FT, HALF, alphas…)
exec((CELLS / 'naming_compatibility.py').read_text())
exec((CELLS / 'data_load.py').read_text())
NESTED = np.load(V4 / 'oof_nested_blend.npy')
BLEND_TEST = {}
exec((CELLS / 'stage2_twins_s3.py').read_text().replace('np.save(OUTPUT_DIR', 'pass  # np.save(OUTPUT_DIR'))
BASE_TEST = pd.read_csv(V4 / 'submission_final_v4_base.csv')
assert (BASE_TEST['id'].to_numpy() == test['id'].to_numpy()).all()
base_test = BASE_TEST[TARGET].to_numpy(dtype='float64')
r1 = corrected_residuals
apply_twins = apply_matching_correction
REF = rmse(y, p2_cf)

# ---- Attributs de chaque ligne ; une paire est un couple d'indices (i, j) : i quelconque, j dans l'apprentissage
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
    """Même sexe, pays et État (pays connu), années de naissance à un an près ; deux âges en .5 donnent une année exacte, qui doit être la même."""
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
    """Même compétition (fédération × année × pays × État) × sexe × équipement."""
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
    """resid : résidus des lignes d'apprentissage. Renvoie (agrégats apprentissage, agrégats test), 5 colonnes chacun."""
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
    target = (r_all[idx_i] * rj / float(np.var(resid))).astype('float32')    # 0 pour les lignes de test, qui ne servent pas à l'ajustement
    half_i = np.where(is_tr, HALF[np.minimum(idx_i, n_train - 1)], -1)
    half_j = HALF[idx_j]
    print(f'[{tag}] {n_pairs:,} paires candidates, dont {int(is_tr.sum()):,} entre lignes d\'apprentissage', flush=True)

    def fit_predict(fit, ev, p):
        booster = lgb.train(p, lgb.Dataset(X[fit], target[fit], feature_name=PF, free_raw_data=True), NR)
        return booster.predict(X[ev])

    s_hat = np.zeros(n_pairs)
    for b in range(BAGS):
        p = dict(PARAMS, seed=SEED + b)
        for h in (0, 1):                    # croisement : les deux lignes d'une paire d'ajustement sont hors de la moitié prédite
            ev = half_i == h
            s_hat[ev] += fit_predict((half_i == 1 - h) & (half_j == 1 - h), ev, p) / BAGS
        s_hat[~is_tr] += fit_predict(is_tr, ~is_tr, p) / BAGS
        print(f'[{tag}] modèle de paires {b + 1}/{BAGS} terminé', flush=True)
    del X
    sp = np.clip(s_hat, 0, None)
    sr, s, s2 = (np.bincount(idx_i, weights=w, minlength=N) for w in (sp * rj, sp, sp * sp))
    order = np.lexsort((sp, idx_i))                          # voisin de plus fort poids pour chaque i
    last = np.r_[idx_i[order][1:] != idx_i[order][:-1], True]
    top = np.zeros(N)
    top[idx_i[order][last]] = (sp * rj)[order][last]
    feats = np.column_stack([sr, sr / (1 + s), sr / (0.3 + s), top, sr * s2 / np.maximum(s, 1e-9) / (1 + s)])
    return feats[:n_train], feats[n_train:]


def solve(Fm, target, ridge=0.0):
    """Moindres carrés ; si ridge > 0, pénalité L2 de ridge × nombre de lignes sur les colonnes standardisées."""
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
rows = [{'variante': 'référence : huit modèles, lissage par groupes', 'oof_rmse': REF}]


def record(name, oof, test_pred, ref_oof):
    final = np.clip(test_pred, 0, None)
    pd.DataFrame({'id': test['id'], TARGET: final}).to_csv(OUTPUT_DIR / f'submission_v5_{name}.csv', index=False)
    np.save(OUTPUT_DIR / f'oof_v5_{name}.npy', oof)
    better = lambda m: rmse(y[m], oof[m]) < rmse(y[m], ref_oof[m])
    rows.append({'variante': name, 'oof_rmse': rmse(y, oof), 'ecart_v4': rmse(y, oof) - REF,
                 'ecart_etape_precedente': rmse(y, oof) - rmse(y, ref_oof),
                 'plis_ameliores': sum(better(FOLD_IDS == k) for k in range(1, 6)),
                 'moities_ameliorees': sum(better(HALF == h) for h in (0, 1)),
                 'ecart_abs_moyen_soumission_v4': float(np.abs(v4_sub - final).mean())})
    print(f'{name}: {rmse(y, oof):.3f}', flush=True)


# ---- Premier passage, A_pair : lissage par groupes + paires de sportifs approchés (cache relu s'il existe)
cache = OUTPUT_DIR / 'pair_feats_A.npz'
if cache.is_file() and int(np.load(cache)['nr']) == NR and int(np.load(cache)['bags']) == BAGS:
    A_TR, A_TE = np.load(cache)['train'], np.load(cache)['test']
    print('Cache relu :', cache.name)
else:
    A_TR, A_TE = pair_features(build_athlete, PF_ATHLETE, r1, 'sportifs')
    np.savez(cache, train=A_TR, test=A_TE, nr=NR, bags=BAGS)
FA, FTA = np.column_stack([F, A_TR]), np.column_stack([FT, A_TE])
oof_A = crossfit(p1_cf, r1, FA)
test_A = apply_twins(base_test, T_MATE, T_LEVEL, alphas) + FTA @ solve(FA, r1)
record('A_pair', oof_A, test_A, p2_cf)

# ---- Second passage : sur les résidus de A_pair, paires d'une même compétition × sexe × équipement
r3 = y - oof_A
M_TR, M_TE = pair_features(build_meet, PF_MEET, r3, 'compétitions')
for name, ridge in [('G_meet', 0.0), ('H_meet_ridge', 1e-2)]:
    record(name, crossfit(oof_A, r3, M_TR, ridge), test_A + M_TE @ solve(M_TR, r3, ridge), oof_A)

table = pd.DataFrame(rows).set_index('variante')
table.to_csv(OUTPUT_DIR / 'summary_experiments.csv')
pd.set_option('display.width', 250)
print(table.round(3).to_string())
print('Résultats écrits dans', OUTPUT_DIR)
