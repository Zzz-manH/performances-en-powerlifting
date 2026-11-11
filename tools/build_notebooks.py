"""Assembler les cinq notebooks du projet.

    python tools/build_notebooks.py <dossier_de_sortie>

Le code des expériences Colab 07–15 est conservé dans tools/cells/*.py.
Ce script ajoute configuration, explications, lecture des résultats et boutons de réentraînement.
Générer dans un dossier temporaire pour vérifier avant de remplacer les notebooks.
"""
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

TOOLS = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else TOOLS.parent


def cell(name):
    return (TOOLS / 'cells' / f'{name}.py').read_text(encoding='utf-8').strip('\n')


def cells(*names):
    return '\n\n\n'.join(cell(n) for n in names)


def md(text=''):
    """Cellule de texte. Le contenu vient de tools/textes/<notebook>.md, dans l'ordre des cellules (voir save)."""
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip('\n'))


def raw_code(text):
    return nbf.v4.new_code_cell(text.strip('\n'))


def gated(flag, body, setup='', env=True, off=None):
    """Exécuter le code seulement si le bouton est activé ; sinon lire les résultats sauvegardés."""
    lines = [f'if {flag}:']
    if env:
        lines.append('    ensure_training_env()')
    if setup:
        lines.append(textwrap.indent(textwrap.dedent(setup).strip('\n'), '    '))
    lines.append(textwrap.indent(body.strip('\n'), '    '))
    lines.append('else:')
    message = off or f'Entraînement désactivé ({flag} = False) : lecture des résultats dans artifacts/.'
    lines.append(f'    print("{message}")')
    return nbf.v4.new_code_cell('\n'.join(lines))


def save(cells_, name):
    # Les constantes sont disponibles avant TRAIN, dans une cellule technique repliable.
    compatibility = code(cell('naming_compatibility'))
    compatibility['metadata'] = {'jupyter': {'source_hidden': True}, 'tags': ['naming-compatibility']}
    first_code_index = next(i for i, c in enumerate(cells_) if c['cell_type'] == 'code')
    cells_.insert(first_code_index, compatibility)
    texts = (TOOLS / 'textes' / name.replace('.ipynb', '.md')).read_text(encoding='utf-8').split('<!-- cellule -->')
    markdown_cells = [c for c in cells_ if c['cell_type'] == 'markdown']
    if len(texts) != len(markdown_cells):
        raise ValueError(f'{name} : {len(markdown_cells)} cellules de texte attendues, {len(texts)} dans tools/textes')
    for c, text in zip(markdown_cells, texts):
        c['source'] = text.strip().replace('{NOTEBOOK_MAP}', NOTEBOOK_MAP.strip()).replace('{TRAIN_NOTE}', TRAIN_NOTE.strip())
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells_
    nb['metadata'] = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'colab': {'provenance': []},
    }
    nbf.validate(nb)
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / name)
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'{name}[{i}]', 'exec')
    print('wrote', OUT / name)


# ============================================================================= Cellules communes
SETUP = '''
# Environnement, chemins et configuration communs aux cinq notebooks
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as _display
from scipy.optimize import nnls
from sklearn.model_selection import KFold, train_test_split

try:
    from google.colab import drive
except ModuleNotFoundError:
    drive = None
if drive is not None:
    drive.mount('/content/drive')


def find_project_dir():
    candidates = []
    if os.environ.get('PL_PROJECT_DIR'):
        candidates.append(Path(os.environ['PL_PROJECT_DIR']))
    here = Path.cwd().resolve()
    candidates += [here, *here.parents]
    candidates.append(Path('/content/drive/MyDrive/Colab Notebooks/performances-en-powerlifting'))
    for candidate in candidates:
        if (candidate / 'data' / 'train.csv').is_file() and (candidate / 'data' / 'test.csv').is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError('Projet introuvable (data/train.csv et data/test.csv requis) ; définir PL_PROJECT_DIR')


PROJECT_DIR = find_project_dir()
DATA_DIR = PROJECT_DIR / 'data'                 # CSV sources, en lecture seule
ARTIFACTS = PROJECT_DIR / 'artifacts'           # Un sous-dossier de résultats par notebook
SEED = 42                                       # Cinq plis fixes : KFold(5, shuffle=True, random_state=42)
N_FOLDS = 5
INNER_STOP_FRACTION = 0.10                      # 10 % du train externe réservé à l’early stopping
MAX_ROUNDS = 20000
MODELS = [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_ABSOLUTE]             # Trois modèles pour les ablations, les plus rapides en premier
RUN_ONLY = None                                 # Liste de variantes à exécuter ; None = toutes
KEEP_DELTA = -0.05                              # kg : garder si ΔRMSE moyen ≤ −0.05 et amélioration sur ≥ 4/5 plis
EXPECTED_SHA256 = {'train.csv': '62319814116506fb95a51f1185dbacdb170ab055642ee4c347a30584093d0856',
                   'test.csv': '563c6fa29f9e535740ef1f09b25954f00fbe242c81c0e5a7343dc18aa962ba46'}
CATBOOST_VERSION = '1.2.10'
GPU_NAME = None
VERSIONS = {'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})

TRAIN = globals().get('TRAIN', {})              # Pas de bouton dans 01/05 ; 01 entraîne un petit HGB adversarial
if sum(bool(v) for v in TRAIN.values()) > 1:
    raise ValueError('Activer un seul entraînement à la fois ; redémarrer le runtime avant le suivant (variables partagées)')
TRAINING = any(TRAIN.values())


def ensure_training_env():
    """Préparer les dépendances et vérifier le GPU uniquement pour un entraînement."""
    global catboost, lgb, xgb, torch, RealMLP_TD_Regressor, TabM_D_Regressor, PYTABKIT_ERROR, GPU_NAME, VERSIONS
    to_install = []
    try:
        installed_catboost = importlib.metadata.version(MODEL_CATBOOST_ABSOLUTE)
    except importlib.metadata.PackageNotFoundError:
        installed_catboost = None
    if installed_catboost != CATBOOST_VERSION:
        if MODEL_CATBOOST_ABSOLUTE in sys.modules:
            raise RuntimeError('Une autre version de CatBoost est déjà importée ; redémarrer le runtime')
        to_install.append(f'catboost=={CATBOOST_VERSION}')
    try:
        importlib.metadata.version('pytabkit')
    except importlib.metadata.PackageNotFoundError:
        to_install.append('pytabkit')
    if to_install:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *to_install])
        importlib.invalidate_caches()
    import catboost
    import lightgbm as lgb
    import torch
    import xgboost as xgb
    try:
        from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor
        PYTABKIT_ERROR = None
    except Exception as exc:                    # Les autres modèles restent disponibles si cet import échoue
        RealMLP_TD_Regressor = TabM_D_Regressor = None
        PYTABKIT_ERROR = repr(exc)
    warnings.filterwarnings('ignore', category=UserWarning)
    if catboost.__version__ != CATBOOST_VERSION:
        raise RuntimeError(f'CatBoost {CATBOOST_VERSION} requis ; version actuelle : {catboost.__version__}')
    if shutil.which('nvidia-smi') is None or not torch.cuda.is_available():
        raise RuntimeError('Un GPU est requis : choisir T4 GPU dans le type de runtime Colab, puis relancer')
    GPU_NAME = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              text=True, capture_output=True, check=False).stdout.strip().splitlines()[0]
    VERSIONS = {MODEL_CATBOOST_ABSOLUTE: catboost.__version__, 'xgboost': xgb.__version__, 'lightgbm': lgb.__version__,
                'torch': torch.__version__,
                'pytabkit': None if PYTABKIT_ERROR else importlib.metadata.version('pytabkit'),
                'numpy': np.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0]}
    print('GPU         :', GPU_NAME)
    print('Versions    :', VERSIONS)
    if PYTABKIT_ERROR:
        print('pytabkit indisponible, RealMLP / TabM seront ignorés :', PYTABKIT_ERROR)


print('Projet :', PROJECT_DIR.name)
print(f'Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__}')
if TRAINING:
    print('Entraînements activés :', [k for k, v in TRAIN.items() if v])
'''

TRAIN_NOTE = '''
**Entraînement.** Tous les boutons de `TRAIN` sont à `False` : le notebook relit les résultats sauvegardés dans `artifacts/`, sur CPU, en quelques minutes. Pour refaire une expérience, mettre son bouton à `True` et choisir un runtime **T4 GPU**. Chaque modèle est sauvegardé dès qu'il est entraîné, donc une exécution interrompue reprend où elle s'était arrêtée. Un seul bouton à la fois, et redémarrer le runtime avant de passer au suivant.
'''

DATA = cell('data_load')

RESULT_HELPERS = '''
# Lecture des summary_*.csv et prédictions de plis ; aucun entraînement
def read_summary(path, index_col=0):
    path = Path(path)
    if not path.is_file():
        print(f'Fichier absent : {path.relative_to(PROJECT_DIR)} ; synchroniser artifacts/ depuis Drive ou activer l’entraînement associé')
        return None
    table = pd.read_csv(path, index_col=index_col)
    # Traduire les anciens libellés en mémoire, sans modifier les caches CSV.
    label_map = {
        LEGACY_SUMMARY_COMBINED_SMOOTHING: '+ lissage combiné des résidus (coefficients croisés)',
        '+ \\u53cc\\u80de\\u80ce\\uff08\\u4ea4\\u53c9\\u62df\\u5408\\uff09': '+ jumeaux (cross-fitting)',
        '+\\u53cc\\u80de\\u80ce': '+matched',
        '08 \\u4e94\\u6298\\u5806\\u53e0\\uff08global_nnls\\uff09': '08 stacking, cinq plis (global_nnls)',
        '3 \\u4e2a\\u5168\\u91cf / 95% \\u91cd\\u8bad': '3 refits',
        '5 \\u4e2a\\u5168\\u91cf / 95% \\u91cd\\u8bad': '5 refits',
        '8 \\u4e2a\\u5168\\u91cf / 95% \\u91cd\\u8bad': '8 refits',
        'v2\\uff0810 \\u6298 \\xd7 \\u79cd\\u5b50\\uff0cNNLS\\uff09': 'v2 (10 plis × graines, NNLS)',
        '\\u500d\\u6570': 'ratio',
        '\\u540c\\u4e00\\u56fd\\u5bb6': 'Même pays',
        '\\u540c\\u4e00\\u573a\\u6bd4\\u8d5b': 'Même compétition',
        '\\u540c\\u4e00\\u6bd4\\u8d5b\\u5e74\\u4efd': 'Même année de compétition',
        '\\u540c\\u4e00\\u8054\\u76df': 'Même fédération',
        '\\u6309 id \\u6392\\u5e8f': 'sorted_by_id',
        '\\u6765\\u6e90': 'source',
        '\\u76f8\\u90bb\\u4e24\\u884c': 'adjacent_rows',
        '\\u878d\\u5408\\uff08\\u5d4c\\u5957\\uff09': 'Ensemble (RMSE locale)',
        '\\u968f\\u673a\\u987a\\u5e8f': 'random_order',
    }
    table = table.rename(columns=label_map, index=label_map)
    for column in table.select_dtypes(include=['object', 'string']):
        table[column] = table[column].map(lambda value: label_map.get(value, value))
    return table


def show(table, columns=None, digits=3):
    if table is not None:
        shown = table if columns is None else table[[c for c in columns if c in table.columns]]
        display(shown.round(digits))


def plot_fold_deltas(table, baseline, title):
    """Comparaison appariée : barre = ΔRMSE moyen, points = plis, pointillé = seuil −0.05."""
    if table is None:
        return
    rows = table.drop(index=baseline, errors='ignore').sort_values('mean_delta', ascending=False)
    folds = rows[[f'd_fold{k}' for k in range(1, N_FOLDS + 1)]].to_numpy(dtype='float64')
    colours = ['#4c72b0' if v <= KEEP_DELTA else '#c44e52' if v > 0 else '#999999' for v in rows['mean_delta']]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.6))
    ax.barh(range(len(rows)), rows['mean_delta'], color=colours)
    for i in range(len(rows)):
        ax.scatter(folds[i], np.full(folds.shape[1], i), color='black', s=12, zorder=3)
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(KEEP_DELTA, color='#4c72b0', lw=0.8, ls='--')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([presentation_label(value) for value in rows.index.tolist()])
    ax.set_xlabel(f'Écart de RMSE par rapport à {presentation_label(baseline)} (kg) ; barres : moyenne, points : plis')
    ax.set_title(presentation_label(title))
    plt.tight_layout()
    plt.show()


def read_folds(directory, stem):
    """Charger OOF et test moyen des cinq plis ; retourner None si un pli manque."""
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for k in range(1, N_FOLDS + 1):
        path = Path(directory) / f'{stem}__fold{k}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    return (oof, test_pred) if not np.isnan(oof).any() else None
'''

FEATURES = cells('features_round1', 'features_round2', 'features_round3')

NOTEBOOK_MAP = '''
| Notebook | Contenu |
|---|---|
| **01_data_exploration** | lecture des données, exploration, plis |
| **02_feature_engineering** | quatre séries de variables et leurs ablations (exp. 07, 09, 10, 14) |
| **03_model_selection** | modèles, cibles, hyperparamètres, façons de combiner (exp. 08, 13) |
| **04_final_model** | ensemble final, corrections, lissage par paires, soumissions (exp. 11, 12, 15–17) |
| **05_error_analysis** | résidus, essais abandonnés, score local et score public |

Les CSV sont dans `data/`, les résultats dans `artifacts/`. Les numéros d'expérience (07 à 17) sont ceux de notre journal de travail.
'''

# ============================================================================= 01
OVERVIEW = '''
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
require(sample_submission.columns.tolist() == ['id', TARGET] and len(sample_submission) == len(test)
        and set(sample_submission['id']) == set(test['id']), 'Le modèle de soumission ne correspond pas au test')
both_raw = pd.concat([train[RAW_FEATURES], test[RAW_FEATURES]], ignore_index=True)
overview = pd.DataFrame({
    'Type': ['Numérique' if c in RAW_NUM else 'Catégorielle' for c in RAW_FEATURES],
    'Valeurs distinctes (train+test)': [int(both_raw[c].nunique()) for c in RAW_FEATURES],
    'Valeur la plus fréquente': [str(both_raw[c].value_counts().index[0]) for c in RAW_FEATURES],
    'Fréquence du mode': [float(both_raw[c].value_counts(normalize=True).iloc[0]) for c in RAW_FEATURES],
    'Proportion Unknown': [float((both_raw[c].astype(str) == 'Unknown').mean()) for c in RAW_FEATURES],
}, index=RAW_FEATURES)
display(overview.round(3))
display(train[RAW_NUM + [TARGET]].describe().T.round(2))
print('Modèle de soumission :', sample_submission.shape, ', colonnes', sample_submission.columns.tolist())
'''

TARGET_PLOTS = '''
sex = train['Sex'].to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].hist(y, bins=80, color='#4c72b0')
axes[0].set_title('TotalKg, all training rows')
axes[0].set_xlabel('kg')
for value, colour in [('M', '#4c72b0'), ('F', '#dd8452')]:
    axes[1].hist(y[sex == value], bins=70, alpha=0.55, density=True, color=colour, label=value)
axes[1].legend()
axes[1].set_title('TotalKg by sex')
axes[1].set_xlabel('kg')
shown_rows = np.random.default_rng(SEED).choice(n_train, 20000, replace=False)
axes[2].scatter(train['BodyweightKg'].to_numpy()[shown_rows], y[shown_rows], s=2, alpha=0.25,
                c=np.where(sex[shown_rows] == 'M', '#4c72b0', '#dd8452'))
axes[2].set_xlabel('BodyweightKg')
axes[2].set_ylabel('TotalKg')
axes[2].set_title('TotalKg vs bodyweight (20,000 random rows; blue = M)')
plt.tight_layout()
plt.show()
display(train.groupby(['Sex', 'Equipment'])[TARGET].agg(['size', 'mean', 'std']).round(1))
'''

GROUP_PLOTS = '''
def mean_by(values, order=None):
    table = pd.DataFrame({'k': np.asarray(values).astype(str), 'y': y}).groupby('k')['y'].agg(['size', 'mean'])
    return table if order is None else table.reindex([o for o in order if o in table.index])


age_cut = pd.cut(train['Age'], [0, 18, 24, 35, 40, 50, 60, np.inf], right=False)
year_cut = pd.cut(train['MeetYear'], [-np.inf, 2000, 2010, 2015, 2020, np.inf], right=False)
largest_feds = train['Federation'].value_counts().head(12).index.tolist()
panels = [('mean TotalKg by age', mean_by(age_cut.astype(str), [str(c) for c in age_cut.cat.categories])),
          ('mean TotalKg by meet year', mean_by(year_cut.astype(str), [str(c) for c in year_cut.cat.categories])),
          ('mean TotalKg, 12 largest federations', mean_by(train['Federation'], largest_feds))]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (title, table) in zip(axes, panels):
    ax.bar(range(len(table)), table['mean'], color='#4c72b0')
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table.index.tolist(), rotation=60, ha='right', fontsize=8)
    ax.axhline(y.mean(), color='black', ls='--', lw=1)
    ax.set_title(presentation_label(title))
plt.tight_layout()
plt.show()
'''

PRECISION = '''
def safe_mean(values):
    return float(np.mean(values)) if len(values) else np.nan


age_half = ((train['Age'] * 2).round() % 2).astype(int).to_numpy()      # 1 = âge en x.5 (année de naissance seule connue)
bw = train['BodyweightKg']
is_int = np.isclose(bw, bw.round())
is_1dec = np.isclose(bw * 10, (bw * 10).round())
is_half = is_1dec & ~is_int & np.isclose((bw * 10).round() % 10, 5)
bw_precision = np.select([is_int, is_half, is_1dec], ['Entier', 'x.5', 'Autre décimale'], default='Deux décimales')
precision_rows = []
for fed in ['(ensemble)'] + train['Federation'].value_counts().head(8).index.tolist():
    m = np.ones(n_train, dtype=bool) if fed == '(ensemble)' else (train['Federation'] == fed).to_numpy()
    precision_rows.append({'Fédération': fed, 'effectif': int(m.sum()), 'Part des âges x.5': age_half[m].mean(),
                           'Total moyen : âge entier': safe_mean(y[m & (age_half == 0)]),
                           'Total moyen : âge x.5': safe_mean(y[m & (age_half == 1)]),
                           'Part des poids à deux décimales': float((bw_precision[m] == 'Deux décimales').mean())})
display(pd.DataFrame(precision_rows).set_index('Fédération').round(3))
display(pd.DataFrame({'effectif': pd.Series(bw_precision).value_counts(),
                      'Total moyen': pd.Series(y).groupby(bw_precision).mean()}).round(1))
'''

TWINS = '''
def matching_record_summary():
    core = ['Age', 'BodyweightKg', 'MeetYear', 'Sex', 'Federation', 'MeetCountry', 'MeetState', 'Country', 'State']
    exact_tr = pd.util.hash_pandas_object(train[RAW_FEATURES], index=False).to_numpy()
    core_tr = pd.util.hash_pandas_object(train[core], index=False).to_numpy()
    exact_te = pd.util.hash_pandas_object(test[RAW_FEATURES], index=False).to_numpy()
    core_te = pd.util.hash_pandas_object(test[core], index=False).to_numpy()
    n_exact = pd.Series(exact_tr).map(pd.Series(exact_tr).value_counts()).to_numpy() - 1    # Sans la ligne elle-même
    n_core = pd.Series(core_tr).map(pd.Series(core_tr).value_counts()).to_numpy() - 1
    level = np.select([n_exact > 0, n_core > 0], ['exact', 'core'], default='none')
    same_total = pd.DataFrame({'k': exact_tr, 'y': y}).duplicated(keep=False).to_numpy()  # Même clé et même résultat sur une autre ligne
    test_exact = np.isin(exact_te, exact_tr)
    test_core = ~test_exact & np.isin(core_te, core_tr)
    display(pd.DataFrame([
        ('Train : jumeau exact sur les 16 variables', int((level == 'exact').sum())),
        ('Train : jumeau core sur les 9 variables seulement', int((level == 'core').sum())),
        ('Part des jumeaux exacts ayant le même TotalKg', f"{same_total[level == 'exact'].mean():.1%}"),
        ('Test : jumeau exact présent dans le train', int(test_exact.sum())),
        ('Test : jumeau core seulement dans le train', int(test_core.sum())),
    ], columns=['Élément', 'Nombre']).set_index('Élément'))
    example_keys = pd.Series(exact_tr[level == 'exact']).drop_duplicates().head(3).to_numpy()
    example = train[np.isin(exact_tr, example_keys)].assign(key=exact_tr[np.isin(exact_tr, example_keys)])
    display(example.sort_values(['key', TARGET]).drop(columns='key').head(8))


matching_record_summary()
'''

QUALITY = '''
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) Validation adversariale : distinguer train et test
encoded = both_raw.copy()
for c in CAT:
    encoded[c] = encoded[c].astype('category').cat.codes
is_test = np.r_[np.zeros(n_train), np.ones(len(test))]
adversarial_auc = cross_val_score(HistGradientBoostingClassifier(max_iter=200, random_state=SEED), encoded, is_test,
                                  cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring='roc_auc').mean()

# 2) Vérifier si les id voisins partagent plus souvent la même compétition
ids_all = pd.concat([train['id'], test['id']], ignore_index=True).to_numpy()
order = np.argsort(ids_all)
perm = np.random.default_rng(0).permutation(len(ids_all))
meet_key = (both_raw['Federation'] + '|' + both_raw['MeetYear'].astype(str) + '|' + both_raw['MeetCountry']
            + '|' + both_raw['MeetState']).to_numpy()
id_rows = []
for name, values in {'Même compétition': meet_key, 'Même fédération': both_raw['Federation'].to_numpy(),
                     'Même année de compétition': both_raw['MeetYear'].to_numpy(), 'Même pays': both_raw['Country'].to_numpy()}.items():
    by_id = float((values[order][1:] == values[order][:-1]).mean())
    shuffled = float((values[perm][1:] == values[perm][:-1]).mean())
    id_rows.append({'Lignes voisines': name, 'Tri par id': by_id, 'Ordre aléatoire': shuffled, 'Rapport': by_id / max(shuffled, 1e-12)})
display(pd.DataFrame(id_rows).set_index('Lignes voisines').round(4))
display(pd.DataFrame([
    ('AUC train/test (0.5 : peu de décalage détectable)', f'{adversarial_auc:.4f}'),
    ('Corrélation Spearman id / TotalKg', f"{float(spearmanr(train['id'], y)[0]):+.4f}"),
    ('Corrélation Spearman id / année (train+test)', f"{float(spearmanr(ids_all, both_raw['MeetYear'])[0]):+.4f}"),
    ('Étendue des id : train / test', f"{train['id'].min()}–{train['id'].max()} / {test['id'].min()}–{test['id'].max()}"),
], columns=['Contrôle', 'Résultat']).set_index('Contrôle'))

# 3) Résultats suspects selon les cibles du train ; essai de suppression dans 05
ratio = y / train['BodyweightKg'].to_numpy(dtype='float64')
suspicious = (np.where(train['Sex'].to_numpy() == 'M', ratio < 2.5, ratio < 1.8)
              | (train['Age'].to_numpy(dtype='float64') < 5))
print(f'Lignes suspectes (total < 2.5 × poids chez les hommes, < 1.8 chez les femmes, ou âge < 5) : {int(suspicious.sum())} ({suspicious.mean():.2%})')
display(train.loc[suspicious, ['Age', 'BodyweightKg', 'Sex', 'Equipment', 'Federation', 'MeetYear', TARGET]].head(8))
'''

FOLDS_TABLE = '''
display(pd.DataFrame({'Lignes de validation': np.bincount(FOLD_IDS)[1:],
                      'Moyenne TotalKg': [y[FOLD_IDS == k].mean() for k in range(1, N_FOLDS + 1)],
                      'Écart-type TotalKg': [y[FOLD_IDS == k].std() for k in range(1, N_FOLDS + 1)]},
                     index=pd.Index(range(1, N_FOLDS + 1), name='fold')).round(2))
'''

nb01 = [
    md(''),
    code(SETUP),
    md(''),
    raw_code(DATA),
    md(''),
    code(OVERVIEW),
    md(''),
    code(TARGET_PLOTS),
    md(''),
    code(GROUP_PLOTS),
    md(''),
    code(PRECISION),
    md(''),
    md(''),
    code(TWINS),
    md(''),
    code(QUALITY),
    md(''),
    md(''),
    code(FOLDS_TABLE),
]

# ============================================================================= 02
CONTEXT_ABLATION_CONFIG = '''
# Précision, déplacements et contexte de compétition ; ancien dossier : artifacts/feature_round1_v1
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / CACHE_CONTEXT_ABLATION
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_PARAM_CHECK = True                 # CatBoost profondeur 8, régularisation renforcée / profondeur 6, régularisation renforcée sur les meilleures variables
'''

HISTORY_ABLATION_CONFIG = '''
# Historique approché du sportif ; ancien dossier : artifacts/feature_round2_v1
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / CACHE_HISTORY_ABLATION
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
'''

CAREER_ABLATION_CONFIG = '''
# Carrière du sportif et ajouts propres aux modèles ; ancien dossier : artifacts/feature_round3_v1
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / CACHE_CAREER_ABLATION
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
'''

FINAL_SETTINGS = '''
SEEDS = [42, 7, 2024]                      # Chaque graine change les plis, l’early stopping et l’aléa du modèle
FINAL_MODELS = [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_LIGHTGBM_DOTS, MODEL_CATBOOST_ABSOLUTE, MODEL_CATBOOST_DOTS, 'tabm']   # Modèles rapides en premier dans chaque pli
FULL_REFIT = True                          # Refit des arbres sur tout le train
REFIT_FACTOR = 1.2                         # Itérations de refit = moyenne CV × 1.2 (environ 39 % de données en plus)
'''

PERFORMANCE_ABLATION_CONFIG = '''
# Statistiques de performance ; référence : variables finales par modèle ; ancien dossier : artifacts/feature_round4_v1
OUTPUT_DIR = ARTIFACTS / '02_feature_engineering' / 'round4'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
''' + FINAL_SETTINGS + '''
SCREEN_MODELS = ['lgb', 'xgb', 'cb_rel']        # Trois modèles de sélection, dont cb_rel
TARGET_MODELS = ['cb_allo', 'cb_gl']            # Deux formules de force relative
INNER_FEATURE_FOLDS = 5                         # Plis internes pour les variables issues des résultats
'''

SHOW_CONTEXT_ABLATION = '''
def show_round1():
    directory = ARTIFACTS / '02_feature_engineering' / CACHE_CONTEXT_ABLATION
    single = read_summary(directory / 'summary_single.csv', index_col=[0, 1])
    if single is not None:
        print('RMSE OOF par modèle, cinq plis (kg) :')
        display(single['oof_rmse'].unstack(0).reindex([VARIANT_WEIGHT_CLASS_BASELINE, BLOCK_TRAVEL, BLOCK_RECORDING_PRECISION, VARIANT_PRECISION_AND_TRAVEL, BLOCK_MEET_CONTEXT, VARIANT_CONTEXT_COMBINED]).round(3))
    blend = read_summary(directory / 'summary_blend.csv')
    print('Ensemble NNLS à trois modèles, critère principal :')
    show(blend, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cb'])
    plot_fold_deltas(blend, VARIANT_WEIGHT_CLASS_BASELINE, 'Round 1: three-model blend vs F0')
    print('Jumeaux : alpha ajusté par cross-fitting en deux groupes core :')
    show(read_summary(directory / 'summary_twins.csv'))


show_round1()
'''

SHOW_HISTORY_ABLATION = '''
def show_round2():
    directory = ARTIFACTS / '02_feature_engineering' / CACHE_HISTORY_ABLATION
    singles = {m: read_summary(directory / f'summary_single_{m}.csv') for m in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_ABSOLUTE]}
    singles = {m: t for m, t in singles.items() if t is not None}
    if singles:
        print('ΔRMSE moyen par rapport à la référence avec contexte (kg ; négatif = mieux ; identifiants approchés pour CatBoost seul) :')
        display(pd.DataFrame({m: t['mean_delta'] for m, t in singles.items()})
                .reindex([BLOCK_EXTENDED_MEET_CONTEXT, BLOCK_DIVISION_CONTEXT, BLOCK_ATHLETE_HISTORY, BLOCK_DOTS_SCALE, BLOCK_PROXY_IDENTIFIERS]).round(3))
    screen = read_summary(directory / 'summary_blend_screen.csv')
    print('Ensemble NNLS à trois modèles :')
    show(screen, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cb'])
    plot_fold_deltas(screen, VARIANT_CONTEXT_BASELINE, 'Historique du sportif : combinaison de trois modèles comparée à la référence avec contexte')
    print('Blocs retenus dans ALL2 :')
    show(read_summary(directory / 'summary_blend_all2.csv'), ['oof_rmse', 'mean_delta', 'folds_better', 'keep'])


show_round2()
'''

SHOW_CAREER_ABLATION = '''
def show_round3():
    directory = ARTIFACTS / '02_feature_engineering' / CACHE_CAREER_ABLATION
    singles = {m: read_summary(directory / f'summary_single_{m}.csv') for m in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_ABSOLUTE]}
    singles = {m: t for m, t in singles.items() if t is not None}
    if singles:
        print('ΔRMSE moyen par rapport à la référence avec historique (kg ; les ajouts propres au modèle ne changent pas LightGBM) :')
        display(pd.DataFrame({m: t['mean_delta'] for m, t in singles.items()}).reindex([BLOCK_ATHLETE_CAREER, BLOCK_MEET_EXPERIENCE, BLOCK_MODEL_SPECIFIC]).round(3))
    screen = read_summary(directory / 'summary_blend_screen.csv')
    print('Ensemble NNLS à trois modèles :')
    show(screen, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cb'])
    plot_fold_deltas(screen, VARIANT_HISTORY_BASELINE, 'Carrière du sportif : combinaison de trois modèles comparée à la référence avec historique')
    print('Les trois blocs satisfont le seuil : ALL3')
    show(read_summary(directory / 'summary_blend_all3.csv'), ['oof_rmse', 'mean_delta', 'folds_better', 'keep'])


show_round3()
'''

SHOW_PERFORMANCE_ABLATION = '''
def show_round4():
    directory = ARTIFACTS / '02_feature_engineering' / CACHE_PERFORMANCE_ABLATION
    screen = read_summary(directory / 'summary_blend_screen.csv')
    print('Ensemble NNLS : LightGBM, XGBoost, cb_rel :')
    show(screen, ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'weights_lgb_xgb_cbrel'])
    plot_fold_deltas(screen, VARIANT_FINAL_BASELINE, 'Statistiques de performance : combinaison de trois modèles comparée aux variables finales')
    net = read_summary(directory / 'summary_net_after_stage2.csv')
    if net is not None:
        print('Après correction des correspondances et lissage combiné des résidus ; scores publics :')
        display(net.assign(Public=pd.Series({VARIANT_FINAL_BASELINE: 76.592, BLOCK_ATHLETE_PERFORMANCE: 76.802})).round(3))


show_round4()
'''

nb02 = [
    md(''),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    CACHE_CONTEXT_ABLATION: False,   # Exp. 07 : 6 variantes × 3 modèles × 5 plis, CatBoost et correspondances d’enregistrements (~1.5–2 h GPU)
    CACHE_HISTORY_ABLATION: False,   # Exp. 09 : deuxième série (~1.5–2 h GPU)
    CACHE_CAREER_ABLATION: False,   # Exp. 10 : troisième série et lissage (~1–1.5 h GPU)
    CACHE_PERFORMANCE_ABLATION: False,   # Exp. 14 : variables avec cible, id et cibles alternatives (~1–1.5 h GPU)
}
'''),
    code(SETUP),
    md(''),
    raw_code(DATA),
    md(''),
    raw_code(cell('features_round1')),
    md(''),
    raw_code(cell('features_round2')),
    md(''),
    raw_code(cell('features_round3')),
    md(''),
    md(''),
    raw_code(RESULT_HELPERS),
    md(''),
    gated('TRAIN[CACHE_CONTEXT_ABLATION]', cells('r1_models', 'r1_run', 'r1_compare', 'r1_residuals', 'r1_catboost_params',
                                   'r1_twins'), setup=CONTEXT_ABLATION_CONFIG),
    code(SHOW_CONTEXT_ABLATION),
    md(''),
    md(''),
    gated('TRAIN[CACHE_HISTORY_ABLATION]', cells('r2_models', 'r2_run', 'r2_compare', 'r2_all2_twins', 'r2_diagnostics'),
          setup=HISTORY_ABLATION_CONFIG),
    code(SHOW_HISTORY_ABLATION),
    md(''),
    md(''),
    gated('TRAIN[CACHE_CAREER_ABLATION]', cells('r3_models', 'r3_run', 'r3_compare', 'r3_all3_twins', 'r3_stage2_variants',
                                   'r3_diagnostics'), setup=CAREER_ABLATION_CONFIG),
    code(SHOW_CAREER_ABLATION),
    md(''),
    md(''),
    gated('TRAIN[CACHE_PERFORMANCE_ABLATION]', cells('final_frames', 'r4_id_check', 'r4_label_features', 'final_model_defs',
                                   'r4_models', 'r4_run', 'r4_compare', 'r4_stage2_net', 'r4_targets'),
          setup=PERFORMANCE_ABLATION_CONFIG),
    code(SHOW_PERFORMANCE_ABLATION),
    md(''),
    md(''),
]

# ============================================================================= 03
ZOO_CONFIG = '''
# Comparaison des modèles et cibles relatives ; ancien dossier : artifacts/model_zoo_v1_ALL
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / 'model_zoo'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_SET = VARIANT_CONTEXT_COMBINED        # Meilleure variante de l’expérience 07 : Variables initiales et catégorie de poids + Précision des enregistrements + Déplacements + Contexte de compétition
ZOO = [MODEL_LIGHTGBM_DOTS, MODEL_CATBOOST_DOTS, 'realmlp', 'tabm']   # Modèles rapides en premier dans chaque pli
RUN_ONLY = None            # Exemple : ['realmlp'] ; None = tous
STAGE2_MIN_GAIN = 0.02     # Gain minimal du lissage en cross-fitting pour la soumission (kg)
CONTEXT_ABLATION_DIR = ARTIFACTS / '02_feature_engineering' / CACHE_CONTEXT_ABLATION      # Résultats de l’expérience 07
TEN_FOLD_ENSEMBLE_DIR = ARTIFACTS / '03_model_selection' / CACHE_TEN_FOLD_ENSEMBLE     # Prédictions ensemble à dix plis et plusieurs graines par pli (ancien outputs_v2)
'''

COMPARE_CONFIG = '''
# Comparaison élargie des familles de modèles ; ancien dossier : artifacts/model_compare_v1
OUTPUT_DIR = ARTIFACTS / '03_model_selection' / 'model_compare'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
''' + FINAL_SETTINGS + '''
PREV_DIR = ARTIFACTS / '04_final_model' / 'v1_six_models'     # Résultats de l’expérience 11
BOOST_DIR = ARTIFACTS / '04_final_model' / 'v2_more_seeds'    # Résultats de l’expérience 12
PREV_SEEDS = [42, 7, 2024]
COMPARE = ['ridge', 'hgb', 'et', 'rf', 'tabpfn', 'mlp_plr', 'resnet', 'ebm']   # Les plus rapides en premier
OPTIONAL = []                         # Options : 'ftt' (lent), 'tabr' (faiss requis)
TABPFN_CONTEXT_ROWS = 10000           # Contexte TabPFN v2 limité à 10 000 lignes
ADD_DELTA = -0.02                     # Seuil pour l’ensemble : ΔRMSE ≤ −0.02 sur ≥ 4/5 plis
TABPFN_TOKEN = os.environ.get('TABPFN_TOKEN')
if not TABPFN_TOKEN and drive is not None:
    try:
        from google.colab import userdata
        TABPFN_TOKEN = userdata.get('TABPFN_TOKEN')
    except Exception:
        TABPFN_TOKEN = None
if TABPFN_TOKEN:
    os.environ['TABPFN_TOKEN'] = TABPFN_TOKEN
print('TabPFN：', 'Version récente (TABPFN_TOKEN fourni)' if TABPFN_TOKEN else 'v2 (sans connexion)')
if not (PREV_DIR / 'summary_weights.csv').is_file():
    raise FileNotFoundError(f'Résultats de l’expérience 11 absents : {PREV_DIR}')
'''

BASELINES = '''
baselines = pd.DataFrame([
    ('sklearn GradientBoosting (encodage ordinal)', '5 plis, moyenne', 82.60, '~80'),
    ('Ensemble LightGBM / CatBoost (CPU local)', '5 plis, early stopping externe', 77.766, '—'),
    ('v1 : XGBoost + CatBoost + MLP, NNLS (GPU Colab)', '5 plis, early stopping externe', 77.775, 78.0),
    ('v2 : LightGBM ×1 + XGBoost ×3 + CatBoost ×3 + MLP ×3 graines, NNLS', '10 plis, early stopping externe', 77.302, 77.799),
    ('v2 par modèle : LightGBM / XGBoost / CatBoost / MLP', 'Idem', '78.189 / 78.432 / 78.093 / 78.395', '—'),
], columns=['Modèle', 'Protocole', 'RMSE locale', 'Public'])
display(baselines.set_index('Modèle'))
'''

SHOW_CB_PARAMS = '''
print('CatBoost sur ALL, cinq plis appariés (C1 = paramètres v2) :')
show(read_summary(ARTIFACTS / '02_feature_engineering' / CACHE_CONTEXT_ABLATION / 'summary_catboost_params.csv'),
     ['oof_rmse', 'mean_delta', 'folds_better', 'keep', 'd_fold1', 'd_fold2', 'd_fold3', 'd_fold4', 'd_fold5'])
'''

SHOW_ZOO = '''
def show_zoo():
    directory = ARTIFACTS / '03_model_selection' / 'model_zoo'
    models = read_summary(directory / 'summary_models.csv')
    print('RMSE individuelle et gain dans l’ensemble (préfixe 07_ : expérience 07) :')
    show(models, ['oof_rmse', 'corr_with_07_blend', 'blend_with_07', 'delta_vs_07_blend', 'folds_better'])
    if models is not None:
        ordered = models.sort_values('oof_rmse', ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(range(len(ordered)), ordered['oof_rmse'],
                color=['#999999' if n.startswith('07_') else '#4c72b0' for n in ordered.index])
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered.index.tolist())
        ax.set_xlim(ordered['oof_rmse'].min() - 0.5, ordered['oof_rmse'].max() + 0.2)
        ax.set_xlabel('5-fold OOF RMSE (kg), features ALL; grey = experiment 07 models')
        plt.tight_layout()
        plt.show()
    print('Trois méthodes de stacking évaluées sur cinq plis :')
    show(read_summary(directory / 'summary_stacking.csv'))


show_zoo()
'''

SHOW_TARGETS = '''
print('Cibles alternatives : variables finales, graine 42, cinq plis ; gain dans LightGBM, XGBoost et CatBoost sur cible DOTS')
show(read_summary(ARTIFACTS / '02_feature_engineering' / CACHE_PERFORMANCE_ABLATION / 'summary_target_engineering.csv'))
'''

SHOW_COMPARE = '''
def show_compare():
    compare = read_summary(ARTIFACTS / '03_model_selection' / 'model_compare' / 'summary_compare.csv')
    show(compare, ['source', 'oof_seed42', 'corr_with_blend', 'blend_nested', 'delta_vs_blend', 'folds_better',
                   'weight_in_blend'])
    if compare is not None:
        ordered = compare.sort_values('oof_seed42', ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(range(len(ordered)), ordered['oof_seed42'],
                color=['#4c72b0' if s == 11 or str(s) == '11' else '#dd8452' for s in ordered['source']])
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered.index.tolist())
        ax.set_xlim(76, ordered['oof_seed42'].max() + 0.5)
        ax.set_xlabel('seed-42 5-fold OOF RMSE (kg); blue = final models, orange = compared in experiment 13')
        plt.tight_layout()
        plt.show()


show_compare()
'''

SHOW_LR = '''
lr_check = read_summary(ARTIFACTS / '04_final_model' / CACHE_EXPANDED_SEED_ENSEMBLE / 'summary_lr_check.csv')
show(lr_check)
if lr_check is not None:
    print(f"Learning rate 0.03 vs 0.05 : {lr_check['delta'].mean():+.3f} kg, {int((lr_check['delta'] < 0).sum())}/5 plis améliorés ; "
          f"ratio des meilleures itérations : {lr_check['best_iter_0.03'].mean() / lr_check['best_iter_0.05'].mean():.2f}")
'''

nb03 = [
    md(''),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    'model_zoo': False,       # Exp. 08 : RealMLP, TabM, CatBoost sur cible DOTS, LightGBM sur cible DOTS, stacking et lissage (~1–1.5 h GPU)
    'model_compare': False,   # Exp. 13 : comparaison et soumission avec HGB (~1.5 h GPU)
}
'''),
    md(''),
    gated("TRAIN['model_compare']", cell('compare_install'), env=False),
    code(SETUP),
    md(''),
    raw_code(DATA),
    gated('TRAINING', FEATURES, env=False, off='Entraînement désactivé : lecture des résultats, sans recalcul des variables.'),
    raw_code(RESULT_HELPERS),
    md(''),
    code(BASELINES),
    md(''),
    code(SHOW_CB_PARAMS),
    md(''),
    md(''),
    gated("TRAIN['model_zoo']", cells('zoo_models', 'zoo_run', 'zoo_compare', 'zoo_stage2'), setup=ZOO_CONFIG),
    code(SHOW_ZOO),
    md(''),
    md(''),
    code(SHOW_TARGETS),
    md(''),
    md(''),
    gated("TRAIN['model_compare']", cells('final_frames', 'final_model_defs', 'compare_models', 'compare_run',
                                          'compare_results', 'compare_blend', 'compare_stage2'),
          setup=COMPARE_CONFIG),
    code(SHOW_COMPARE),
    md(''),
    md(''),
    code(SHOW_LR),
    md(''),
]

# ============================================================================= 04
FINAL_SHARED = FINAL_SETTINGS + '\n' + cells('final_frames', 'final_model_defs')

SIX_MODEL_ENSEMBLE_CONFIG = '''
# Ensemble à six candidats et trois graines ; ancien dossier : artifacts/final_v1
OUTPUT_DIR = ARTIFACTS / '04_final_model' / CACHE_SIX_MODEL_ENSEMBLE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_CLEANING_TEST = True                   # Essai de nettoyage présenté dans 05
'''

EXPANDED_SEED_ENSEMBLE_CONFIG = '''
# Graines de réentraînement supplémentaires ; ancien dossier : artifacts/final_v2
OUTPUT_DIR = ARTIFACTS / '04_final_model' / CACHE_EXPANDED_SEED_ENSEMBLE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR = ARTIFACTS / '04_final_model' / CACHE_SIX_MODEL_ENSEMBLE     # Résultats de l’expérience 11
PREV_SEEDS = [42, 7, 2024]
EXTRA_SEEDS = [11, 23, 57, 99, 123]                          # Graines supplémentaires pour le refit complet
EXTRA_MODELS = [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_LIGHTGBM_DOTS, MODEL_CATBOOST_DOTS]           # Les plus rapides en premier
TABM_FULL_SEEDS = [42, 7, 2024]
TABM_HOLDOUT = 0.05                                          # TabM : 5 % réservés à l’early stopping
RUN_LR_CHECK = True
if not (PREV_DIR / 'summary_weights.csv').is_file() or not (PREV_DIR / 'oof_nested_blend.npy').is_file():
    raise FileNotFoundError(f'Résultats de l’expérience 11 absents : {PREV_DIR}')
'''

FINAL_ENSEMBLE_CONFIG = '''
# Ensemble final à huit candidats ; ancien dossier : artifacts/final_v3
OUTPUT_DIR = ARTIFACTS / '04_final_model' / CACHE_FINAL_ENSEMBLE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SIX_MODEL_ENSEMBLE_DIR = ARTIFACTS / '04_final_model' / CACHE_SIX_MODEL_ENSEMBLE          # Expérience 11
EXPANDED_SEED_ENSEMBLE_DIR = ARTIFACTS / '04_final_model' / CACHE_EXPANDED_SEED_ENSEMBLE          # Expérience 12
COMPARE_DIR = ARTIFACTS / '03_model_selection' / 'model_compare'   # Expérience 13
PERFORMANCE_ABLATION_DIR = ARTIFACTS / '02_feature_engineering' / CACHE_PERFORMANCE_ABLATION       # Expérience 14
PREV_SEEDS = [42, 7, 2024]
GL_FULL_SEEDS = [42, 7, 2024, 11, 23]
HGB_FULL_SEEDS = [42, 7, 2024]
TABM_EXTRA_SEEDS = [11, 23]
TABM_HOLDOUT = 0.05
for directory in (SIX_MODEL_ENSEMBLE_DIR, EXPANDED_SEED_ENSEMBLE_DIR):
    if not directory.is_dir():
        raise FileNotFoundError(f'Dossier de résultats précédent absent : {directory}')
'''

SHOW_SIX_MODEL_ENSEMBLE = '''
def show_six_model_ensemble():
    directory = ARTIFACTS / '04_final_model' / CACHE_SIX_MODEL_ENSEMBLE
    print('RMSE OOF par graine et moyenne des trois graines :')
    show(read_summary(directory / 'summary_models.csv'))
    weights = read_summary(directory / 'summary_weights.csv')
    if weights is not None:
        print('Poids NNLS :', weights['weight'].round(3).to_dict())
    print('Post-traitement (cross-fitting des coefficients) :')
    show(read_summary(directory / 'summary_stage2.csv'))
    display(pd.DataFrame([('Moyenne des 15 modèles de plis', 76.520),
                          ('Moitié refit complet, moitié moyenne des plis', 76.421),
                          ('Arbres : refit sur 105 000 lignes × 3 graines ; TabM : moyenne des plis', 76.360)],
                         columns=['Prédiction test avec correspondances et lissage des résidus', 'Public']).set_index('Prédiction test avec correspondances et lissage des résidus'))


show_six_model_ensemble()
'''

SHOW_EXPANDED_SEED_ENSEMBLE = '''
def show_expanded_seed_ensemble():
    six_model_directory = ARTIFACTS / '04_final_model' / CACHE_SIX_MODEL_ENSEMBLE
    expanded_seed_directory = ARTIFACTS / '04_final_model' / CACHE_EXPANDED_SEED_ENSEMBLE
    counts = {m: len(list(six_model_directory.glob(f'{m}__s*__full.npz'))) + len(list(expanded_seed_directory.glob(f'{m}__s*__full.npz')))
              for m in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_LIGHTGBM_DOTS, MODEL_CATBOOST_ABSOLUTE, MODEL_CATBOOST_DOTS]}
    counts['tabm'] = len(list(expanded_seed_directory.glob('tabm__s*__nearfull.npz')))
    print('Nombre de refits par modèle (TabM : 95 % des données) :', counts)
    display(pd.DataFrame([('v2_seeds : 8 graines de refit, TabM en moyenne des plis', 76.359),
                          ('v2_seeds_tabmfull : ajout de TabM à 95 % × 3 graines', 76.337)],
                         columns=['Soumission (poids et post-traitement de l’expérience 11)', 'Public']).set_index('Soumission (poids et post-traitement de l’expérience 11)'))


show_expanded_seed_ensemble()
'''

SHOW_HISTOGRAM_BOOSTING_ENSEMBLE = '''
print('Exp. 13 : exp. 12 + HGB (moyenne des cinq plis), ensemble et post-traitement :')
show(read_summary(ARTIFACTS / '03_model_selection' / 'model_compare' / 'summary_stage2.csv'))
'''

SHOW_FINAL_ENSEMBLE = '''
def show_final_ensemble():
    """Recalculer l’ensemble depuis les prédictions sauvegardées et vérifier la soumission, sans entraînement."""
    six_model_directory = ARTIFACTS / '04_final_model' / CACHE_SIX_MODEL_ENSEMBLE
    expanded_seed_directory = ARTIFACTS / '04_final_model' / CACHE_EXPANDED_SEED_ENSEMBLE
    final_ensemble_source = ARTIFACTS / '04_final_model' / CACHE_FINAL_ENSEMBLE
    six = [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_LIGHTGBM_DOTS, MODEL_CATBOOST_ABSOLUTE, MODEL_CATBOOST_DOTS, 'tabm']
    seeds = [42, 7, 2024]

    def refit_files(model):
        if model in (MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_LIGHTGBM_DOTS, MODEL_CATBOOST_ABSOLUTE, MODEL_CATBOOST_DOTS):
            return sorted(six_model_directory.glob(f'{model}__s*__full.npz')) + sorted(expanded_seed_directory.glob(f'{model}__s*__full.npz'))
        if model == 'tabm':
            return sorted(expanded_seed_directory.glob('tabm__s*__nearfull.npz')) + sorted(final_ensemble_source.glob('tabm__s*__nearfull.npz'))
        if model == MODEL_HISTOGRAM_GRADIENT_BOOSTING:
            return sorted(final_ensemble_source.glob('hgb__s*__nearfull.npz'))
        return sorted(final_ensemble_source.glob(f'{model}__s*__full.npz'))

    per, rows = {}, []
    for model in six + [MODEL_HISTOGRAM_GRADIENT_BOOSTING, MODEL_CATBOOST_IPF_GL]:
        loaded = {s: read_folds(six_model_directory if model in six else final_ensemble_source, f'{model}__s{s}') for s in seeds}
        loaded = {s: v for s, v in loaded.items() if v is not None}
        if not loaded:
            print(f'{model} : résultats des cinq plis absents, ignoré')
            continue
        oof = np.mean([o for o, _ in loaded.values()], axis=0)
        files = refit_files(model)
        full = (np.mean([np.load(p)['test'] for p in files], axis=0) if files
                else np.mean([t for _, t in loaded.values()], axis=0))
        per[model] = {'oof': oof, 'full': full}
        rows.append({'model': model, **{f'Graine {s}': rmse(y, o) for s, (o, _) in loaded.items()},
                     'OOF moyen, trois graines': rmse(y, oof),
                     'Prédiction test': f'{len(files)} refits (100 % arbres ; 95 % TabM ; 90 % HGB)' if files else 'Moyenne des plis'})
    if not per:
        return
    names = list(per)
    weights, _ = nnls(np.column_stack([per[n]['oof'] for n in names]), y)
    table = pd.DataFrame(rows).set_index('model')
    table['Poids NNLS'] = pd.Series(dict(zip(names, weights)))
    display(table.round(3))

    def nested(columns):
        P_sub = np.column_stack([per[n]['oof'] for n in columns])
        out = np.zeros(n_train)
        for k in range(1, N_FOLDS + 1):
            fit, va = FOLD_IDS != k, FOLD_IDS == k
            w, _ = nnls(P_sub[fit], y[fit])
            out[va] = P_sub[va] @ w
        return out

    steps = {'Exp. 11 : six modèles': [m for m in names if m in six],
             '+ HGB (exp. 13)': [m for m in names if m in six + [MODEL_HISTOGRAM_GRADIENT_BOOSTING]],
             '+ HGB + cb_gl (final, exp. 15)': names}
    display(pd.DataFrame([{'Ensemble': k, 'Nombre de modèles': len(v), 'RMSE locale': rmse(y, nested(v))}
                          for k, v in steps.items() if v]).set_index('Ensemble').round(3))
    print('Post-traitement (cross-fitting des coefficients) :')
    show(read_summary(final_ensemble_source / 'summary_stage2.csv'))

    blend_test = np.clip(np.column_stack([per[n]['full'] for n in names]) @ weights, 0, None)
    saved_path = final_ensemble_source / SUBMISSION_FINAL_BASE_FILENAME
    if saved_path.is_file() and len(names) == 8:
        saved = pd.read_csv(saved_path)
        require(saved['id'].equals(test['id']), f'Ordre des id différent de test.csv dans {saved_path.name}')
        print(f'Écart maximal entre le recalcul et {saved_path.name} : '
              f'{np.abs(blend_test - saved[TARGET].to_numpy()).max():.2e} kg')


show_final_ensemble()
'''

PAIR_RUN = '''
# Expériences 16–17 ; résultats dans artifacts/04_final_model/v5_pair_smoothing
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm'])
process = subprocess.Popen([sys.executable, '-W', 'ignore', str(PROJECT_DIR / 'tools' / 'pair_smoothing.py'),
                            str(PROJECT_DIR)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    print(line, end='')
require(process.wait() == 0, 'tools/pair_smoothing.py a échoué')
'''

SHOW_PAIR_SMOOTHING = '''
def show_pair_smoothing():
    # Recalculer le score après chaque correction à partir des prédictions sauvegardées, sans entraînement.
    eight = ARTIFACTS / '04_final_model' / CACHE_FINAL_ENSEMBLE
    pairs = ARTIFACTS / '04_final_model' / 'v5_pair_smoothing'
    steps = [('Combinaison des huit modèles', eight / 'oof_nested_blend.npy', np.nan),
             ('+ jumelles et lissage par groupes (exp. 15)', eight / 'oof_final_crossfit.npy', 76.298),
             ('+ paires de sportifs (exp. 16)', pairs / 'oof_v5_A_pair.npy', 76.089),
             ('+ paires dans la compétition (exp. 17, finale)', pairs / 'oof_v5_H_meet_ridge.npy', 76.030)]
    if not all(path.is_file() for _, path, _ in steps):
        print('Résultats du lissage par paires absents : récupérer artifacts/ ou activer pair_smoothing')
        return
    rows, previous = [], None
    for name, path, public in steps:
        oof = np.load(path)
        row = {'Étape': name, 'RMSE locale': rmse(y, oof), 'RMSE publique': public}
        if previous is not None:
            row['Écart avec l’étape précédente'] = rmse(y, oof) - rmse(y, previous)
            row['Plis améliorés'] = sum(rmse(y[FOLD_IDS == k], oof[FOLD_IDS == k]) < rmse(y[FOLD_IDS == k], previous[FOLD_IDS == k])
                                        for k in range(1, N_FOLDS + 1))
        rows.append(row)
        previous = oof
    table = pd.DataFrame(rows).set_index('Étape')
    display(table.round(3))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(table))
    ax.plot(x, table['RMSE locale'], marker='o', label='RMSE locale')
    ax.plot(x, table['RMSE publique'], marker='s', linestyle='--', label='RMSE publique')
    for i, (local, public) in enumerate(zip(table['RMSE locale'], table['RMSE publique'])):
        ax.annotate(f'{local:.3f}', (i, local), xytext=(0, -14), textcoords='offset points', ha='center')
        if np.isfinite(public):
            ax.annotate(f'{public:.3f}', (i, public), xytext=(0, 7), textcoords='offset points', ha='center')
    ax.set_xticks(x, ['Combinaison', '+ jumelles, groupes', '+ paires de sportifs', '+ paires par compétition'])
    ax.set_ylabel('RMSE (kg)')
    ax.legend()
    plt.show()


show_pair_smoothing()
'''

FINAL_FILE_CHECK = '''
final_file = ARTIFACTS / '04_final_model' / 'v5_pair_smoothing' / 'submission_v5_H_meet_ridge.csv'
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
if final_file.is_file():
    final_sub = pd.read_csv(final_file)
    require(final_sub.columns.tolist() == ['id', TARGET], 'Colonnes attendues : id, TotalKg')
    require(len(final_sub) == len(sample_submission) == 45000 and final_sub['id'].is_unique
            and set(final_sub['id']) == set(sample_submission['id']), 'Les id diffèrent de sample_submission')
    require(np.isfinite(final_sub[TARGET]).all() and (final_sub[TARGET] > 0).all(), 'Valeurs de prédiction invalides')
    print(f'Contrôles OK : {final_file.relative_to(PROJECT_DIR)}, {len(final_sub)} lignes ; moyenne {final_sub[TARGET].mean():.1f}, '
          f'étendue {final_sub[TARGET].min():.1f}–{final_sub[TARGET].max():.1f} (moyenne train {y.mean():.1f})')
else:
    print(f'Fichier absent : {final_file.relative_to(PROJECT_DIR)}')

leaderboard = pd.DataFrame([
    ('11', '6 modèles × 3 graines, moyenne des modèles de plis, jumelles et lissage', 76.044, 76.520),
    ('11', 'Les mêmes, arbres réentraînés sur toutes les lignes', np.nan, 76.360),
    ('12', 'Réentraînement avec 8 graines', np.nan, 76.359),
    ('12', '+ TabM sur 95 % × 3 graines', np.nan, 76.337),
    ('13', '+ HGB', 76.010, 76.334),
    ('15', '8 modèles (CatBoost IPF GL et HGB sur 3 graines, TabM sur 5)', 75.966, 76.298),
    ('16', '+ lissage par paires de sportifs', 75.607, 76.089),
    ('17', '+ lissage par paires dans la compétition (finale)', 75.575, 76.030),
], columns=['Expérience', 'Soumission', 'RMSE locale', 'RMSE publique'])
display(leaderboard.set_index('Expérience'))
'''

nb04 = [
    md(''),
    md(TRAIN_NOTE),
    code('''
TRAIN = {
    'six_models': False,     # Exp. 11 : 6 modèles × 5 plis × 3 graines, refit et nettoyage (~2.5 h GPU)
    'more_seeds': False,     # Exp. 12 : learning rate CatBoost sur cible DOTS, 5 graines de plus, TabM à 95 % (~1 h GPU)
    'eight_models': False,   # Exp. 15 : CatBoost IPF GL et HGB à 3 graines, réentraînement et soumission (~1 h GPU)
    'pair_smoothing': False, # Exp. 16–17 : lissage par paires (CPU, beaucoup de RAM ; demande les résultats de l'exp. 15)
}
'''),
    code(SETUP),
    md(''),
    raw_code(DATA + "\nMODEL_TRAINING = any(v for k, v in TRAIN.items() if k != 'pair_smoothing')   # le lissage par paires relit seulement des prédictions"),
    gated('MODEL_TRAINING', FEATURES, env=False, off='Pas d’entraînement de modèles : lecture des résultats, sans recalcul des variables.'),
    gated('MODEL_TRAINING', FINAL_SHARED, off='Pas d’entraînement de modèles : définitions inutiles.'),
    raw_code(RESULT_HELPERS),
    md(''),
    gated("TRAIN['six_models']", cells('v1_run', 'v1_blend', 'stage2_twins_s3', 'v1_cleaning_test'), setup=SIX_MODEL_ENSEMBLE_CONFIG),
    code(SHOW_SIX_MODEL_ENSEMBLE),
    md(''),
    md(''),
    gated("TRAIN['more_seeds']", cells('v2_setup', 'v2_lr_check', 'v2_refit', 'v2_assemble', 'stage2_twins_s3'),
          setup=EXPANDED_SEED_ENSEMBLE_CONFIG),
    code(SHOW_EXPANDED_SEED_ENSEMBLE),
    md(''),
    code(SHOW_HISTOGRAM_BOOSTING_ENSEMBLE),
    md(''),
    gated("TRAIN['eight_models']", cells('v4_setup', 'v4_cv', 'v4_refit', 'v4_assemble', 'stage2_twins_s3',
                                         'v4_final_check'), setup=FINAL_ENSEMBLE_CONFIG),
    code(SHOW_FINAL_ENSEMBLE),
    md(''),
    md(''),
    gated("TRAIN['pair_smoothing']", PAIR_RUN, env=False),
    code(SHOW_PAIR_SMOOTHING),
    md(''),
    md(''),
    code(FINAL_FILE_CHECK),
    md(''),
]

# ============================================================================= 05
RESIDUALS = '''
def athlete_key(frame, with_federation=True, extra=()):
    birth = np.floor(frame['MeetYear'] - frame['Age'] + 1e-9).astype(int).astype(str)
    key = frame['Sex'] + '|' + frame['Country'] + '|' + frame['State']
    if with_federation:
        key = key + '|' + frame['Federation']
    key = key + '|' + birth
    for column in extra:
        key = key + '|' + frame[column].astype(str)
    return key.reset_index(drop=True)


def final_residuals():
    final_ensemble_source = ARTIFACTS / '04_final_model' / CACHE_FINAL_ENSEMBLE
    before = np.load(final_ensemble_source / 'oof_nested_blend.npy')        # Ensemble NNLS des huit modèles
    after = np.load(final_ensemble_source / 'oof_final_crossfit.npy')       # Puis correspondances d’enregistrements + Lissage combiné des résidus (coefficients cross-fittés)
    require(len(before) == n_train and len(after) == n_train, 'Longueur OOF incorrecte')
    ensemble_residuals, corrected_residuals = y - before, y - after
    print(f'RMSE locale : ensemble {rmse(y, before):.3f} → post-traitement {rmse(y, after):.3f}')
    keys_all = pd.concat([athlete_key(train), athlete_key(test)], ignore_index=True)
    appearances = keys_all.map(keys_all.value_counts()).iloc[:n_train].to_numpy()
    appearances = np.where(train['Country'].to_numpy() == 'Unknown', 0, appearances)
    age_half = ((train['Age'] * 2).round() % 2).astype(int).to_numpy()
    largest = train['Federation'].value_counts().head(8).index
    groups = {
        'Sexe × équipement': (train['Sex'] + '|' + train['Equipment']).to_numpy(),
        'Contrôle antidopage': train['Tested'].to_numpy(),
        'Tranche d’âge': pd.cut(train['Age'], [0, 18, 24, 35, 40, 50, 60, np.inf], right=False).astype(str).to_numpy(),
        'Tranche de poids': pd.cut(train['BodyweightKg'], [0, 60, 75, 90, 105, 120, 140, np.inf], right=False).astype(str).to_numpy(),
        'Année de compétition': pd.cut(train['MeetYear'], [-np.inf, 2000, 2010, 2015, 2020, np.inf], right=False).astype(str).to_numpy(),
        'Fédération (8 principales)': np.where(train['Federation'].isin(largest), train['Federation'], 'Autre'),
        'Âge en x.5': np.where(age_half == 1, 'x.5', 'Entier'),
        'Apparitions du même athlète (train+test)': pd.cut(appearances, [-1, 0, 1, 2, 5, np.inf],
                                                labels=['Nationalité inconnue', '1', '2', '3-5', '6+']).astype(str),
    }
    parts = []
    for name, key in groups.items():
        frame = pd.DataFrame({'k': np.asarray(key).astype(str), 'r0': ensemble_residuals, 'r1': corrected_residuals})
        part = frame.groupby('k').agg(effectif=('r1', 'size'), residu_moyen_ensemble=('r0', 'mean'), residu_moyen_final=('r1', 'mean'),
                                      RMSE_finale=('r1', lambda v: float(np.sqrt(np.mean(np.square(v))))))
        part.index = pd.MultiIndex.from_product([[name], part.index], names=['Groupe', 'Valeur'])
        parts.append(part)
    display(pd.concat(parts).round(2))

    big = np.abs(corrected_residuals) > 150
    print(f'|résidu| > 150 kg : {big.mean():.1%} des lignes, {np.sum(corrected_residuals[big] ** 2) / np.sum(corrected_residuals ** 2):.0%} de l’erreur quadratique')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(np.clip(corrected_residuals, -400, 400), bins=120, color='#4c72b0')
    axes[0].set_title('Final OOF residual (true - predicted), clipped at +-400 kg')
    deciles = pd.qcut(after, 10, labels=False, duplicates='drop')
    binned = pd.DataFrame({'d': deciles, 'r': corrected_residuals}).groupby('d')['r'].agg(
        ['mean', lambda v: float(np.sqrt(np.mean(np.square(v))))])
    binned.columns = ['mean residual', 'RMSE']
    binned.plot.bar(ax=axes[1], color=['#999999', '#dd8452'])
    axes[1].set_xlabel('decile of the prediction')
    axes[1].set_title('Residual mean and RMSE by prediction decile')
    plt.tight_layout()
    plt.show()


final_residuals()
'''

ATHLETE_CORR = '''
def athlete_residual_correlation():
    """Corrélation avec le résidu moyen des autres lignes de même clé (leave-one-out)."""
    final_ensemble_source = ARTIFACTS / '04_final_model' / CACHE_FINAL_ENSEMBLE
    before = y - np.load(final_ensemble_source / 'oof_nested_blend.npy')
    after = y - np.load(final_ensemble_source / 'oof_final_crossfit.npy')
    known = (train['Country'] != 'Unknown').to_numpy()
    keys = {
        'A0 sexe + nationalité + État + naissance': (athlete_key(train, with_federation=False), known),
        'A1 + fédération': (athlete_key(train), known),
        'A2 + équipement': (athlete_key(train, extra=['Equipment']), known),
        'Compétition × sexe × équipement': ((train['Federation'] + '|' + train['MeetYear'].astype(str) + '|' + train['MeetCountry']
                             + '|' + train['MeetState'] + '|' + train['Sex'] + '|' + train['Equipment'])
                            .reset_index(drop=True), np.ones(n_train, dtype=bool)),
    }

    def loo_corr(residual, key, valid):
        frame = pd.DataFrame({'k': key.to_numpy(), 'r': residual})
        total = frame.groupby('k')['r'].transform('sum').to_numpy()
        count = frame.groupby('k')['r'].transform('size').to_numpy()
        mask = valid & (count > 1)
        others = (total - residual) / np.maximum(count - 1, 1)
        return float(np.corrcoef(residual[mask], others[mask])[0, 1]), int(mask.sum())

    rows = []
    for name, (key, valid) in keys.items():
        c0, n0 = loo_corr(before, key, valid)
        c1, _ = loo_corr(after, key, valid)
        rows.append({'Clé': name, 'Lignes avec une autre ligne de même clé': n0, 'Corrélation après ensemble': c0, 'Corrélation après post-traitement': c1})
    display(pd.DataFrame(rows).set_index('Clé').round(3))


athlete_residual_correlation()
'''

STAGE2_HISTORY = '''
def stage2_history():
    round1 = ARTIFACTS / '02_feature_engineering' / CACHE_CONTEXT_ABLATION
    zoo = ARTIFACTS / '03_model_selection' / 'model_zoo'
    round3 = ARTIFACTS / '02_feature_engineering' / CACHE_CAREER_ABLATION
    final = ARTIFACTS / '04_final_model' / CACHE_FINAL_ENSEMBLE
    print('1) Jumeaux (exp. 07, ensemble à trois modèles) :')
    show(read_summary(round1 / 'summary_twins.csv'))
    print('2) Lissage individuel S0 (exp. 08) sur le stacking et v2 :')
    show(read_summary(zoo / 'summary_stage2.csv'),
         ['oof_rmse', 'twins_crossfit', 'twins_athlete_crossfit', 'athlete_gain', 'beta_A0', 'beta_A1', 'beta_A2',
          'beta_A3', 'beta_M2'])
    print('3) Cinq lissages (exp. 10, ALL3 + jumeaux) :')
    show(read_summary(round3 / 'summary_stage2_variants.csv'))
    print('4) Version finale (exp. 15) :')
    show(read_summary(final / 'summary_stage2.csv'))


stage2_history()
'''

CLEANING = '''
def cleaning_test():
    six_model_directory = ARTIFACTS / '04_final_model' / CACHE_SIX_MODEL_ENSEMBLE
    ratio = y / train['BodyweightKg'].to_numpy(dtype='float64')
    suspicious = (np.where(train['Sex'].to_numpy() == 'M', ratio < 2.5, ratio < 1.8)
                  | (train['Age'].to_numpy(dtype='float64') < 5))
    print(f'Lignes suspectes : {int(suspicious.sum())} ({suspicious.mean():.2%})')
    rows = []
    for model in [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_CATBOOST_ABSOLUTE]:
        base_all, clean_all, base_normal, clean_normal = [], [], [], []
        for k in range(1, N_FOLDS + 1):
            base_path = six_model_directory / f'{model}__s{SEED}__fold{k}.npz'
            clean_path = six_model_directory / f'clean_{model}__s{SEED}__fold{k}.npz'
            if not (base_path.is_file() and clean_path.is_file() and clean_path.with_suffix('.json').is_file()):
                continue                                   # Comme exp. 11 : npz et json requis pour un résultat complet
            base, clean = np.load(base_path), np.load(clean_path)
            idx = base['valid_idx']
            require(np.array_equal(idx, clean['valid_idx']), 'Lignes de validation différentes entre nettoyage et référence')
            normal = ~suspicious[idx]
            base_all.append(rmse(y[idx], base['val']))
            clean_all.append(rmse(y[idx], clean['val']))
            base_normal.append(rmse(y[idx][normal], base['val'][normal]))
            clean_normal.append(rmse(y[idx][normal], clean['val'][normal]))
        if base_all:
            delta = np.array(clean_all) - np.array(base_all)
            rows.append({'Modèle': model, 'Plis terminés': len(base_all), 'Référence : toute la validation': np.mean(base_all),
                         'Train nettoyé : toute la validation': np.mean(clean_all), 'Δ': delta.mean(),
                         'Plis améliorés': int((delta < 0).sum()), 'Référence : lignes non suspectes': np.mean(base_normal),
                         'Train nettoyé : lignes non suspectes': np.mean(clean_normal)})
    if rows:
        display(pd.DataFrame(rows).set_index('Modèle').round(3))


cleaning_test()
'''

LEADERBOARD = '''
def leaderboard_analysis():
    history = pd.DataFrame([
        (ENSEMBLE_INITIAL, 77.775, 78.0), (ENSEMBLE_TEN_FOLD, 77.302, 77.799), (HISTORY_TEN_FOLD_MATCHING, 77.060, 77.604),
        (HISTORY_WEIGHT_CLASS_MATCHING, 77.701, 77.947), (HISTORY_CONTEXT_REGULARIZED, 77.210, 77.337), (HISTORY_CONTEXT_REGULARIZED_MATCHING, 76.966, 77.158),
        (HISTORY_MODEL_STACKING, 77.013, 77.207), (HISTORY_STACKING_MATCHING, 76.757, 77.018), (HISTORY_STACKING_RESIDUAL_SMOOTHING, 76.521, 76.731),
        (HISTORY_TEN_FOLD_RESIDUAL_SMOOTHING, 76.774, 77.125), (HISTORY_FINAL_FEATURES_MATCHING, 76.649, 76.884),
        (HISTORY_EXACT_BIRTHYEAR_SMOOTHING, 76.539, 76.775), (HISTORY_COMBINED_RESIDUAL_SMOOTHING, 76.510, 76.716), (HISTORY_SIX_MODEL_FOLD_AVERAGE, 76.044, 76.520),
        (HISTORY_HISTOGRAM_BOOSTING_ENSEMBLE, 76.010, 76.334), (HISTORY_FINAL_FEATURES_REFERENCE, 76.361, 76.592), (HISTORY_ATHLETE_PERFORMANCE, 76.508, 76.802),
        (HISTORY_FINAL_ENSEMBLE, 75.966, 76.298),
    ], columns=['submission', 'cv', 'public'])
    history['public - cv'] = history['public'] - history['cv']
    display(history.rename(columns={'cv': 'local', 'public - cv': 'public - local'}).set_index('submission').round(3))
    early = history['submission'].isin([ENSEMBLE_INITIAL, ENSEMBLE_TEN_FOLD, HISTORY_TEN_FOLD_MATCHING, HISTORY_TEN_FOLD_RESIDUAL_SMOOTHING])   # Early stopping externe
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(history.loc[~early, 'cv'], history.loc[~early, 'public'], color='#4c72b0', label='Arrêt anticipé sur validation interne')
    ax.scatter(history.loc[early, 'cv'], history.loc[early, 'public'], color='#c44e52', label='Arrêt anticipé sur le pli externe : premiers ensembles')
    for _, row in history.iterrows():
        ax.annotate(presentation_label(row['submission']), (row['cv'], row['public']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    lims = [history[['cv', 'public']].min().min() - 0.1, history[['cv', 'public']].max().max() + 0.1]
    ax.plot(lims, lims, color='black', lw=0.8, ls='--', label='public = local')
    ax.set_xlabel('RMSE locale (kg)')
    ax.set_ylabel('RMSE publique (kg)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    print(f"Corrélation de Spearman local/public avec arrêt anticipé interne : "
          f"{history.loc[~early, ['cv', 'public']].corr(method='spearman').iloc[0, 1]:.2f}")


def public_noise():
    """Tirer 13 500 lignes OOF et comparer l’ensemble final à l’ensemble avec HistGradientBoosting, comme approximation du public."""
    histogram_ensemble_predictions = np.load(ARTIFACTS / '03_model_selection' / 'model_compare' / 'oof_final_crossfit.npy')
    final_ensemble_source = np.load(ARTIFACTS / '04_final_model' / CACHE_FINAL_ENSEMBLE / 'oof_final_crossfit.npy')
    rng = np.random.default_rng(SEED)
    single, diff = [], []
    for _ in range(500):
        idx = rng.choice(n_train, 13500, replace=False)
        a, b = rmse(y[idx], histogram_ensemble_predictions[idx]), rmse(y[idx], final_ensemble_source[idx])
        single.append(b)
        diff.append(b - a)
    diff = np.array(diff)
    public_gap = 76.298 - 76.334                       # Scores publics archivés : ensemble final à huit candidats 76.298, ensemble avec HistGradientBoosting 76.334
    print(f'105 000 lignes : ensemble final {rmse(y, final_ensemble_source):.3f}, ensemble avec HistGradientBoosting {rmse(y, histogram_ensemble_predictions):.3f}, écart {rmse(y, final_ensemble_source) - rmse(y, histogram_ensemble_predictions):+.3f}')
    print(f'Tirages de 13 500 lignes : écart-type RMSE {np.std(single):.3f} ; écart-type de la différence {np.std(diff):.3f} ; '
          f'ordre inversé dans {np.mean(diff > 0):.0%} des tirages')
    print(f'Écart public observé entre ensemble final et ensemble avec HistGradientBoosting : {public_gap:+.3f}')
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(diff, bins=40, color='#4c72b0')
    ax.axvline(rmse(y, final_ensemble_source) - rmse(y, histogram_ensemble_predictions), color='black', ls='--', label='difference on all rows')
    ax.axvline(public_gap, color='#dd8452', lw=1.5, label='actual public leaderboard')
    ax.axvline(0, color='#c44e52', lw=1)
    ax.set_xlabel('Écart de RMSE : ensemble final − ensemble avec HistGradientBoosting, sur 13 500 lignes aléatoires (kg)')
    ax.set_title('Variabilité du score public sur 30 % des lignes')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


leaderboard_analysis()
public_noise()
'''

nb05 = [
    md(''),
    code(SETUP),
    md(''),
    raw_code(DATA),
    raw_code(RESULT_HELPERS),
    md(''),
    code(RESIDUALS),
    md(''),
    md(''),
    code(ATHLETE_CORR),
    md(''),
    code(STAGE2_HISTORY),
    md(''),
    md(''),
    code(CLEANING),
    md(''),
    md(''),
    md(''),
    code(LEADERBOARD),
    md(''),
    md(''),
]

save(nb01, '01_data_exploration.ipynb')
save(nb02, '02_feature_engineering.ipynb')
save(nb03, '03_model_selection.ipynb')
save(nb04, '04_final_model.ipynb')
save(nb05, '05_error_analysis.ipynb')
