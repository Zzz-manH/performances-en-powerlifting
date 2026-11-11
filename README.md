# Performances en powerlifting : prédire `TotalKg`

Projet du Data Challenge d'apprentissage supervisé. Il s'agit de prédire le total des trois mouvements (`TotalKg`, en kg) d'un athlète à partir des informations de sa participation à une compétition. La métrique est la RMSE. Le jeu contient 105 000 lignes d'apprentissage, 45 000 de test et 16 variables (3 numériques, 13 catégorielles).

**Résultat.** Soumission finale : [submissions/20261110_v6_meet_smoothing.csv](submissions/20261110_v6_meet_smoothing.csv), RMSE locale **75.575**, score public **76.030**. Notre première soumission était à 78.0.

Le détail des expériences est dans [RESULTS.md](RESULTS.md) et l'historique des soumissions dans [submissions/README.md](submissions/README.md).

## Les cinq notebooks

| Notebook | Contenu |
|---|---|
| [01_data_exploration](01_data_exploration.ipynb) | lecture des données, exploration, précision de saisie, lignes jumelles, plis |
| [02_feature_engineering](02_feature_engineering.ipynb) | quatre séries de variables et leurs ablations |
| [03_model_selection](03_model_selection.ipynb) | premiers essais, réglage de CatBoost, autres modèles, cibles de force relative |
| [04_final_model](04_final_model.ipynb) | huit modèles × trois graines, combinaison NNLS, corrections, lissage par paires, soumissions |
| [05_error_analysis](05_error_analysis.ipynb) | résidus, essais abandonnés, score local et score public |

Les notebooks sont enregistrés avec leurs sorties : on peut lire les tableaux et les figures sans rien exécuter.

## La solution en bref

| Étape | Ce qui est fait |
|---|---|
| Variables | les 16 d'origine, la catégorie de poids, la précision de saisie, les déplacements, le contexte de la compétition, l'historique et la carrière du sportif ; 57, 73 ou 78 colonnes selon le modèle |
| Modèles | LightGBM, XGBoost, CatBoost, TabM, HistGradientBoosting ; LightGBM et CatBoost aussi sur des cibles de force relative (DOTS, IPF GL) |
| Validation | 5 plis × 3 graines (42, 7, 2024), 10 % de la partie d'apprentissage réservés à l'early stopping |
| Test | modèles réentraînés sur toutes les lignes, plusieurs graines |
| Combinaison | moindres carrés non négatifs (NNLS) : 76.319 |
| Corrections | lignes jumelles, lissage des résidus par groupes (75.966), puis lissage par paires (75.575) |

Les corrections utilisent les autres résultats connus d'un même sportif. Leurs scores locaux sont obtenus en ajustant les coefficients sur une moitié des données et en les évaluant sur l'autre ; ce n'est pas une validation de toute la chaîne (voir le rapport).

## Données

Les données ne sont pas dans le dépôt. Télécharger `train.csv`, `test.csv` et `sample_submission.csv` depuis la [page Kaggle de la compétition](https://www.kaggle.com/competitions/performances-en-powerlifting/data) et les placer dans `data/`. Les notebooks vérifient l'empreinte des fichiers au démarrage.

Nous n'utilisons aucune donnée externe : la base OpenPowerlifting contient les lignes du test.

## Exécution

- **Lire seulement** : ouvrir les notebooks, les sorties sont déjà là.
- **Réexécuter sans entraîner** : par défaut tous les boutons `TRAIN` sont à `False` et les notebooks relisent les prédictions sauvegardées dans `artifacts/`. Ce dossier (environ 470 Mo) n'est pas dans le dépôt ; sans lui, il faut réentraîner.
- **Réentraîner** : sur Colab avec un GPU T4, placer le dossier du projet dans `MyDrive/Colab Notebooks/` (ou définir `PL_PROJECT_DIR`), mettre un bouton à `True` et exécuter. Un seul bouton à la fois, en redémarrant le runtime entre deux.
- **Tout refaire depuis zéro** : dans 04, `six_models` (environ 2 h 30), `more_seeds` (1 h), `eight_models` (1 h), puis `pair_smoothing` (CPU). Les notebooks 02 et 03 ne sont nécessaires que pour refaire les ablations.

Dépendances : [requirements.txt](requirements.txt). CatBoost 1.2.10 et pytabkit sont installés par les notebooks quand un entraînement est lancé.

## Organisation

```text
01–05 *.ipynb      les cinq notebooks
data/              CSV de la compétition (à télécharger)
artifacts/         prédictions et résultats sauvegardés (hors dépôt)
submissions/       fichiers soumis et historique des scores
tools/             code des cellules, textes et générateur des notebooks, lissage par paires
RESULTS.md         résultats des expériences
```

Les numéros d'expérience (07 à 17) qui apparaissent dans le code et les tableaux sont ceux de notre journal de travail. Les premiers notebooks de travail, avant réorganisation, sont dans le premier commit du dépôt.
