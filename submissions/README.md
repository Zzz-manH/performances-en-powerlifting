# Soumissions

Le score public porte sur environ 30 % du test (13 500 lignes). Nous choisissons les versions sur le score local ; le public sert à confirmer.

## Fichiers gardés

| Fichier | Contenu | RMSE locale | Score public |
|---|---|---:|---:|
| **`20261110_v6_meet_smoothing.csv`** | **soumission finale** : huit modèles, jumelles, lissage par groupes, lissage par paires (deux passages) | **75.575** | **76.030** |
| `20261110_v5_pair_smoothing.csv` | la même sans le second passage | 75.607 | 76.089 |
| `20261017_FINAL_v4_eight_models.csv` | huit modèles, jumelles, lissage par groupes | 75.966 | 76.298 |
| `20261017_v3_plus.csv` | six modèles + HGB | 76.010 | 76.334 |
| `20261017_v2_seeds_tabmfull.csv` | six modèles, 8 graines de réentraînement, TabM sur 95 % | — | 76.337 |

Ce sont des copies de fichiers produits dans `artifacts/` (`04_final_model/v5_pair_smoothing/`, `04_final_model/v4_eight_models/`, `03_model_selection/model_compare/`, `04_final_model/v2_more_seeds/`).

## Historique

| Date | Soumission | Exp. | RMSE locale | Score public |
|---|---|---|---:|---:|
| 2026-10-02 | GradientBoosting de scikit-learn | — | 82.60 | ≈ 80 |
| 2026-10-02 | v1 : XGBoost + CatBoost + MLP, NNLS | — | 77.775 | 78.0 |
| 2026-10-16 | v2 : quatre familles, plusieurs graines, 10 plis | — | 77.302 | 77.799 |
| 2026-10-16 | v2 + jumelles | — | 77.060 | 77.604 |
| 2026-10-16 | variables de la série 1, CatBoost régularisé, trois modèles | 07 | 77.210 | 77.337 |
| 2026-10-16 | idem + jumelles | 07 | 76.966 | 77.158 |
| 2026-10-16 | variables de départ + jumelles (témoin) | 07 | 77.701 | 77.947 |
| 2026-10-16 | + TabM, cibles DOTS, RealMLP | 08 | 77.013 | 77.207 |
| 2026-10-16 | idem + jumelles | 08 | 76.757 | 77.018 |
| 2026-10-16 | idem + lissage par groupes | 08 | 76.521 | 76.731 |
| 2026-10-16 | v2 + jumelles + lissage par groupes | 08 | 76.774 | 77.125 |
| 2026-10-16 | variables finales, trois modèles + jumelles | 10 | 76.649 | 76.884 |
| 2026-10-16 | idem + lissage (même année) | 10 | 76.539 | 76.775 |
| 2026-10-16 | idem + lissage combiné | 10 | 76.510 | 76.716 |
| 2026-10-16 | six modèles × 3 graines, moyenne des modèles de plis | 11 | 76.044 | 76.520 |
| 2026-10-16 | idem, moitié modèles de plis, moitié réentraînement | 11 | — | 76.421 |
| 2026-10-16 | idem, arbres réentraînés sur toutes les lignes | 11 | — | 76.360 |
| 2026-10-17 | 8 graines de réentraînement | 12 | — | 76.359 |
| 2026-10-17 | + TabM sur 95 % | 12 | — | 76.337 |
| 2026-10-17 | + HGB | 13 | 76.010 | 76.334 |
| 2026-10-17 | trois modèles, référence de la série 4 | 14 | 76.361 | 76.592 |
| 2026-10-17 | + niveau du sportif (moins bon) | 14 | 76.508 | 76.802 |
| 2026-10-17 | huit modèles | 15 | 75.966 | 76.298 |
| 2026-11-10 | + lissage par paires de sportifs | 16 | 75.607 | 76.089 |
| 2026-11-10 | **+ lissage par paires dans la compétition** | 17 | **75.575** | **76.030** |

« — » : pas de score local quand toutes les lignes servent à l'entraînement.
