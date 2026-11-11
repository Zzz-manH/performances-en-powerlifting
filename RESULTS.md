# Résultats des expériences

Sauf mention contraire, les scores sont des RMSE hors pli en kg, calculées sur les 105 000 lignes avec `KFold(5, shuffle=True, random_state=42)`. Les numéros d'expérience sont ceux de notre journal ; le notebook correspondant est indiqué dans chaque titre.

## Vue d'ensemble

| Étape | RMSE locale | Score public |
|---|---:|---:|
| v1 : XGBoost + CatBoost + MLP, NNLS | 77.775 | 78.0 |
| v2 : quatre familles, plusieurs graines, 10 plis | 77.302 | 77.799 |
| v2 + lignes jumelles | 77.060 | 77.604 |
| 07 : précision de saisie, déplacements, contexte ; CatBoost régularisé ; jumelles | 76.966 | 77.158 |
| 08 : + TabM, cibles DOTS, RealMLP ; lissage par groupes | 76.521 | 76.731 |
| 10 : + historique et carrière du sportif ; lissage combiné | 76.510 | 76.716 |
| 11 : six modèles × 3 graines, arbres réentraînés sur toutes les lignes | 76.044 (modèles de plis) | 76.360 |
| 12 : 8 graines de réentraînement, TabM sur 95 % | — | 76.337 |
| 13 : + HGB | 76.010 | 76.334 |
| 15 : huit modèles (+ CatBoost IPF GL) | 75.966 | 76.298 |
| 16 : + lissage par paires de sportifs | 75.607 | 76.089 |
| **17 : + lissage par paires dans la compétition (finale)** | **75.575** | **76.030** |

Les plus gros gains : variables de précision et de contexte (−0.67), moyenne de plusieurs graines (−0.4 à −0.8 par modèle), lissage par paires (−0.39), lissage par groupes (−0.24), jumelles (environ −0.2), réentraînement sur toutes les lignes (−0.16 au public).

Les versions v1 et v2 arrêtaient l'entraînement sur le pli de validation, d'où un score local trop optimiste (0.5 kg sous le public pour v2). Depuis l'expérience 07, l'early stopping se fait sur 10 % de la partie d'apprentissage et l'écart est de 0.13 à 0.25 kg.

## Données (notebook 01)

- Pas de cellule vide ; les valeurs inconnues sont codées `Unknown`. 72 454 hommes, 32 537 femmes, 9 Mx ; 62 % en Raw.
- 48 % des âges sont en .5 (seule l'année de naissance est connue) ; la précision du poids dépend de la fédération. Sur les résidus de v2 à l'USAPL : +19.0 pour les âges entiers, −5.5 pour les âges en .5.
- 1 109 lignes d'apprentissage (1.06 %) ont une jumelle exacte sur les 16 variables, avec le même total dans 73.4 % des cas ; 100 autres coïncident sur 9 variables. Dans le test : 465 jumelles exactes et 54 sur 9 variables. Sur ces lignes, la RMSE de v2 passe de 68.5 à 35.8 quand on utilise le total de la jumelle.
- L'id est aléatoire (aucun lien avec la compétition ni l'année).
- 265 lignes suspectes (0.25 %) : total inférieur à 2.5 × poids (hommes) ou 1.8 × poids (femmes), ou âge inférieur à 5 ans.
- 5.6 % des lignes ont un résidu de plus de 150 kg en valeur absolue et pèsent 37 % de l'erreur quadratique de v2.

## Variables (notebook 02)

Chaque bloc est jugé sur la combinaison NNLS de LightGBM, XGBoost et CatBoost. Seuil : −0.05 kg en moyenne et au moins 4 plis améliorés sur 5.

**Série 1 (expérience 07), référence = 19 colonnes de départ**

| Variante | LightGBM | XGBoost | CatBoost | Combinaison | Écart | Plis |
|---|---:|---:|---:|---:|---:|---:|
| Départ | 78.579 | 79.323 | 78.824 | 77.956 | — | — |
| Déplacements | 78.533 | 79.251 | 78.763 | 77.912 | −0.044 | 4/5 |
| Précision | 78.293 | 78.958 | 78.557 | 77.631 | −0.326 | 5/5 |
| Contexte | 78.254 | 78.955 | 78.349 | 77.594 | −0.362 | 5/5 |
| **Les trois** | **77.923** | **78.806** | **78.065** | **77.283** | **−0.673** | 5/5 |

**Série 2 (expérience 09)**

| Bloc | Combinaison | Écart | Plis |
|---|---:|---:|---:|
| Référence | 77.208 | — | — |
| **Historique du sportif** | **77.081** | **−0.127** | 5/5 |
| Identifiants (CatBoost seulement) | 77.161 | −0.047 | 5/5 |
| Contexte enrichi | 77.212 | +0.003 | 2/5 |
| Coefficient DOTS | 77.260 | +0.052 | 0/5 |
| Divisions | 77.263 | +0.055 | 0/5 |

Avec l'historique, le résidu moyen des sportifs vus une seule fois passe de −4.5 à −1.8.

**Série 3 (expérience 10)**

| Bloc | Combinaison | Écart | Plis |
|---|---:|---:|---:|
| Référence | 77.079 | — | — |
| Carrière | 76.993 | −0.086 | 5/5 |
| Expérience du plateau | 77.007 | −0.072 | 4/5 |
| Ajouts par modèle | 76.970 | −0.110 | 5/5 |
| **Les trois** | **76.876** | **−0.203** | 5/5 |

**Série 4 (expérience 14) : variables tirées des totaux, calculées hors pli**

| Variante | Combinaison | Écart | Plis | Score public après corrections |
|---|---:|---:|---:|---:|
| Référence | 76.717 | — | — | 76.592 |
| Niveau du sportif | 76.856 | +0.139 | 0/5 | 76.802 |
| Niveau de la compétition | 77.233 | +0.516 | 0/5 | |

**Variables finales** : 57 colonnes pour LightGBM, TabM et HGB ; 78 pour XGBoost (+ contexte enrichi, DOTS, divisions) ; 73 pour CatBoost (+ contexte enrichi, identifiants).

## Modèles (notebook 03)

**Premiers essais** (5 plis, moyenne des plis) : Ridge 105.39, arbre de décision 88.48, forêt aléatoire 85.16, GradientBoosting 82.60. Un essai de target encoding sur tout l'apprentissage donnait 80.40, mais avec une fuite de cible : résultat écarté. Dans v2, les quatre familles font 78.1 à 78.4 seules, avec des corrélations de 0.990 à 0.995.

**CatBoost (expérience 07)** : profondeur 8, L2 3 : 78.065 ; **profondeur 8, L2 10, Bernoulli 0.8 : 77.846** (5/5) ; profondeur 6, L2 10 : 77.890.

**Autres modèles (expérience 08)**, ajoutés à la combinaison de l'expérience 07 :

| Modèle | Seul | Écart dans la combinaison | Plis | Temps par pli |
|---|---:|---:|---:|---:|
| TabM | 77.636 | −0.153 | 5/5 | 3.7 min |
| CatBoost DOTS | 77.761 | −0.055 | 5/5 | 1.4 min |
| LightGBM DOTS | 77.944 | −0.044 | 5/5 | 14 s |
| RealMLP | 78.483 | −0.072 | 5/5 | 15 min |

Façons de combiner : NNLS global 77.013, NNLS par sexe × équipement 77.017, méta-modèle de résidus 77.098.

**Cibles (expérience 14)** : CatBoost DOTS 77.160 ; CatBoost IPF GL 77.162 (corrélation 0.9975, −0.076 dans la combinaison, 5/5) ; cible allométrique 77.327 (−0.043).

**Grande comparaison (expérience 13)**, 57 variables, graine 42 :

| Modèle | Seul | Apport aux six modèles de l'expérience 11 |
|---|---:|---|
| CatBoost DOTS / TabM / CatBoost / LightGBM DOTS / LightGBM / XGBoost | 77.167 / 77.493 / 77.497 / 77.646 / 77.709 / 78.458 | (déjà dedans) |
| **HGB** | 77.881 | 76.395 → 76.361, 5/5, poids 0.147 |
| ExtraTrees, EBM, TabPFN, forêt aléatoire | 81.1 à 81.6 | poids nul |
| Ridge | 84.969 | poids nul |

**Taux d'apprentissage du CatBoost DOTS (expérience 12)** : 0.03 contre 0.05, −0.070 en moyenne, 4 plis sur 5, 1.44 fois plus d'itérations.

## Ensemble final (notebook 04)

**Expérience 11 : six modèles × 5 plis × 3 graines.** Combinaison 76.395, avec jumelles 76.172, avec lissage 76.044. Au public : moyenne des modèles de plis 76.520, moitié-moitié 76.421, arbres réentraînés sur toutes les lignes **76.360**.

**Expérience 15 : huit modèles**

| Modèle | RMSE, moyenne de 3 graines | Prédiction du test | Poids NNLS |
|---|---:|---|---:|
| CatBoost IPF GL | 76.781 | 5 réentraînements complets | 0.289 |
| CatBoost DOTS | 76.784 | 8 réentraînements complets | 0.218 |
| HGB | 77.242 | 3 réentraînements | 0.200 |
| TabM | 77.050 | 5 réentraînements sur 95 % | 0.168 |
| XGBoost | 77.693 | 8 réentraînements complets | 0.125 |
| LightGBM / LightGBM DOTS / CatBoost | 77.200 / 77.167 / 77.097 | | 0 |

| Étape | RMSE locale |
|---|---:|
| Six modèles | 76.395 |
| + HGB | 76.351 |
| + CatBoost IPF GL | 76.319 |
| + jumelles (alpha 0.854 pour une jumelle exacte, 0.523 sur 9 variables) | 76.097 |
| + lissage par groupes | 75.966 |

**Expérience 16 : lissage par paires de sportifs.** 19 597 625 paires candidates, dont 13 603 214 entre lignes d'apprentissage.

| Variante | RMSE locale | Écart | Plis / moitiés | Public |
|---|---:|---:|---:|---:|
| Huit modèles, lissage par groupes | 75.966 | — | — | 76.298 |
| **A : + lissage par paires** | **75.607** | **−0.359** | 5/5, 2/2 | **76.089** |
| B : A + paires dont un seul État est inconnu | 75.608 | −0.358 | 5/5, 2/2 | |
| B + ridge (1e-3 / 1e-2) | 75.607 / 75.606 | | | |
| B sans les statistiques de groupes | 75.633 | −0.332 | | |
| B avec 600 itérations au lieu de 300 | 75.634 | −0.332 | | |

Une première version de B affichait 75.548 : le croisement laissait passer des paires d'ajustement dont une ligne était dans la moitié évaluée. Une fois corrigé, le gain disparaît.

**Expérience 17 : second passage, paires d'une même compétition × sexe × équipement.**

| Variante | RMSE locale | Écart avec A | Plis / moitiés | Public |
|---|---:|---:|---:|---:|
| A | 75.607 | — | | 76.089 |
| G : + paires dans la compétition | 75.580 | −0.027 | 5/5, 1/2 | |
| **H : G avec ridge 1e-2** | **75.575** | **−0.032** | 5/5, 2/2 | **76.030** |

Le seuil fixé avant l'essai était de 0.03 kg. On s'arrête là.

## Analyse des erreurs (notebook 05)

**Variantes du lissage par groupes (expérience 10)**, à partir de 76.649 (trois modèles + jumelles) :

| Variante | RMSE locale | Public |
|---|---:|---:|
| Même année de naissance | 76.539 | 76.775 |
| Année à ±1 | 76.544 | |
| Pondération temporelle | 76.527 | |
| **Les trois, combinaison linéaire** | **76.510** | **76.716** |
| Les trois, petit LightGBM | 76.858 | |

**Retrait des 265 lignes suspectes (expérience 11)** : LightGBM +0.043, XGBoost +0.063, CatBoost +0.081 sur toute la validation ; −0.01 à −0.03 sur les seules lignes non suspectes.

**Résidus de la version à huit modèles** : moins de 1.5 kg d'écart moyen dans la plupart des groupes ; femmes en Multi-ply +4.3 (380 lignes), 140 kg et plus +3.9, plus de 60 ans −2.5. RMSE de 54 kg (femmes en Raw) à 111 kg (hommes en Multi-ply).

**Score local et score public** : corrélation de rang 0.98 depuis l'expérience 07 (lissage par paires compris). L'écart public − local est de 0.13 à 0.33 kg jusqu'aux huit modèles, puis de 0.45 à 0.48 kg avec le lissage par paires. En tirant 13 500 lignes au hasard parmi les prédictions hors pli, l'écart-type d'un score est de 0.59 kg, celui de la différence entre deux soumissions proches de 0.03 kg.

**Ce qui n'a pas marché** : variables tirées des totaux ; target encoding grossier (+0.27) ; fréquences et interactions simples (±0.05) ; TabPFN, forêt aléatoire, ExtraTrees, EBM, Ridge ; NNLS par groupe ; lissage par LightGBM ; retrait des lignes suspectes ; cible log. La base OpenPowerlifting n'est pas utilisée car elle contient les lignes du test.
