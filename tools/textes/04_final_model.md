# 04 · Modèle final et soumissions

La soumission finale combine huit modèles (expérience 15) puis applique trois corrections, dont le lissage par paires (expériences 16–17) : **75.575 en local, 76.030 au score public**. Avant le lissage par paires : 75.966 / 76.298.

| Étape | Ce qui est fait |
|---|---|
| Variables | jeu final du notebook 02 : 57, 73 ou 78 colonnes selon le modèle |
| Modèles | LightGBM, XGBoost, CatBoost, LightGBM DOTS, CatBoost DOTS, CatBoost IPF GL, TabM, HGB |
| Validation | 5 plis × 3 graines (42, 7, 2024), early stopping interne, moyenne des graines |
| Test | arbres réentraînés sur toutes les lignes (itérations = moyenne de validation × 1.2) ; TabM sur 95 %, HGB sur 90 % |
| Combinaison | NNLS global |
| Corrections | lignes jumelles, lissage par groupes, lissage par paires ; coefficients ajustés sur une moitié des données et évalués sur l'autre |

Attention : les scores locaux des corrections ne sont pas une validation de toute la chaîne. Les coefficients sont bien évalués sur des lignes qui n'ont pas servi à les ajuster, mais les modèles de base ne sont pas réentraînés pour ce découpage.

Pour tout refaire depuis zéro : `six_models`, redémarrer, `more_seeds`, redémarrer, `eight_models`, puis `pair_smoothing`.
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
## 0. Données, variables et modèles

Les variables et les définitions des modèles ne sont chargées que si un entraînement de modèles est lancé.
<!-- cellule -->
## 1. Six modèles × cinq plis × trois graines (expérience 11)

LightGBM et XGBoost gardent les paramètres de v2, CatBoost le réglage choisi dans le notebook 03. Pour les modèles DOTS : cible = total ÷ coefficient, poids = coefficient², puis retour en kg. TabM utilise les paramètres par défaut de pytabkit avec les embeddings linéaires par morceaux ; les valeurs manquantes sont remplacées par la médiane avec un indicateur, et la cible est standardisée.

Trois façons de prédire le test sont comparées : la moyenne des 15 modèles de plis, des arbres réentraînés sur toutes les lignes (un par graine), ou la moyenne des deux. Le même bouton lance l'essai de nettoyage du notebook 05.
<!-- cellule -->
Moyenner trois graines fait gagner 0.4 à 0.8 kg à chaque modèle. Le CatBoost DOTS est le meilleur seul (76.784, poids 0.45). En local : 76.395 pour la combinaison, 76.172 avec les jumelles, 76.044 avec le lissage.

Au score public, la moyenne des modèles de plis fait 76.520 et le réentraînement complet **76.360**. Plus de données, c'est mieux : toutes les soumissions suivantes utilisent le réentraînement complet. Il n'a pas de score local, puisque toutes les lignes servent à l'entraînement.
<!-- cellule -->
## 2. Plus de graines et TabM sur 95 % (expérience 12)

Cinq graines de plus pour le CatBoost DOTS, XGBoost, le LightGBM DOTS et LightGBM, soit huit réentraînements chacun. Les nouveaux CatBoost DOTS utilisent un taux de 0.03 (notebook 03). TabM n'a pas de nombre d'itérations qu'on puisse reprendre : il est entraîné sur 95 % des lignes, avec 5 % pour l'early stopping, sur trois graines. Les poids et les coefficients des corrections restent ceux de l'expérience 11.
<!-- cellule -->
## 3. Ajout de HGB (expérience 13)

Dans la grande comparaison du notebook 03, seul HGB passe le seuil. Ajouté à la version précédente, il donne **76.334** au score public. L'entraînement est dans le bouton `model_compare` du notebook 03.
<!-- cellule -->
## 4. Huit modèles (expérience 15)

On réutilise ce qui existe et on complète :

| Étape | Travail |
|---|---|
| 1 | CatBoost IPF GL : cinq plis pour les graines 7 et 2024 (la graine 42 vient de l'expérience 14) |
| 2 | HGB : cinq plis pour les graines 7 et 2024 (la graine 42 vient de l'expérience 13) |
| 3 | CatBoost IPF GL réentraîné sur toutes les lignes, 5 graines, taux 0.03 |
| 4 | HGB réentraîné avec 10 % d'early stopping interne, 3 graines |
| 5 | TabM sur 95 % : 2 graines de plus, donc 5 |
| 6 | NNLS sur les 8 modèles, corrections, export, contrôles |

Avant l'export, le notebook vérifie que les cinq plis existent pour chaque modèle et chaque graine, et s'arrête si un fichier manque. Les scores ci-dessous sont recalculés à partir des prédictions sauvegardées, puis comparés au fichier exporté.
<!-- cellule -->
Le CatBoost IPF GL (76.781) et le CatBoost DOTS (76.784) se valent ; le premier a le plus gros poids (0.29). HGB est moins bon seul mais se trompe autrement : poids 0.20. LightGBM, le LightGBM DOTS et le CatBoost en kg tombent à zéro.

La combinaison passe de 76.395 à 76.351 avec HGB, puis à **76.319** avec le CatBoost IPF GL, et à **75.966** après les deux premières corrections.
<!-- cellule -->
## 5. Lissage par paires (expériences 16–17)

Le lissage par groupes prend la moyenne des résidus d'un groupe sexe + pays + État + année de naissance. Mais un tel groupe contient huit lignes en moyenne, et la plupart ne sont pas le même sportif. Ici on estime, paire par paire, ce que le résidu d'une ligne d'apprentissage `j` dit de celui d'une ligne `i`.

| Étape | Ce qui est fait |
|---|---|
| Paires candidates | même sexe, pays et État (pays connu), années de naissance à un an près ; si les deux âges sont en .5, même année exigée. Environ 19.6 millions de paires |
| Modèle de paires | LightGBM qui régresse `r_i × r_j / Var(r)` sur l'écart de poids, d'année, de naissance, l'égalité de fédération / équipement / catégorie, le nombre de candidats… La prédiction, coupée à zéro, sert de poids |
| Agrégats | cinq sommes de résidus voisins pondérées par ces poids, ajoutées aux 13 statistiques du lissage par groupes dans la même régression |
| Second passage | la même chose sur les résidus obtenus, pour les paires d'une même compétition × sexe × équipement (groupes de 300 lignes au plus), avec une régression ridge |
| Validation | les deux mêmes moitiés que pour les autres corrections ; le modèle de paires n'apprend que sur des paires dont les deux lignes sont dans l'autre moitié |

Le code est dans `tools/pair_smoothing.py`. Il relit les prédictions de l'expérience 15, ne réentraîne aucun modèle de base et tourne sur CPU. Les scores ci-dessous sont recalculés à partir des prédictions sauvegardées dans `artifacts/04_final_model/v5_pair_smoothing/`.
<!-- cellule -->
Le lissage par paires de sportifs fait passer le score local de 75.966 à **75.607** (−0.359), sur les 5 plis et les 2 moitiés, et le score public de 76.298 à 76.089. Le second passage ajoute −0.032 en local (**75.575**) et donne 76.030 au public. C'est juste au-dessus du seuil de 0.03 qu'on s'était fixé avant de le lancer, donc on s'arrête là.

Variantes essayées sans gain (détail dans `RESULTS.md`) : paires avec un État inconnu d'un seul côté, ridge au premier passage, 600 itérations au lieu de 300, lissage par paires sans les statistiques de groupes.
<!-- cellule -->
## 6. Fichiers de soumission et score public

Le score public porte sur environ 30 % du test (13 500 lignes). Un écart de moins de 0.05 kg entre deux soumissions ne veut pas dire grand-chose (voir la simulation du notebook 05).
<!-- cellule -->
Fichiers gardés dans `submissions/` :

1. `20261110_v6_meet_smoothing.csv` : soumission finale, meilleur score local et public (75.575 / 76.030) ;
2. `20261017_FINAL_v4_eight_models.csv` : la version sans lissage par paires (75.966 / 76.298).

Les deux passages du lissage par paires gagnent 0.391 kg en local et 0.268 kg au public : même sens, mais moins au public.
