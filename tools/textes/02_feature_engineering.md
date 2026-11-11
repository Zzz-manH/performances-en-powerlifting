# 02 · Construction des variables

Quatre séries de variables, testées l'une après l'autre. Les trois premières n'utilisent que les caractéristiques des lignes (apprentissage et test réunis), jamais la cible. La quatrième utilise les totaux connus, calculés hors pli.

| Série | Blocs testés | Résultat |
|---|---|---|
| 1 (exp. 07) | précision de saisie, déplacements, contexte de compétition | gardés ensemble : −0.67 kg |
| 2 (exp. 09) | contexte enrichi, divisions, historique du sportif, coefficient DOTS, identifiants | seul l'historique passe le seuil : −0.13 kg |
| 3 (exp. 10) | carrière, expérience du plateau, ajouts propres à chaque modèle | gardés : −0.20 kg |
| 4 (exp. 14) | niveau du sportif et de la compétition, à partir des totaux | moins bon, abandonné |

**Règle.** Chaque bloc est jugé sur la combinaison NNLS de LightGBM, XGBoost et CatBoost, avec les cinq plis fixes et l'early stopping interne. On le garde s'il fait gagner au moins 0.05 kg en moyenne et s'il améliore au moins 4 plis sur 5.
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
## 0. Données et plis
<!-- cellule -->
## 1. Première série (expérience 07)

Le point de départ est l'analyse des résidus de la version v2 (RMSE hors pli 77.302). Plusieurs groupes ont un résidu moyen nettement différent de zéro :

| Groupe | Résidu moyen (kg) | Ce que ça suggère |
|---|---|---|
| Âge entier / en .5 | +2.6 / −3.6 (USAPL : +19.0 / −5.5) | la précision de la date de naissance dépend de la compétition |
| Poids entier / à deux décimales | −2.9 / +1.3 | idem pour la pesée |
| Compétition dans un autre État / dans le sien | +3.6 / −1.8 | on se déplace pour les compétitions plus relevées |
| À l'étranger / dans son pays | +1.6 / −0.5 | compétitions internationales |
| Plus petites compétitions (1/8 des lignes) | −3.6 | compétitions locales |

Une « compétition » est approchée par fédération × année × pays × État.

| Bloc | Variables |
|---|---|
| Départ | 19 colonnes : les 16 d'origine, plus `WeightClass_num`, `Is_SHW`, `BW_minus_Class` |
| Précision | `age_half` ; `bw_precision` (0 entier, 1 en .5, 2 une décimale, 3 deux décimales) ; `bw_lb_int` |
| Déplacements | `abroad`, `other_state` (NaN si la résidence est inconnue) |
| Contexte | taille de la compétition, du groupe sexe/équipement, de la fédération dans l'année ; parts de déplacements, d'âges en .5, d'équipements, d'hommes ; âge moyen |
<!-- cellule -->
## 2. Deuxième série (expérience 09)

On part des 33 colonnes de la première série.

| Bloc | Contenu |
|---|---|
| Contexte enrichi | diversité des États, pays et catégories ; parts de contrôles antidopage, d'équipements, de précisions de poids, de juniors et masters ; dispersion des âges, poids moyen par sexe, taille nationale |
| Divisions | taille de la division (compétition × sexe × équipement × catégorie), rang de poids et d'âge, nombre de jumelles |
| Historique du sportif | pour un sportif approché : nombre d'apparitions, années actives, temps depuis la première et jusqu'à la dernière, équipements utilisés, écart de poids |
| Coefficient DOTS | coefficient poids–sexe : total = score DOTS × coefficient |
| Identifiants | clés de compétition et de sportif données comme catégories, pour CatBoost seulement |

Clé du sportif : sexe + pays + État + fédération + année de naissance. L'année de naissance est estimée par `floor(MeetYear − Age)` et peut être fausse d'un an quand l'âge est entier.
<!-- cellule -->
## 3. Troisième série (expérience 10)

On part des 41 colonnes obtenues en ajoutant l'historique.

| Bloc | Contenu |
|---|---|
| Carrière | plus grande compétition fréquentée, nombre d'États, compétitions à l'étranger, catégories, amplitude de poids, âge à la première apparition, apparitions par an, part de compétitions contrôlées, correspondances avec une année de naissance à ±1 |
| Expérience du plateau | pour une compétition : nombre moyen d'apparitions et d'années d'expérience des participants, part de vétérans (3 ans ou plus) et de débutants |
| Ajouts par modèle | contexte enrichi + identifiants pour CatBoost ; contexte enrichi + DOTS + divisions pour XGBoost ; rien de plus pour LightGBM |

Le dernier bloc vient de la série 2 : certains blocs aidaient un modèle sans aider les autres.
<!-- cellule -->
## 4. Variables tirées des résultats (expérience 14)

Les variables sans cible n'apportent presque plus rien. Le lissage du notebook 04 utilise déjà les autres totaux d'un sportif, mais après la prédiction. Ici on essaie de les donner directement au modèle :

- niveau du sportif : nombre, moyenne, maximum et moyenne pondérée dans le temps de ses autres scores DOTS, écarts d'année et de poids ;
- niveau de la compétition : force relative des autres participants.

Pour éviter la fuite, la partie d'apprentissage de chaque pli est redécoupée en 5 : une ligne ne voit que les totaux des autres sous-plis. La validation et le test utilisent toute la partie d'apprentissage.
<!-- cellule -->
## 5. Ablations

Dans les tableaux : `mean_delta` est la variation moyenne de RMSE (négative = mieux), `folds_better` le nombre de plis améliorés, `keep` indique si le seuil est atteint. Sur les figures, la barre est la moyenne, les points sont les plis, le trait pointillé le seuil de −0.05.
<!-- cellule -->
### 5.1 Première série (expérience 07)

LightGBM (CPU), XGBoost et CatBoost (GPU) gardent les paramètres de la version v2. Le même entraînement sert aussi à tester la régularisation de CatBoost (notebook 03) et la correction des lignes jumelles.
<!-- cellule -->
La précision fait gagner 0.33 kg et le contexte 0.36 kg, sur les 5 plis à chaque fois. Les déplacements seuls ne gagnent que 0.04 kg, sous le seuil, mais leurs colonnes servent à calculer le contexte et on les garde. **Les trois blocs ensemble gagnent 0.67 kg sur les 5 plis.** La correction des jumelles ajoute environ 0.25 kg.

Au score public : 77.947 avec les variables de départ, 77.158 avec les trois blocs et le CatBoost régularisé (les deux avec correction des jumelles).
<!-- cellule -->
### 5.2 Deuxième série (expérience 09)

CatBoost utilise maintenant les paramètres choisis dans le notebook 03 (L2 10, Bernoulli 0.8).
<!-- cellule -->
**Seul l'historique du sportif passe le seuil : −0.13 kg sur les 5 plis.** Les identifiants aident CatBoost (−0.12) mais presque pas la combinaison (−0.05). Le contexte enrichi aide CatBoost et XGBoost (−0.06) et gêne LightGBM (+0.07). Les divisions et le coefficient DOTS n'aident que XGBoost. D'où l'idée de donner à chaque modèle ses propres ajouts dans la série suivante.
<!-- cellule -->
### 5.3 Troisième série (expérience 10)

Les variantes de lissage essayées pendant cette expérience sont dans le notebook 05.
<!-- cellule -->
Carrière : −0.09 (5 plis sur 5). Expérience du plateau : −0.07 (4 sur 5). Ajouts par modèle : −0.11 (5 sur 5). **Les trois ensemble : −0.20 kg**, soit 76.876 pour la combinaison des trois modèles. Après cela, la plupart des résidus moyens par groupe sont sous 2 kg.
<!-- cellule -->
### 5.4 Quatrième série (expérience 14)

La référence est le jeu de variables final. La combinaison réunit LightGBM, XGBoost et le CatBoost à cible DOTS (notebook 03). Comme ces variables recoupent le lissage, on regarde aussi le score après lissage.

Le même bouton lance le contrôle des id (notebook 01) et les cibles alternatives (notebook 03).
<!-- cellule -->
Le niveau du sportif dégrade la RMSE de 0.14 kg et le niveau de la compétition de 0.52 kg, sans aucun pli amélioré. C'est pareil après lissage et au score public (76.802 contre 76.592). Deux explications possibles : à l'entraînement ces statistiques sont calculées sur 80 % des lignes seulement, donc pas tout à fait comme pour le test ; et elles sont bruitées, ce qui pousse au surapprentissage. La même information marche mieux dans le lissage, après les modèles.
<!-- cellule -->
## 6. Variables finales

| Modèles | Blocs | Colonnes |
|---|---|---:|
| LightGBM (kg et DOTS), TabM, HGB | départ + précision + déplacements + contexte + historique + carrière + expérience du plateau | 57 |
| XGBoost | les mêmes + contexte enrichi + DOTS + divisions | 78 |
| CatBoost (kg, DOTS, IPF GL) | les mêmes + contexte enrichi + identifiants | 73 |

Les notebooks 03 et 04 réutilisent le code des sections 1 à 3 quand un entraînement est lancé.
