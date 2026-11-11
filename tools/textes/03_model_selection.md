# 03 · Choix des modèles

Comparaison des modèles, des cibles et des façons de les combiner, toujours sur les cinq plis fixes avec early stopping interne. Dans l'ordre : premiers essais, réglage de CatBoost, autres modèles et cibles de force relative, cibles alternatives, grande comparaison, taux d'apprentissage.
<!-- cellule -->
{TRAIN_NOTE}
<!-- cellule -->
L'expérience 13 doit installer EBM et TabPFN avant l'import de numpy, d'où cette cellule placée avant la configuration.
<!-- cellule -->
## 0. Données et variables

Les variables ne sont recalculées (avec le code du notebook 02) que si un entraînement est lancé.
<!-- cellule -->
## 1. Premiers essais

Les tout premiers essais (Ridge, arbre, forêt aléatoire, gradient boosting de scikit-learn, puis LightGBM et CatBoost en local) sont dans le premier commit du dépôt. Ensuite viennent deux ensembles sur GPU : v1 (XGBoost, CatBoost, MLP) et v2 (plus LightGBM, plusieurs graines, dix plis).

Dans v2, les quatre familles font toutes entre 78.1 et 78.4 seules, avec des corrélations de 0.990 à 0.995 entre leurs prédictions : c'est l'information disponible qui limite, pas le modèle. Moyenner 3 graines fait passer le MLP de 80.05 à 78.40. Ces deux versions arrêtaient l'entraînement sur le pli de validation, ce qui rendait leur score trop optimiste (voir notebook 01, section 8).
<!-- cellule -->
## 2. Régularisation de CatBoost (expérience 07)

Trois réglages sur les variables de la première série :

- profondeur 8, L2 = 3 (le réglage de v2) ;
- profondeur 8, L2 = 10, échantillonnage Bernoulli 0.8 ;
- profondeur 6, L2 = 10, Bernoulli 0.8.

L'entraînement se fait avec la première série du notebook 02.
<!-- cellule -->
**Le deuxième réglage gagne 0.22 kg sur les 5 plis**, le troisième 0.17 kg. On garde le deuxième pour toute la suite.
<!-- cellule -->
## 3. Autres modèles, force relative et combinaison (expérience 08)

| Modèle | Pourquoi l'essayer |
|---|---|
| RealMLP (pytabkit) | MLP avec de bons réglages par défaut |
| TabM (pytabkit) | 32 MLP qui partagent leurs poids, avec des embeddings numériques linéaires par morceaux |
| CatBoost et LightGBM sur cible DOTS | prédire le score DOTS puis revenir en kg |

Pour les cibles relatives, chaque ligne est pondérée par le carré du coefficient : on minimise donc toujours l'erreur quadratique en kg, mais le modèle a une forme multiplicative et fait des erreurs différentes.

Trois façons de combiner sont comparées sur les mêmes plis : NNLS global, NNLS par sexe × équipement, et un méta-modèle LightGBM sur les résidus. Le lissage par sportif testé dans la même expérience est dans le notebook 05.

Non essayés : TabPFN-2.5 (prévu pour 50 000 lignes au plus), TabR (demande faiss), AutoGluon (mémoire insuffisante sur T4), cible log (biais au retour en kg).
<!-- cellule -->
TabM est le meilleur modèle seul (**77.636**) et fait gagner 0.15 kg à la combinaison, sur les 5 plis. Le CatBoost DOTS gagne 0.06 kg, RealMLP 0.07 kg (mais 15 minutes par pli), le LightGBM DOTS 0.04 kg. Les trois façons de combiner se valent ; on garde la plus simple, le NNLS global (poids positifs, sans constante, somme libre).
<!-- cellule -->
## 4. Cibles alternatives (expérience 14)

Deux autres barèmes que DOTS : une cible **allométrique** (total ÷ poids^(2/3)) et **IPF GL**, dont les coefficients dépendent du sexe et de l'équipement (Raw et Wraps comptent comme « sans équipement »). Même pondération et même retour en kg. L'entraînement se fait avec la quatrième série du notebook 02.
<!-- cellule -->
**Le CatBoost IPF GL fait 77.162**, autant que le CatBoost DOTS (77.160), avec une corrélation de 0.9975 entre les deux. Pourtant il fait gagner 0.076 kg à la combinaison sur les 5 plis : on le garde. La cible allométrique gagne 0.043 kg, sous le seuil.
<!-- cellule -->
## 5. Grande comparaison (expérience 13)

Tous les modèles reçoivent les 57 variables communes et les cinq plis de la graine 42. On regarde le score seul, puis ce que le modèle apporte quand on l'ajoute aux six modèles de l'expérience 11. Un modèle moyen seul peut être utile s'il se trompe autrement que les autres.

MLP-PLR et ResNet n'ont pas tourné (il manquait la dépendance skorch).
<!-- cellule -->
Les arbres boostés et TabM sont loin devant. TabPFN, la forêt aléatoire, ExtraTrees, EBM et Ridge sont 3 à 7 kg derrière et reçoivent un poids nul. Seul **HGB** apporte quelque chose : 77.881 seul, −0.034 kg dans la combinaison sur les 5 plis, poids 0.15. La soumission correspondante fait 76.334 au score public.
<!-- cellule -->
## 6. Taux d'apprentissage du CatBoost DOTS (expérience 12)

Comparaison 0.03 contre 0.05 sur les cinq plis de la graine 42. L'entraînement se lance avec le bouton `more_seeds` du notebook 04.
<!-- cellule -->
Avec 0.03, le gain moyen est de 0.07 kg (4 plis sur 5), pour environ 1.44 fois plus d'itérations. On l'utilise pour les réentraînements des deux CatBoost relatifs.

## 7. Les huit modèles gardés pour la suite

| Modèle | Cible | Réglage |
|---|---|---|
| LightGBM | kg | paramètres de v2 |
| XGBoost | kg | paramètres de v2, GPU |
| CatBoost | kg | profondeur 8, L2 10, Bernoulli 0.8 |
| LightGBM DOTS | score DOTS | comme LightGBM |
| CatBoost DOTS | score DOTS | comme CatBoost |
| CatBoost IPF GL | score IPF GL | comme CatBoost |
| TabM | kg standardisé | pytabkit, embeddings linéaires par morceaux |
| HGB | kg | HistGradientBoosting de scikit-learn, catégories natives |

Leurs poids dans la combinaison sont dans le notebook 04.
