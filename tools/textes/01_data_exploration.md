# 01 · Exploration des données

On veut prédire `TotalKg`, le total des trois mouvements en kg, à partir des informations d'une participation à une compétition. La métrique est la RMSE. Il y a 105 000 lignes d'apprentissage, 45 000 de test, 3 variables numériques (âge, poids, année) et 13 catégorielles.

Ce notebook vérifie les fichiers, regarde les données et fixe les cinq plis utilisés partout ensuite. Il tourne sur CPU.

{NOTEBOOK_MAP}
<!-- cellule -->
## 1. Lecture et contrôles

Vérification des fichiers (empreinte, dimensions, colonnes, valeurs manquantes) et construction des cinq plis `FOLD_IDS`. Les autres notebooks commencent par la même cellule.
<!-- cellule -->
## 2. Variables

Il n'y a pas de cellule vide : les valeurs inconnues sont codées `Unknown`, surtout pour l'État de résidence, l'État de compétition et la fédération parente.
<!-- cellule -->
## 3. La cible

Les distributions sont très différentes entre hommes et femmes. Le poids corporel explique une partie du total, mais deux athlètes de même poids peuvent avoir plusieurs centaines de kilos d'écart. Le niveau de chacun n'est pas dans les variables : c'est ce qui limite la précision, et la RMSE finale reste autour de 76 kg.
<!-- cellule -->
## 4. Âge, année et fédération

Le trait pointillé est le total moyen de l'apprentissage.
<!-- cellule -->
## 5. Précision de saisie

En regardant les valeurs de près, on remarque deux choses :

- beaucoup d'âges finissent par **.5** : seule l'année de naissance est connue ;
- le poids est noté tantôt en entier, tantôt en .5, tantôt avec une ou deux décimales, selon les habitudes de pesée de la fédération.

Un arbre a du mal à retrouver ce genre de motif avec des seuils, donc on en fera des variables (notebook 02). Sur les résidus d'un premier modèle, l'écart entre âges entiers et âges en .5 atteint 24.5 kg à l'USAPL.
<!-- cellule -->
À l'USAPL, 79 % des âges sont en .5, avec un total moyen de 455 kg contre 498 kg pour les âges entiers. À la FPR : 491 contre 554 kg. L'USPA ne note jamais d'âge en .5 ; la THSPA garde deux décimales de poids dans 89 % des cas. Ces habitudes de saisie renseignent donc indirectement sur le type de compétition.
<!-- cellule -->
## 6. Lignes « jumelles »

Environ 1 % des lignes d'apprentissage ont une autre ligne identique sur les 16 variables. Dans trois cas sur quatre, le total est identique aussi : le même résultat est inscrit dans plusieurs divisions. Le test contient aussi des lignes qui ont une jumelle dans l'apprentissage.

On s'en sert dans le notebook 04 pour rapprocher la prédiction du total connu (gain d'environ 0.2 kg au score public). La clé `core` est moins stricte : âge, poids, année, sexe, fédération, pays et État de compétition, pays et État de résidence.
<!-- cellule -->
## 7. Qualité des données

Trois vérifications :

- **validation adversariale** : un classifieur essaie de distinguer apprentissage et test ; une AUC proche de 0.5 veut dire qu'il n'y arrive pas ;
- **ordre des id** : deux lignes voisines viennent-elles plus souvent de la même compétition ?
- **totaux suspects** : totaux très faibles par rapport au poids, souvent une compétition ratée ou incomplète.
<!-- cellule -->
L'AUC adversariale vaut 0.502 : apprentissage et test se ressemblent.

Après tri par id, deux lignes voisines ne partagent pas plus souvent la même compétition, fédération, année ou pays qu'avec un ordre aléatoire (rapport de 0.97 à 1.02). La corrélation entre id et total est de −0.003. L'id n'est donc pas utilisé comme variable.

Les 265 lignes suspectes (0.25 %) sont gardées : les retirer dégrade le score (notebook 05).
<!-- cellule -->
## 8. Cinq plis fixes

Les plis viennent de `KFold(5, shuffle=True, random_state=42)`. Toutes les variantes sont comparées sur les mêmes lignes de validation. L'ensemble final ajoute les graines 7 et 2024, qui ont chacune leur propre découpage.

Dans chaque pli, 10 % de la partie d'apprentissage sont mis de côté pour l'early stopping, et le pli de validation ne sert qu'au score. Nos deux premières versions (v1, v2) arrêtaient l'entraînement directement sur le pli de validation : leur score local était trop optimiste, environ 0.5 kg sous le score public pour v2, contre 0.13 à 0.25 kg depuis ce changement.
