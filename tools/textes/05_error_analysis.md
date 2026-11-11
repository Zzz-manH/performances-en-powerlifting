# 05 · Analyse des erreurs

Où le modèle se trompe encore, pourquoi les corrections après prédiction marchent, ce qui n'a pas marché, et dans quelle mesure on peut se fier au score local. Ce notebook ne fait que relire les résultats des notebooks 02 à 04.
<!-- cellule -->
## 0. Données
<!-- cellule -->
## 1. Résidus

Résidu = total réel − total prédit, en kg. On compare la combinaison des huit modèles et la même après jumelles et lissage par groupes (expérience 15, donc avant le lissage par paires).
<!-- cellule -->
Dans la plupart des groupes, le résidu moyen est à moins de 1.5 kg de zéro. Les écarts les plus nets concernent de petits groupes : femmes en Multi-ply (380 lignes) +4.3, poids de 140 kg et plus +3.9, plus de 60 ans −2.5, sportifs vus une seule fois −1.3. Le groupe Mx n'a que 9 lignes.

La RMSE va de 54 kg pour les femmes en Raw à 111 kg pour les hommes en Multi-ply. Les 5.3 % de lignes dont l'erreur dépasse 150 kg pèsent 36 % de l'erreur quadratique. Ce sont souvent des compétitions incomplètes (par exemple 155 kg au total pour un athlète de 216 kg), mais les retirer n'arrange rien (section 3).
<!-- cellule -->
## 2. Structure des résidus et corrections

Les variables ne disent pas le niveau de quelqu'un, mais l'apprentissage contient souvent d'autres résultats de la même personne. Pour le voir, on corrèle le résidu d'une ligne avec le résidu moyen des autres lignes de la même clé. Avec les clés du tableau, cette corrélation vaut environ 0.07 pour la version v2, 0.045 pour la combinaison des huit modèles (qui ont les variables d'historique et de carrière), et à peu près 0 après les corrections.

Pour la clé de compétition, elle passe de −0.008 à 0.037 : les corrections laissent une petite corrélation entre lignes d'une même compétition, que le second passage du lissage par paires exploite (notebook 04).
<!-- cellule -->
Les coefficients sont ajustés sur une moitié des groupes sexe + pays + État et évalués sur l'autre, de sorte que toutes les lignes d'une personne soient du même côté.

1. **Jumelles** : `pred + alpha × (moyenne des jumelles − pred)`, avec alpha ≈ 0.85 pour une ligne identique sur les 16 colonnes et ≈ 0.5 pour la clé à 9 colonnes.
2. **Lissage par groupes** : moyenne des résidus des autres lignes du groupe, ramenée vers zéro quand le groupe est petit, puis combinaison linéaire. Les groupes : sexe + pays + État + année de naissance, puis en ajoutant la fédération, l'équipement, la catégorie ; plus un groupe « même compétition ».

Cinq variantes sont comparées : même année de naissance ; année à ±1 ; pondération qui décroît avec l'écart d'années de compétition ; les trois ensemble, combinées linéairement ; les trois ensemble, combinées par un petit LightGBM.
<!-- cellule -->
Les jumelles font gagner 0.22 à 0.26 kg en local et 0.18 / 0.19 kg sur deux soumissions publiques. Le lissage gagne 0.24 kg dans l'expérience 08 et 0.29 / 0.48 kg au public. **La combinaison linéaire des trois variantes est la meilleure** ; le petit LightGBM surapprend (+0.21). Une fois les variables d'historique ajoutées, le lissage ne gagne plus que 0.13 kg : les deux se recoupent en partie.

Le lissage par groupes traite toutes les lignes d'un groupe de la même façon. Le lissage par paires (notebook 04, section 5) apprend un poids pour chaque paire et gagne encore 0.39 kg : 75.966 → 75.607 → 75.575.
<!-- cellule -->
## 3. Essai de nettoyage (expérience 11)

Lignes suspectes : total inférieur à 2.5 × poids chez les hommes, à 1.8 × poids chez les femmes, ou âge inférieur à 5 ans. Cela fait environ 0.25 % des lignes.

On les retire de l'apprentissage et de l'early stopping, mais on garde toute la validation. Comparaison pli par pli avec le modèle normal, puis sur les seules lignes non suspectes.
<!-- cellule -->
Les trois modèles sont **moins bons de 0.04 à 0.08 kg**, avec au mieux 2 plis améliorés sur 5. Sur les lignes non suspectes seules, le gain est de 0.01 à 0.03 kg. Comme le test contient lui aussi des résultats incomplets, on garde toutes les lignes.
<!-- cellule -->
## 4. Ce qui n'a pas marché

| Piste | Résultat | Où |
|---|---|---|
| Niveau du sportif et de la compétition à partir des totaux | +0.14 / +0.52, public moins bon aussi | 02, série 4 |
| Contexte enrichi, divisions, coefficient DOTS (dans la combinaison) | +0.00 à +0.05 | 02, série 2 |
| Déplacements seuls | −0.04, sous le seuil | 02, série 1 |
| Target encoding grossier (sexe × équipement × catégorie) | +0.27 | premiers essais |
| Fréquences, interaction sexe × équipement, année de naissance | ±0.05 | premiers essais |
| TabPFN, forêt aléatoire, ExtraTrees, EBM, Ridge | 3 à 7 kg derrière, poids nul | 03, section 5 |
| NNLS par groupe, méta-modèle de résidus | pas mieux que le NNLS global | 03, section 3 |
| Lissage combiné par LightGBM | surapprentissage, +0.21 | section 2 |
| Retrait des lignes suspectes | +0.04 à +0.08 | section 3 |
| id comme variable | ordre aléatoire | 01, section 7 |
| Cible log | biais au retour en kg | — |
| Base OpenPowerlifting | contient les lignes du test : ce serait une fuite, on ne l'utilise pas | — |
<!-- cellule -->
## 5. Score local et score public

Toutes les soumissions qui ont un score local sont comparées à leur score public. Ensuite, pour savoir ce que vaut un écart au public, on tire au hasard 13 500 lignes parmi les prédictions hors pli, comme le fait le classement public avec le test.
<!-- cellule -->
Depuis l'expérience 07 (early stopping interne), le classement local et le classement public sont presque les mêmes : corrélation de rang 0.98. Le score public est en général 0.2 à 0.3 kg au-dessus du local. L'écart monte à 0.45–0.48 kg avec le lissage par paires : son gain local est plus grand que son gain public.

Dans la simulation, le score d'une soumission varie beaucoup d'un tirage à l'autre (écart-type 0.59 kg), mais la différence entre deux soumissions proches très peu (0.03 kg). Autrement dit, le niveau absolu du score public dépend du tirage, alors que l'écart entre deux soumissions est fiable.

Exemple : la version à huit modèles gagne 0.045 kg en local sur la précédente et 0.036 kg au public ; dans la simulation l'ordre s'inverse dans 4 % des tirages.
<!-- cellule -->
## 6. Bilan

Ce qui a le plus compté : repérer la structure des données (précision de saisie, jumelles, autres résultats du même sportif), moyenner plusieurs graines et réentraîner sur toutes les lignes. Les cibles DOTS et IPF GL complètent bien les modèles en kg. Les variables sans cible ne progressent plus après la troisième série, et les totaux connus servent mieux après les modèles que dedans.

Résultat : 77.775 en local et 78.0 au public pour la première version ; 75.966 / 76.298 pour les huit modèles ; **75.575 / 76.030** avec le lissage par paires.
