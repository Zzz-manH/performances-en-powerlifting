# tools

- `build_notebooks.py` assemble les cinq notebooks à partir du code de `cells/` et des textes de `textes/`.
- `cells/*.py` : le code des expériences, une cellule par fichier. `naming_compatibility.py` fait le lien entre les noms courts utilisés dans les fichiers de résultats (`cb_rel`, `ALL3`, `S3`…) et les libellés affichés.
- `textes/*.md` : le texte des notebooks, les cellules étant séparées par `<!-- cellule -->`.
- `pair_smoothing.py` : lissage par paires (expériences 16–17). Appelé par la section 5 du notebook 04, ou directement : `python tools/pair_smoothing.py <racine du projet>`.

Pour régénérer les notebooks :

```bash
python tools/build_notebooks.py /tmp/notebooks
```

Les notebooks générés n'ont pas de sorties. Ceux de la racine ont été exécutés ; générer dans un autre dossier pour ne pas les écraser.

| Préfixe des cellules | Expériences |
|---|---|
| `data_load`, `features_round1`–`3`, `final_frames`, `final_model_defs`, `stage2_twins_s3` | parties communes : données, variables, modèles, jumelles et lissage par groupes |
| `r1_` à `r4_` | 07, 09, 10, 14 : ablations du notebook 02 |
| `zoo_`, `compare_` | 08, 13 : comparaison de modèles du notebook 03 |
| `v1_`, `v2_`, `v4_` | 11, 12, 15 : ensemble final du notebook 04 |
