def checked_vector(saved, key, size):
    values = saved[key]
    require(values.shape == (size,) and values.dtype.kind in 'fiu' and np.isfinite(values).all(),
            f'{key} doit contenir {size} prédictions numériques finies')
    return values


def checked_predictions(path, expected_indices=None):
    try:
        with np.load(path, allow_pickle=False) as saved:
            test_predictions = checked_vector(saved, 'test', len(test))
            if expected_indices is None:
                return test_predictions
            valid_indices = saved['valid_idx']
            require(valid_indices.dtype.kind in 'iu' and valid_indices.shape == expected_indices.shape
                    and np.array_equal(np.sort(valid_indices), expected_indices),
                    'valid_idx ne correspond pas au pli de validation attendu')
            return valid_indices, checked_vector(saved, 'val', len(expected_indices)), test_predictions
    except Exception as exc:
        raise ValueError(f'Prédictions finales invalides dans {path}: {exc}') from exc


def load_cv(paths, expected_indices):
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for path, indices in zip(paths, expected_indices):
        valid_indices, valid_predictions, test_predictions = checked_predictions(path, indices)
        oof[valid_indices] = valid_predictions
        test_pred += test_predictions / N_FOLDS
    return oof, test_pred


def refit_paths(directory, model, seeds, suffix='full'):
    # L’ordre lexicographique par dossier conserve l’ordre historique des moyennes.
    return sorted(directory / f'{model}__s{seed}__{suffix}.npz' for seed in seeds)


def full_files(model):
    if model == 'tabm':
        return (refit_paths(EXPANDED_SEED_ENSEMBLE_DIR, model, PREV_SEEDS, 'nearfull')
                + refit_paths(OUTPUT_DIR, model, TABM_EXTRA_SEEDS, 'nearfull'))
    if model == MODEL_HISTOGRAM_GRADIENT_BOOSTING:
        return refit_paths(OUTPUT_DIR, model, HGB_FULL_SEEDS, 'nearfull')
    if model == MODEL_CATBOOST_IPF_GL:
        return refit_paths(OUTPUT_DIR, model, GL_FULL_SEEDS)
    files = refit_paths(SIX_MODEL_ENSEMBLE_DIR, model, PREV_SEEDS)
    if model != MODEL_CATBOOST_ABSOLUTE:
        # Les quatre arbres réentraînés dans l’expérience 12 utilisent ces cinq graines.
        files += refit_paths(EXPANDED_SEED_ENSEMBLE_DIR, model, [11, 23, 57, 99, 123])
    return files


ALL_MODELS = [MODEL_LIGHTGBM, MODEL_XGBOOST, MODEL_LIGHTGBM_DOTS, MODEL_CATBOOST_ABSOLUTE, MODEL_CATBOOST_DOTS, 'tabm', MODEL_HISTOGRAM_GRADIENT_BOOSTING, MODEL_CATBOOST_IPF_GL]
cv_files = {}
for model in ALL_MODELS:
    directory = SIX_MODEL_ENSEMBLE_DIR if model in FINAL_MODELS else OUTPUT_DIR
    cv_files[model] = {seed: [directory / f'{model}__s{seed}__fold{k}.npz' for k in range(1, N_FOLDS + 1)]
                       for seed in PREV_SEEDS}
refit_files = {model: full_files(model) for model in ALL_MODELS}
expected_files = [path for runs in cv_files.values() for paths in runs.values() for path in paths]
expected_files += [path for paths in refit_files.values() for path in paths]
missing = [path for npz_path in expected_files for path in (npz_path, npz_path.with_suffix('.json'))
           if not path.is_file()]
if missing:
    missing_list = '\n'.join(f'  - {path.relative_to(ARTIFACTS)}' for path in missing)
    raise FileNotFoundError(f'Export final interrompu : {len(missing)} fichier(s) de prédictions ou de métadonnées manquant(s).\n{missing_list}')

sample_ids = pd.read_csv(DATA_DIR / 'sample_submission.csv', usecols=['id'])['id']
require(len(test) == len(sample_ids) == 45000 and test['id'].is_unique and sample_ids.is_unique
        and set(test['id']) == set(sample_ids), 'Les id de test.csv diffèrent de sample_submission.csv')
fold_indices = {}
for seed in PREV_SEEDS:
    ids = folds_for(seed)
    fold_indices[seed] = [np.flatnonzero(ids == k) for k in range(1, N_FOLDS + 1)]

PER, rows = {}, []
for model in ALL_MODELS:
    loaded = {s: load_cv(paths, fold_indices[s]) for s, paths in cv_files[model].items()}
    oof = np.mean([o for o, _ in loaded.values()], axis=0)
    files = refit_files[model]
    full = np.mean([checked_predictions(p) for p in files], axis=0)
    PER[model] = {'oof': oof, 'full': full}
    rows.append({'model': model, 'cv_seeds': list(loaded), 'oof_seed_avg': rmse(y, oof),
                 'test_from': f'{len(files)} full / 95% refits'})


def nested_nnls(oof_prediction_matrix):
    out = np.zeros(n_train)
    for k in range(1, N_FOLDS + 1):
        fit, va = FOLD_IDS != k, FOLD_IDS == k
        w, _ = nnls(oof_prediction_matrix[fit], y[fit])
        out[va] = oof_prediction_matrix[va] @ w
    return out


NAMES = list(PER)
oof_prediction_matrix = np.column_stack([PER[n]['oof'] for n in NAMES])
WEIGHTS, _ = nnls(oof_prediction_matrix, y)
NESTED = nested_nnls(oof_prediction_matrix)
model_table = pd.DataFrame(rows).set_index('model')
model_table['weight'] = pd.Series(dict(zip(NAMES, WEIGHTS)))
display(model_table.round(3))
model_table.to_csv(OUTPUT_DIR / 'summary_models.csv')

steps = {'Six models from experiment 11': [m for m in NAMES if m in FINAL_MODELS],
         '+ HGB': [m for m in NAMES if m in FINAL_MODELS + [MODEL_HISTOGRAM_GRADIENT_BOOSTING]],
         SUMMARY_FINAL_CANDIDATES: NAMES}
step_rows = [{'blend': k, 'n_models': len(v), 'Local OOF RMSE': rmse(y, nested_nnls(np.column_stack([PER[n]['oof'] for n in v])))}
             for k, v in steps.items() if v]
display(pd.DataFrame(step_rows).set_index('blend').round(3))
BLEND_TEST = {ENSEMBLE_FINAL: np.column_stack([PER[n]['full'] for n in NAMES]) @ WEIGHTS}
print('NNLS Weights: ', dict(zip(NAMES, WEIGHTS.round(3))))
