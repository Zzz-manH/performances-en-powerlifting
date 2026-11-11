def load_seed(model, seed):
    oof, test_pred = np.full(n_train, np.nan), np.zeros(len(test))
    for fold in range(1, N_FOLDS + 1):
        path = OUTPUT_DIR / f'{model}__s{seed}__fold{fold}.npz'
        if not path.is_file():
            return None
        saved = np.load(path)
        oof[saved['valid_idx']] = saved['val']
        test_pred += saved['test'] / N_FOLDS
    require(not np.isnan(oof).any(), f'{model} seed {seed} has incomplete OOF predictions')
    return oof, test_pred


def fold_rmse(pred, ids=FOLD_IDS):
    return pd.Series({k: rmse(y[ids == k], pred[ids == k]) for k in range(1, N_FOLDS + 1)})


PER_MODEL = {}
rows = []
for model in FINAL_MODELS:
    loaded = {s: load_seed(model, s) for s in SEEDS}
    loaded = {s: v for s, v in loaded.items() if v is not None}
    if not loaded:
        continue
    oof = np.mean([o for o, _ in loaded.values()], axis=0)
    fold_avg = np.mean([t for _, t in loaded.values()], axis=0)
    fulls = [np.load(OUTPUT_DIR / f'{model}__s{s}__full.npz')['test'] for s in loaded
             if (OUTPUT_DIR / f'{model}__s{s}__full.npz').is_file()]
    full = np.mean(fulls, axis=0) if fulls else fold_avg          # No full TabM refit here; use fold-averaged predictions.
    PER_MODEL[model] = {'oof': oof, 'fold_avg': fold_avg, 'full': full}
    rows.append({'model': model, 'seeds': list(loaded), 'full_refits': len(fulls),
                 **{f'oof_seed{s}': rmse(y, o) for s, (o, _) in loaded.items()},
                 'oof_seed_avg': rmse(y, oof)})
model_table = pd.DataFrame(rows).set_index('model').sort_values('oof_seed_avg')
display(model_table.round(3))
model_table.to_csv(OUTPUT_DIR / 'summary_models.csv')

NAMES = list(PER_MODEL)
oof_prediction_matrix = np.column_stack([PER_MODEL[n]['oof'] for n in NAMES])
WEIGHTS, _ = nnls(oof_prediction_matrix, y)
NESTED = np.zeros(n_train)
for k in range(1, N_FOLDS + 1):                       # Fit weights on four groups of existing OOF, evaluate on the fifth; this is not full nested CV.
    fit, va = FOLD_IDS != k, FOLD_IDS == k
    w_k, _ = nnls(oof_prediction_matrix[fit], y[fit])
    NESTED[va] = oof_prediction_matrix[va] @ w_k
BLEND_TEST = {kind: np.column_stack([PER_MODEL[n][kind] for n in NAMES]) @ WEIGHTS for kind in ['fold_avg', 'full']}
BLEND_TEST['hybrid'] = 0.5 * (BLEND_TEST['fold_avg'] + BLEND_TEST['full'])
print('NNLS Weights: ', dict(zip(NAMES, WEIGHTS.round(3))), '| sum of weights', round(float(WEIGHTS.sum()), 4))
print(f'OOF blend: in-sample weights {rmse(y, oof_prediction_matrix @ WEIGHTS):.3f}, cross-fitted weights {rmse(y, NESTED):.3f}')
print('Mean absolute differences between test variants: ',
      {f'{a}-{b}': round(float(np.abs(BLEND_TEST[a] - BLEND_TEST[b]).mean()), 2)
       for a, b in [('fold_avg', 'full'), ('fold_avg', 'hybrid'), ('full', 'hybrid')]})
pd.DataFrame({'model': NAMES, 'weight': WEIGHTS}).to_csv(OUTPUT_DIR / 'summary_weights.csv', index=False)
