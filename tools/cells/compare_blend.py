if KEEP_NEW:
    names = base_names + KEEP_NEW
    ITEMS = {**BASE, **{n: NEW[n] for n in KEEP_NEW}}
    P_all = np.column_stack([ITEMS[n]['oof'] for n in names])
    WEIGHTS_PLUS, _ = nnls(P_all, y)
    NESTED = nested_nnls(P_all)
    BLEND_TEST = {ENSEMBLE_WITH_HISTOGRAM_BOOSTING: np.column_stack([ITEMS[n]['test'] for n in names]) @ WEIGHTS_PLUS}
    print('Weights: ', dict(zip(names, WEIGHTS_PLUS.round(3))))
    print(f'Blend with cross-fitted weights: {rmse(y, base_nested):.3f} → {rmse(y, NESTED):.3f}')
else:
    NESTED, BLEND_TEST = None, None
    print('No new model meets the selection rule; retain the experiment 11 / 12 submission.')
