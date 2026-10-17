if KEEP_NEW:
    names = base_names + KEEP_NEW
    ITEMS = {**BASE, **{n: NEW[n] for n in KEEP_NEW}}
    P_all = np.column_stack([ITEMS[n]['oof'] for n in names])
    WEIGHTS_PLUS, _ = nnls(P_all, y)
    NESTED = nested_nnls(P_all)
    BLEND_TEST = {'v3_plus': np.column_stack([ITEMS[n]['test'] for n in names]) @ WEIGHTS_PLUS}
    print('权重：', dict(zip(names, WEIGHTS_PLUS.round(3))))
    print(f'嵌套融合：{rmse(y, base_nested):.3f} → {rmse(y, NESTED):.3f}')
else:
    NESTED, BLEND_TEST = None, None
    print('没有新模型达到门槛，最终提交保持 11 / 12 的版本。')
