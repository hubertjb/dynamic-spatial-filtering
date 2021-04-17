"""Plot results.
"""

import pandas as pd
import seaborn as sns

sns.set(context='talk', style='whitegrid', font_scale=1.2)


def plot_noise_robustness(results, lw=3, ms=12, err_style=None, ci='sd',
                          alpha=0.9, y='bal_acc'):
    """Visualize performance of model against increasing noise strength.

    Parameters
    ----------
    results : pd.DataFrame
        Results of the `evaluate_noise_robustness.py` script. Each row is the
        performance of a single model on the test split given a specified
        noise strength.

    Returns
    -------
    fig, axes
    """
    dsf_type_mapping = {
        'vanilla': 'Vanilla net',
        'dsfd': 'DSFd',
        'dsfm_st': 'DSFm-st'
    }
    results['Model'] = results['dsf_type'].map(dsf_type_mapping)

    col_order = ['no_denoising', 'autoreject', 'data_augm']
    hue_order = ['Vanilla net', 'DSFd', 'DSFm-st']
    if 'acc' in y:
        results[y] *= 100
        if y == 'acc':
            ylabel = 'Accuracy (%)'
        elif y == 'bal_acc':
            ylabel = 'Balanced accuracy (%)'

    gs = sns.relplot(
        data=results, x='noise_strength', y=y, col='denoising', row='dataset',
        hue='Model', markers=True, err_style='bars', ms=ms, lw=lw, alpha=alpha,
        palette='colorblind', marker='o', ci=ci, kind='line',
        col_order=col_order, hue_order=hue_order)

    gs.axes[0, 0].set_ylabel(ylabel)
    for i, col in enumerate(
            ['No denoising', 'Autoreject', 'Data augmentation']):
        gs.axes[0, i].set_title(col)
        gs.axes[0, i].set_xlabel('Noise strength')

    return gs.fig, gs.axes
