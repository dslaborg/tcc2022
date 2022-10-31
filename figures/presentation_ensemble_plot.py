import colorsys

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_scores(df):
    rmse_means = []
    rmse_stds = []
    for i in range(1, max(df['ensemble_size']) + 1):
        rmse_means.append(df[df['ensemble_size'] == i]['root_mean_squared_error_mean'].mean())
        rmse_stds.append(df[df['ensemble_size'] == i]['root_mean_squared_error_std'].mean())
    return np.array(rmse_means), np.array(rmse_stds)


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0., min(1., amount * c[1])), c[2])


def main():
    plt.rcParams.update({'font.size': 16})

    # GBERT
    means, stds = get_scores(pd.read_csv('../training/ensemble_scores_gbert.csv', delimiter=";"))
    x_axis = np.arange(1, 61)
    plt.plot(x_axis, means, '-', label='GBERT', color='deepskyblue')
    plt.fill_between(x_axis, means - stds, means + stds, alpha=0.2, linestyle='-', edgecolor='deepskyblue',
                     facecolor='lightskyblue')

    # GPT-2 XL Wechsel without handcrafted features
    means, stds = get_scores(
        pd.read_csv('../training/ensemble_scores_gpt2_xl_wechsel_german_without_handcrafted_features.csv',
                    delimiter=";"))
    x_axis = np.arange(1, 61)
    plt.plot(x_axis, means, linestyle=(0, (5, 5)), label='GPT-2-W w/o read. feat.',
             color=adjust_lightness('darkgreen', 1.0))

    # GPT-2 XL Wechsel
    means, stds = get_scores(pd.read_csv('../training/ensemble_scores_gpt2_xl_wechsel_german.csv', delimiter=";"))
    x_axis = np.arange(1, 61)
    plt.plot(x_axis, means, '-', label='GPT-2-W', color='darkgreen')
    plt.fill_between(x_axis, means - stds, means + stds, alpha=0.2, linestyle='-', edgecolor='darkgreen',
                     facecolor='darkgreen')

    # GPT-2 XL Wechsel + GBERT
    means, stds = get_scores(pd.read_csv('../training/ensemble_scores_gpt2_xl_wechsel_german_gbert.csv', delimiter=";"))
    x_axis = np.arange(2, 61, 2)  # ensemble size n means n gtp2 and n gbert models combined
    means = means[:30]
    stds = stds[:30]
    plt.plot(x_axis, means, '-', label='GPT-2-W + GBERT', color='darkviolet')
    plt.fill_between(x_axis, means - stds, means + stds, alpha=0.2, linestyle='-', edgecolor='darkviolet',
                     facecolor='darkviolet')

    plt.xlabel('ensemble size')
    plt.ylabel('mean of average RMSE scores')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'presentation_ensemble_plot.svg', bbox_inches='tight')


if __name__ == '__main__':
    main()
