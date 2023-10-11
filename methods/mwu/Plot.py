import matplotlib.pyplot as plt
import os
import numpy as np

import seaborn as sns
sns.set_style('darkgrid')


def plot_strategy(strategy, dir_name, filename='bid_strategy.pdf', max_value=20):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # remove too small values
    strategy[strategy < 1e-4] = 0
    # rotate
    strategy = np.flip(strategy.T, axis=0)

    V, B = strategy.shape
    # plot heatmap with ticks
    fig, ax = plt.subplots()
    im = ax.imshow(strategy, extent=[0, max_value+1, 0, max_value+1], cmap='GnBu')
    ax.set_xticks(np.arange(V))
    ax.set_yticks(np.arange(B))
    ax.spines['bottom'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['right'].set_color('grey')
    ax.spines['left'].set_color('grey')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('bid probability', rotation=-90, va='bottom')

    # sns.heatmap(strategy, annot=False, cmap="crest", cbar=True)

    plt.ylabel('bid')
    plt.xlabel('valuation')

    plt.savefig(os.path.join(dir_name, filename), format="pdf", bbox_inches="tight")
    print('The result computed by MWU has been saved at:', os.path.join(dir_name, filename))
    plt.close()
