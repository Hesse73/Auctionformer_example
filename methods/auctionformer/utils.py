import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm

import seaborn as sns

sns.set_style('darkgrid')


def mechanism_enc(mechanism, entry):
    encode = {'first': 0, 'second': 1}[mechanism]
    if entry > 0:
        encode += 2
    return encode


def transpose_y0(args, y0):
    if args.softmax_out:
        y = y0.softmax(dim=-1)  # softmax over bids
    else:
        y = torch.abs(y0)  # abs
        y = y / torch.sum(y, dim=-1, keepdim=True)  # normalize

    return y


def generate_gaussian_value_hist(args, lower_values: list, upper_values: list):
    player_value_hist = []
    for player_lower_value, player_upper_value in zip(lower_values, upper_values):
        player_val_size = player_upper_value - player_lower_value
        player_possible_values = np.arange(player_lower_value, player_upper_value + 1)
        mean, size = (player_lower_value + player_upper_value) / 2, player_val_size
        transform = lambda x: 6 / size * (x - mean)
        tmp = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(
            transform(player_possible_values - 0.5))
        tmp[0] = norm.cdf(transform(player_lower_value + 0.5))
        tmp[-1] = 1 - norm.cdf(transform(player_upper_value - 0.5))
        gaussian_value_hist = np.zeros(args.valuation_range)
        gaussian_value_hist[player_lower_value:player_upper_value + 1] = tmp
        player_value_hist.append(gaussian_value_hist)
    return player_value_hist


def generate_uniform_value_hist(args, lower_values: list, upper_values: list):
    player_value_hist = []
    for player_lower_value, player_upper_value in zip(lower_values, upper_values):
        tmp = np.zeros(args.valuation_range)
        player_val_size = player_upper_value - player_lower_value
        tmp[player_lower_value:player_upper_value + 1] = 1 / (player_val_size + 1)
        player_value_hist.append(tmp)
    return player_value_hist


def generate_value_query(N, V, device):
    player_values = torch.arange(V).view(-1, 1).repeat(1, N).flatten()  # (V*N)
    player_values = player_values.unsqueeze(-1) == torch.arange(V)  # (V*N) * V
    return player_values.unsqueeze(0).float().to(device)  # 1*VN*2V


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
    im = ax.imshow(strategy, extent=[0, max_value + 1, 0, max_value + 1], cmap='GnBu')
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
    print('The result computed by Auctionformer has been saved at:', os.path.join(dir_name, filename))
    plt.close()

