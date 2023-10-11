import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from Example import Example, example_1, example_2, example_3


def plot_uniform(lower, upper, dir_name):
    x = np.arange(0, 21)
    y = np.zeros(21)
    y[lower:upper + 1] = 1 / (upper - lower + 1)

    plt.bar(x, y)
    plt.ylim((0, 0.2))

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(os.path.join(dir_name, f'Uniform-[{lower},{upper}].pdf'), format="pdf", bbox_inches="tight")
    plt.close()


def plot_gaussian(lower, upper, dir_name):
    # x = np.arange(0, 21, 0.001)
    # mean, sigma = (upper+lower)/2, (upper-lower)/6
    #
    # # plot normal distribution with mean 0 and standard deviation 1
    # plt.plot(x, norm.pdf(x, mean, sigma))

    mean, size = (lower + upper) / 2, upper - lower
    transform = lambda x: 6 / size * (x - mean)
    player_possible_values = np.arange(lower, upper + 1)
    tmp = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(
        transform(player_possible_values - 0.5))
    tmp[0] = norm.cdf(transform(lower + 0.5))
    tmp[-1] = 1 - norm.cdf(transform(upper - 0.5))
    gaussian_value_hist = np.zeros(21)
    gaussian_value_hist[lower:upper + 1] = tmp

    plt.bar(np.arange(21), gaussian_value_hist)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(os.path.join(dir_name, f'Gaussian-[{lower},{upper}].pdf'), format="pdf", bbox_inches="tight")
    plt.close()


def plot_example_distribution(example: Example, dir_name: str):
    if example.dist_type == 'uniform':
        for pid in range(example.player_num):
            plot_uniform(lower=example.lower_values[pid], upper=example.upper_values[pid], dir_name=dir_name)
    else:
        for pid in range(example.player_num):
            plot_gaussian(lower=example.lower_values[pid], upper=example.upper_values[pid], dir_name=dir_name)


if __name__ == '__main__':
    save_dir = './distribution_plot/'
    for example in [example_1, example_2, example_3]:
        plot_example_distribution(example, dir_name=save_dir)
    print('Figures saved at:', save_dir)
