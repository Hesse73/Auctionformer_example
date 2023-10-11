import numpy as np
from scipy.stats import norm


def generate_gaussian_value_hist(lower_values: list, upper_values: list):
    player_value_hist = []
    for pid, (player_lower_value, player_upper_value) in enumerate(zip(lower_values, upper_values)):
        player_val_size = player_upper_value - player_lower_value
        player_possible_values = np.arange(player_lower_value, player_upper_value + 1)
        mean, size = (player_lower_value + player_upper_value) / 2, player_val_size
        transform = lambda x: 6 / size * (x - mean)
        tmp = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(
            transform(player_possible_values - 0.5))
        tmp[0] = norm.cdf(transform(player_lower_value + 0.5))
        tmp[-1] = 1 - norm.cdf(transform(player_upper_value - 0.5))
        gaussian_value_hist = np.zeros(player_upper_value + 1)
        gaussian_value_hist[player_lower_value:player_upper_value + 1] = tmp
        player_value_hist.append(gaussian_value_hist)
    return player_value_hist


def generate_uniform_value_hist(lower_values: list, upper_values: list):
    player_value_hist = []
    for pid, (player_lower_value, player_upper_value) in enumerate(zip(lower_values, upper_values)):
        tmp = np.zeros(player_upper_value + 1)
        player_val_size = player_upper_value - player_lower_value
        tmp[player_lower_value:player_upper_value + 1] = 1 / (player_val_size + 1)
        player_value_hist.append(tmp)
    return player_value_hist