import random
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import json


def transform_dataset(dataset):
    new_format = {
        'number': [],
        'uniform_value_hist': [],
        'gaussian_value_hist': [],
    }
    for item in dataset:
        new_format['number'].append(item['number'])
        new_format['uniform_value_hist'].append(item['uniform_value_hist'])
        new_format['gaussian_value_hist'].append(item['gaussian_value_hist'])
    return new_format


if __name__ == '__main__':
    max_player = 3
    lower_value, upper_value = 0, 20
    val_range = upper_value - lower_value + 1

    random.seed(42)

    dataset = []

    dataset_num = upper_value ** max_player
    for game_idx in tqdm(range(dataset_num)):
        cur_uniform_value_hist, cur_gaussian_value_hist = [], []
        for i in range(max_player):
            player_upper_value = game_idx % upper_value + 1
            game_idx = int(game_idx / upper_value)
            player_lower_value, player_val_size = 0, player_upper_value
            # uniform value hist
            uniform_value_hist = np.zeros(val_range)
            uniform_value_hist[player_lower_value:player_upper_value + 1] = 1 / (player_val_size + 1)
            # gaussian value hist
            player_possible_values = np.arange(player_lower_value, player_upper_value + 1)
            mean, size = (player_lower_value + player_upper_value) / 2, player_val_size
            transform = lambda x: 6 / size * (x - mean)
            tmp = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(
                transform(player_possible_values - 0.5))
            tmp[0] = norm.cdf(transform(player_lower_value + 0.5))
            tmp[-1] = 1 - norm.cdf(transform(player_upper_value - 0.5))
            gaussian_value_hist = np.zeros(val_range)
            gaussian_value_hist[player_lower_value:player_upper_value + 1] = tmp

            cur_uniform_value_hist.append(uniform_value_hist.tolist())
            cur_gaussian_value_hist.append(gaussian_value_hist.tolist())

        dataset.append({
            'number': max_player,
            'uniform_value_hist': cur_uniform_value_hist,
            'gaussian_value_hist': cur_gaussian_value_hist,
        })

    # shuffle
    random.shuffle(dataset)
    dataset = transform_dataset(dataset)

    json.dump(dataset, open('N={}_V={}_zero_{}games.json'.format(max_player, upper_value, dataset_num), 'w'))
