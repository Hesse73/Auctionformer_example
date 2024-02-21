import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import json


if __name__ == '__main__':
    max_player = 3
    lower_value, upper_value = 0, 20
    val_range = upper_value - lower_value + 1
    dataset_num = 8000
    start_from_zero = False
    fix_player = False

    np.random.seed(42)

    dataset = {
        'number': [],
        'uniform_value_hist': [],
        'gaussian_value_hist': [],
    }
    # store data info by: number -> value ranges (length of 2n)
    dataset_info = {player_num: [] for player_num in range(2, max_player + 1)}

    for game_idx in tqdm(range(dataset_num)):
        is_repeated = True
        while is_repeated:
            if fix_player:
                cur_player_num = max_player
            else:
                cur_player_num = np.random.randint(2, max_player + 1)

            cur_uniform_value_hist,cur_gaussian_value_hist = [],[]
            cur_value_ranges = []
            for _ in range(cur_player_num):
                # generate game
                player_val_size = np.random.randint(1, val_range)  # [1,20]
                if start_from_zero:
                    player_lower_value = 0
                else:
                    player_lower_value = np.random.randint(0, upper_value - player_val_size + 1)  # [0,V_max-range]
                player_upper_value = player_lower_value + player_val_size
                player_value_range = [player_lower_value, player_upper_value]  # [a,b]

                # uniform value hist
                uniform_value_hist = np.zeros(val_range)
                uniform_value_hist[player_lower_value:player_upper_value+1] = 1 / (player_val_size+1)
                # gaussian value hist
                player_possible_values = np.arange(player_lower_value, player_upper_value + 1)
                mean, size = (player_lower_value + player_upper_value) / 2, player_val_size
                transform = lambda x: 6 / size * (x - mean)
                tmp = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(
                    transform(player_possible_values - 0.5))
                tmp[0] = norm.cdf(transform(player_lower_value + 0.5))
                tmp[-1] = 1 - norm.cdf(transform(player_upper_value - 0.5))
                gaussian_value_hist = np.zeros(val_range)
                gaussian_value_hist[player_lower_value:player_upper_value+1] = tmp

                cur_uniform_value_hist.append(uniform_value_hist.tolist())
                cur_gaussian_value_hist.append(gaussian_value_hist.tolist())
                cur_value_ranges.append(player_value_range)

            # check repetition
            is_repeated = cur_value_ranges in dataset_info[cur_player_num]
            if is_repeated:
                print('repeated!')

        dataset['number'].append(cur_player_num)
        dataset['uniform_value_hist'].append(cur_uniform_value_hist)
        dataset['gaussian_value_hist'].append(cur_gaussian_value_hist)
        dataset_info[cur_player_num].append(cur_value_ranges)

    postfix = 'zero' if start_from_zero else 'nonzero'
    prefix = 'fix_' if fix_player else ''
    json.dump(dataset, open('{}N={}_V={}_{}_{}games.json'.format(prefix, max_player, upper_value, postfix, dataset_num), 'w'))
    json.dump(dataset_info, open('{}N={}_V={}_{}_{}game_info.json'.format(prefix, max_player, upper_value, postfix, dataset_num), 'w'))
