import numpy as np
from Config import Config
from Env import PrivateAuction
from Plot import plot_strategy
from utils import generate_gaussian_value_hist, generate_uniform_value_hist


def solve_player1_strategy(example, dir_name, filename):
    configs = Config
    # load example
    configs.mechanism = 'first_price' if example.mechanism == 'first' else 'second_price'
    configs.entry_fee = example.entry
    configs.player_num = example.player_num
    configs.agt_values = example.upper_values
    configs.valuation_range = configs.bidding_range = [v+1 for v in example.upper_values]
    if example.dist_type == 'uniform':
        player_value_hist = generate_uniform_value_hist(example.lower_values, example.upper_values)
    else:
        player_value_hist = generate_gaussian_value_hist(example.lower_values, example.upper_values)
    env = PrivateAuction(configs, player_value_hist)
    print('Running MWU algorithm in auction simulation environment...')
    # set seed
    np.random.seed(42)
    env.run()
    plot_strategy(env.export_p0_strategy(), dir_name, filename, max_value=example.upper_values[0])




if __name__ == '__main__':
    import sys
    sys.path.append('../../example_distribution')
    from Example import example_1, example_2, example_3

    np.random.seed(42)

    solve_player1_strategy(example_1, '../../bidding_results/mwu/', 'example1.pdf')
    solve_player1_strategy(example_2, '../../bidding_results/mwu/', 'example2.pdf')
    solve_player1_strategy(example_3, '../../bidding_results/mwu/', 'example3.pdf')

