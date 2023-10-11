import matplotlib.pyplot as plt
import os
from fpa_bne import *
from scipy.stats import norm


def generate_value_hist(lower_values: list, upper_values: list, dist_type='uniform'):
    player_value_hist = []
    for player_lower_value, player_upper_value in zip(lower_values, upper_values):
        if dist_type == 'uniform':
            player_val_size = player_upper_value - player_lower_value
            tmp = np.ones(player_val_size + 1) * (1 / (player_val_size + 1))
            player_value_hist.append(tmp)
        else:
            player_val_size = player_upper_value - player_lower_value
            player_possible_values = np.arange(player_lower_value, player_upper_value + 1)
            mean, size = (player_lower_value + player_upper_value) / 2, player_val_size
            transform = lambda x: 6 / size * (x - mean)
            tmp = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(
                transform(player_possible_values - 0.5))
            tmp[0] = norm.cdf(transform(player_lower_value + 0.5))
            tmp[-1] = 1 - norm.cdf(transform(player_upper_value - 0.5))
            player_value_hist.append(tmp)
    return player_value_hist


def solve_player1_strategy(example, dir_name, filename, violin_plot = False):
    assert example.entry == 0
    assert example.mechanism == 'first'
    bidder_values = [np.arange(example.lower_values[i], example.upper_values[i] + 1) for i in range(example.player_num)]
    bidder_probs = generate_value_hist(example.lower_values, example.upper_values, example.dist_type)

    bidders = [
        Bidder(bidder_values[i], bidder_probs[i]) for i in range(len(bidder_values))
    ]

    # Compute bidding strategies with the solver
    compute_bidding_stategies(bidders, 0.0, tol=1e-8)
    min_winning_bid = compute_min_winning_bid(bidders)

    # print('Min winning bid: {}'.format(min_winning_bid))
    # print('Max winning bid: {}'.format(bidders[0].strategy.F_jump_points[-1][0]))

    bid_range = np.linspace(max(bidders[0].strategy.F_jump_points[0][0],0),
                            bidders[0].strategy.F_jump_points[-1][0],
                            num=10000)

    # Compute pdf
    min_value, max_value = min(bidder_values[0]), max(bidder_values[0])
    pdf_per_v = {v: {'bid': [], 'pdf': []} for v in range(min_value, max_value + 1)}
    for j in range(len(bid_range)):
        cdf, pdf, values = prob_dist(bidders, bid_range[j])
        if len(values) > 0:
            pdf_per_v[values[0]]['bid'].append(bid_range[j])
            pdf_per_v[values[0]]['pdf'].append(pdf[0])

    # sample bid from distribution
    sample_per_v = []
    for v in range(min_value, max_value + 1):
        bids = pdf_per_v[v]['bid']
        pdf = pdf_per_v[v]['pdf']
        if len(pdf) == 0 and v == 0:
            samples = [0]
        else:
            samples = np.random.choice(bids, size=1000, p=[prob / sum(pdf) for prob in pdf])
        sample_per_v.append(samples)

    fig, ax = plt.subplots()
    ax.spines['bottom'].set_color('#555')
    ax.spines['top'].set_color('#555')
    ax.spines['right'].set_color('#555')
    ax.spines['left'].set_color('#555')
    if violin_plot:
        # violin plot of solved distribution at each value
        plt.violinplot(sample_per_v, np.arange(min_value, max_value + 1), showmeans=True, showmedians=False)
        plt.ylim(-0.2, max_value)
        plt.grid(axis='both', color='0.95')
        plt.xlabel('Valuation')
        plt.ylabel('Bid (distribution)')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(os.path.join(dir_name, filename), format="pdf", bbox_inches="tight")
        plt.close()
    else:
        # plot the mean of each bidding distribution at every value
        plt.plot(np.arange(min_value, max_value + 1), [sum(bids)/len(bids) for bids in sample_per_v], 's-')
        plt.ylim((0,20))
        plt.grid(axis='both', color='0.95')
        plt.xlabel('Valuation')
        plt.ylabel('Bid')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(os.path.join(dir_name, filename), format="pdf", bbox_inches="tight")
        plt.close()
    print('The result computed by bne_solver has been saved at:', os.path.join(dir_name, filename))


if __name__ == '__main__':
    import sys
    sys.path.append('../../example_distribution')
    from Example import example_1, example_2, example_3

    solve_player1_strategy(example_1, '../../bidding_results/bne_solver/', 'example1.pdf')
    solve_player1_strategy(example_2, '../../bidding_results/bne_solver/', 'example2.pdf')

    # # run this will cause exception
    # solve_player1_strategy(example_3, 'example3.pdf')
