import torch
import random
import numpy as np
import os
from Config import Config
from model import BertValuedSolver
from utils import generate_uniform_value_hist, generate_gaussian_value_hist, generate_value_query, transpose_y0, \
    plot_strategy, mechanism_enc
from loss import calculate_exploit
import time


def solve_player1_strategy(example, dir_name, filename, model_dir='model_ckpt/'):
    configs = Config
    os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    # set seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)

    # load model
    model_name = 'Auctionformer.ckpt'
    model_path = os.path.join(model_dir, model_name)
    model = BertValuedSolver(configs).to(device)
    if configs.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    print('Successfully load model from:', model_path)

    # generate input
    if example.dist_type == 'uniform':
        player_value_hist = generate_uniform_value_hist(configs, example.lower_values, example.upper_values)
    else:
        player_value_hist = generate_gaussian_value_hist(configs, example.lower_values, example.upper_values)
    player_num, mechanism, entry = example.player_num, mechanism_enc(example.mechanism, example.entry), example.entry
    # run with model
    value_dists = np.array(player_value_hist + [np.zeros(configs.valuation_range)] * (configs.max_player - player_num))
    # value_dists = np.array(player_value_hist)
    input_value_dists = torch.from_numpy(value_dists).unsqueeze(0).float().to(device)
    input_mechanism, input_entry = torch.tensor([mechanism]).to(device), torch.tensor([entry]).to(device)
    input_player_num = torch.tensor([player_num]).to(device)
    N, V = input_value_dists.shape[1:]
    player_values = generate_value_query(N, V, device)
    with torch.no_grad():
        start = time.time()
        y0 = model((input_mechanism, input_entry, input_value_dists, player_values))
        y = transpose_y0(configs, y0)
        end = time.time()
        print('using time:', end - start)
        # calculate max exploit
        exploits, _, _ = calculate_exploit(input_mechanism, input_entry, input_player_num,
                                           input_value_dists,
                                           y, device, overbid_punish=False)
        print('p0 Exploit:', exploits[0, 0].cpu())
        # plot strategy
        print('saving strategy of p0')
        plot_strategy(y[0, 0].cpu().numpy(), dir_name, filename, max_value=20)


if __name__ == '__main__':
    import sys

    sys.path.append('../../example_distribution')
    from Example import example_1, example_2, example_3

    np.random.seed(42)

    solve_player1_strategy(example_1, '../../bidding_results/auctionformer/', 'example1.pdf',
                           model_dir='../../model_ckpt/')
    solve_player1_strategy(example_2, '../../bidding_results/auctionformer/', 'example2.pdf',
                           model_dir='../../model_ckpt/')
    solve_player1_strategy(example_3, '../../bidding_results/auctionformer/', 'example3.pdf',
                           model_dir='../../model_ckpt/')
