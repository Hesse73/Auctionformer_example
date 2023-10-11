from example_distribution.Example import example_1, example_2, example_3
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='run_example', choices=['run_example', 'train_mlp', 'show_dist'],
                    help='choose different mode: evaluate on examples, train MLPNet or show example distribution')
parser.add_argument('--example', type=int, nargs='+', default=[2], choices=[1, 2, 3], help='choose example id')
parser.add_argument('--method', type=str, default='auctionformer',
                    choices=['auctionformer', 'mlpnet', 'mwu', 'bne_solver'],
                    help='choose different method to solve')

if __name__ == '__main__':
    args = parser.parse_args()
    example_games = [example_1, example_2, example_3]
    if args.mode == 'run_example':
        # load the corresponding method
        if args.method == 'bne_solver':
            sys.path.append('methods/bne_solver/')
            from methods.bne_solver.run_solver import solve_player1_strategy
        elif args.method == 'mwu':
            sys.path.append('methods/mwu/')
            from methods.mwu.run_simulation import solve_player1_strategy
        elif args.method == 'auctionformer':
            sys.path.append('methods/auctionformer/')
            from methods.auctionformer.evaluate import solve_player1_strategy
        elif args.method == 'mlpnet':
            sys.path.append('methods/mlpnet/')
            from methods.mlpnet.evaluate import solve_player1_strategy
        # solve example game
        for eid in args.example:
            solve_player1_strategy(example=example_games[eid - 1], dir_name=f'bidding_results/{args.method}/',
                                   filename=f'example{eid}.pdf')
    elif args.mode == 'train_mlp':
        sys.path.append('methods/train_mlpnet/')
        from methods.train_mlpnet.main import train

        train('./model_ckpt', './methods/train_mlpnet/dataset/')
    elif args.mode == 'show_dist':
        sys.path.append('example_distribution/')
        from example_distribution.plot_distribution import plot_example_distribution

        save_dir = './example_distribution/distribution_plot/'
        for example in example_games:
            plot_example_distribution(example, dir_name=save_dir)
        print('Figures saved at:', save_dir)
