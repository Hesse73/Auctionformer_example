import torch
import random
import numpy as np
import os
from Config import Config
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from utils import get_model_name
from model import MLPSolver
from dataloader import get_train_loader, get_test_loader
from loss import calculate_exploit


def train(model_dir, data_dir):
    args = Config

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    # prepare for model saving
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = get_model_name(args)
    model_path = os.path.join(model_dir, model_name)

    # set seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)

    # get dataloader
    train_loader, test_loader = get_train_loader(args, data_dir), get_test_loader(args, data_dir)

    # set model & optimizer
    model = MLPSolver(args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_decay > 0:
        scheduler = StepLR(optim, step_size=args.lr_decay, gamma=0.5)
    else:
        scheduler = None

    print('model will be saved at:', model_path)

    for epoch in range(1, args.max_epoch + 1):
        print('-------------------------------------------------------')
        model.train()
        train_loss = 0.0
        train_bar = tqdm(enumerate(train_loader))
        for b_id, X in train_bar:
            # B, B, B, B*N*V
            cur_player_num, cur_mechanism, cur_entry, value_dists = (data.to(device) for data in X)
            value_dists = value_dists.float()

            # B*N*V*V
            y = model((cur_mechanism, cur_entry, value_dists))

            # player's exploits: B*N
            exploits = calculate_exploit(cur_mechanism, cur_entry, cur_player_num, value_dists, y, device,
                                         overbid_punish=True)

            # loss = max exploit or sum exploit
            if args.only_update_max:
                max_exploits, _ = exploits.max(dim=-1)
                loss = max_exploits.mean()
            else:
                loss = exploits.mean()

            loss.backward()
            optim.step()
            optim.zero_grad()

            # show results
            loss_value = loss.cpu().item()
            train_loss += loss_value
            avg_train_loss = train_loss / (b_id + 1)
            train_bar.set_description("Epoch %d Avg Loss %.5f" % (epoch, avg_train_loss))
        if scheduler is not None:
            scheduler.step()

        # test
        model.eval()
        test_eps = [0, 0, 0, 0, 0]
        with torch.no_grad():
            for X in test_loader:
                cur_player_num, cur_mechanism, cur_entry, value_dists = (data.to(device) for data in X)
                value_dists = value_dists.float()

                # predicted strategy
                y = model((cur_mechanism, cur_entry, value_dists))
                # random strategy
                random_y = torch.rand_like(y).softmax(dim=-1)
                # zero strategy
                zero_y = torch.zeros_like(y)  # B*N*V*V
                zero_y[:, :, :, 0] = 1
                # truthful strategy
                B, N, V = y.shape[:3]
                truthful_y = torch.arange(V).view(-1, 1) == torch.arange(V)  # V*V
                truthful_y = truthful_y.float().to(device).view(1, 1, V, V).repeat(B, N, 1, 1)
                # trivial strategy
                trivial_bid = (((cur_player_num - 1) / cur_player_num).view(-1, 1).repeat(1, V) * torch.arange(V).to(
                    device)).int()  # B*V
                trivial_y = trivial_bid.unsqueeze(-1) == torch.arange(V).to(device)  # B*V*V
                trivial_y = trivial_y.float().view(B, 1, V, V).repeat(1, N, 1, 1)  # B*N*V*V

                for idx, strategy in enumerate([y, random_y, zero_y, truthful_y, trivial_y]):
                    exploits = calculate_exploit(cur_mechanism, cur_entry, cur_player_num, value_dists, strategy,
                                                 device, overbid_punish=False)
                    max_exploits, _ = exploits.max(dim=-1)
                    test_eps[idx] += max_exploits.sum().cpu().item()
        predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = (eps / len(test_loader.dataset) for eps in
                                                                        test_eps)
        print(
            'Eps on test data: [Predict | Random | Zero | Truthful | Trivial] = [%.5f | %.5f | %.5f | %.5f | %.5f]' % (
                predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
    # save model
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    train('../../model_ckpt/', './dataset')
