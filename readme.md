# Run Examples

## Fetch trained model

The Auctionformer model's checkpoint is saved on an [anonymous google drive](https://drive.google.com/file/d/1JBOHIdjlhM9EpoIfwyF6NLveHSUQ3K7B/view?usp=drive_link), please download it and save it in `./model_ckpt/` before you run the model.
The MLPNet model's checkpoint file is already provided in `./model_ckpt/`

## Example Settings

The three examples in our paper are:
- Example 1: First price auction : Bidder 1's value distribution F1=G(10,10/3) and Bidder 2 and 3 is F2=F3=G(5,5/3)
- Example 2: First price auction + Entry Fee=3 : Bidder 1's value distribution F1=G(10,10/3), and Bidder 2-4's value distribution is F2=F3=F4=G(10,5/3)
- Example 3: First price auction : Bidder 1's value distribution is F1=U\[0,20\] and Bidder 2,3's value distribution is F2=U\[0,10\]

You can check the histogram of the value distributions described above by running:
```shell
python3 main.py --mode show_dist
```
The plotted results will be saved in `./example_distribution/distribution_plot/`.

All the 3 examples are configured in file `./example_distribution/Example.py`, so you can modify it to run different example settings.


## Run Numerical BNE Solver
Here we incorporate the BNE solver proposed in [\[1\]](#bne_solver)<!-- @IGNORE PREVIOUS: anchor -->, and the solver's code is adopted from this
[GitHub Repo](https://github.com/shen-weiran/discrete_fpa_bne).

Run example 1,3 with:
```shell
python3 main.py --mode run_example --example 1 3 --method bne_solver
```
The results will be saved in `./bidding_results/bne_solver/`


Since the solver does not support entrance fee in first-price auction, it cannot compute the solution of example 2.

## Run MWU
The MWU algorithm we are using is described in [\[2\]](#MWU)<!-- @IGNORE PREVIOUS: anchor -->.

Run with:
```shell
python3 main.py --mode run_example --example 1 2 3 --method mwu
```
The results will be saved in `./bidding_results/mwu/`


## Run MLPNet

> We've already trained the MLPNet with 3-player dataset and saved it at `./model_ckpt/MLPNet.ckpt`.
>
> The detailed MLPNet model's training codes are located in methods/train_mlp. is you want to train a new model checkpoint

We've already trained the MLPNet with 3-player dataset and saved it at `./model_ckpt/MLPNet.ckpt`.
So you can directly run example 1,3 using the trained model with:
``` shell
python3 main.py --mode run_example --example 1 3 --method mlpnet
```

As described in our paper, the MLPNet's network input size is fixed given the number of players,
so the 3-player model cannot be adapted to a 4-player game,
thus it cannot run on example 2. (To force run `python3 main.py --mode run_example --example 2 --method mlpnet` will throw a PyTorch matrix mismatch error)

If you want to train a new MLPNet model, the detailed training codes are located in `./methods/train_mlpnet/`.
And you can train it by running:
```shell
python3 main.py --mode train_mlp
```

## Run Auctionformer
We've already trained the Auctionformer model and saved it at `./model_ckpt/Auctionformer.ckpt`.
So you can run example 1,2,3 using the trained model with:
``` shell
python3 main.py --mode run_example --example 1 2 3 --method auctionformer
```

##  Reference

<a name="bne_solver">\[1\]</a> Zihe Wang, Weiran Shen, and Song Zuo. 2020. Bayesian Nash Equilibrium in
First-Price Auction with Discrete Value Distributions. In Proceedings of the 19th
International Conference on Autonomous Agents and MultiAgent Systems (Auckland, New Zealand) (AAMAS ’20).

<a name="MWU">\[2\]</a> Sanjeev Arora, Elad Hazan, and Satyen Kale. 2012. The Multiplicative Weights
Update Method: a Meta-Algorithm and Applications. Theory of Computing 8, 6
(2012), 121–164.
