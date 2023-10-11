class Config:
    mechanisms = ['first']
    max_player = 3
    valuation_range = 21
    max_entry = 3
    distributions = ['uniform']
    start_from_zero = 1
    dataset_size = 8000
    test_size = 500
    train_enlarge = 10
    test_enlarge = 1
    batch_size = 1024
    lr = 0.0001
    lr_decay = 50
    gpu = '0'
    max_epoch = 300
    only_update_max = 1
    plot_freq = 500
    use_wandb = 0
