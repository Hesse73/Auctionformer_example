class Config:
    max_player = 10
    valuation_range = 21
    use_bert = 1
    requires_value = 1
    pretrain_requires_grad = 0
    n_layers = 6
    d_model = 768
    hidden_dim = 512
    mlp_layers = 2
    use_pos_emb = 1
    query_mlp = 1
    from_pretrain = 0
    data_parallel = 0
    model_name = 'bert'
    entry_emb = 4
    softmax_out = 1
    gpu = '0'
    query_style = 'branch_add'
    detach_branch = 1
    test_name = 'auctionformer'
    example = 3
