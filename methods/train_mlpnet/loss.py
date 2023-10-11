import torch


def calculate_exploit(mechanism, entry, player_num, valuation_dist, bidding_strategy, device, overbid_punish=True):
    """

        Args:
            mechanism: B
            entry: B
            player_num: B
            valuation_dist: B*N*V
            bidding_strategy: B*N*V*V
            device: torch device

        Returns: player's exploit (B*N)

    """
    B, N, V = valuation_dist.shape

    ####################################
    # 1. get market price distribution #
    ####################################
    # valid mask: B*N or B*N*V
    valid_player_mask_bn = player_num.view(-1, 1) > torch.arange(N).to(device)
    valid_player_mask_bnv = valid_player_mask_bn.unsqueeze(-1).repeat(1, 1, V)

    # get each player's marginal bid (B*N*1*V @ B*N*V*V => B*N*1*V) --> B*N*V
    marginal_bid = torch.matmul(valuation_dist.unsqueeze(-2), bidding_strategy).squeeze(-2)
    cumulative_bid = torch.cumsum(marginal_bid, dim=-1)  # B*N*V
    # set invalid player with zero bid (i.e. cum_bid = [1,1,1,1,...])
    cumulative_bid[~valid_player_mask_bnv] = 1

    # get each player's other players' bid
    # B*N*V --> B*1*N*V --> repeat to B*N*N*V
    others_cum_bid = cumulative_bid.unsqueeze(1).repeat(1, N, 1, 1)
    # set self with zero bid (i.e. cum_bid = [1,1,1,1,...])
    self_mask = (torch.arange(N).view(-1, 1) == torch.arange(N)).view(1, N, N, 1).repeat(B, 1, 1, V).to(
        device)  # B*N*N*V
    others_cum_bid[self_mask] = 1

    # market price cdf for each bidder, prod on other player's dim (B*N*N*V -- > B*N*V)
    market_cdf = torch.prod(others_cum_bid, dim=-2)

    # market price pdf (B*N*V)
    tmp = torch.zeros_like(market_cdf)
    tmp[:, :, 1:] = market_cdf[:, :, :-1]
    market_pdf = market_cdf - tmp

    ###############################
    # 2. calculate utility matrix #
    ###############################
    # utility matrix V*V*V (given different value|market|bid)
    value = torch.arange(V).view(-1, 1, 1).repeat(1, V, V).float().to(device)
    market = torch.arange(V).view(1, -1, 1).repeat(V, 1, V).float().to(device)
    bid = torch.arange(V).view(1, 1, -1).repeat(V, V, 1).float().to(device)

    fp_utility_v_m_b = (value - bid) * (bid > market)
    sp_utility_v_m_b = (value - market) * (bid > market)

    # batched utility (B*V*V*V)
    fp_utility = fp_utility_v_m_b.repeat(B, 1, 1, 1)
    sp_utility = sp_utility_v_m_b.repeat(B, 1, 1, 1)

    # mechanism
    is_first = (mechanism % 2 == 0).view(-1, 1, 1, 1).repeat(1, V, V, V)

    # entrance fee:
    entries = entry.view(-1, 1, 1, 1).repeat(1, V, V, V)
    entries[:, :, :, 0] = 0  # bid=0's entry is 0

    # batched reward matrix, given different mechanism & entry (B*V*V*V)
    utility_v_m_b = (fp_utility * is_first + sp_utility * (~is_first)) - entries

    # overbid punishment
    if overbid_punish:
        is_overbid = bid + entries > value
        utility_v_m_b[is_overbid] = -10
    #################################
    # 3. calculate expected utility #
    #################################
    # expectation on each player's market price (each player's expected utility under different value & bid)
    utility_v_b = torch.matmul(market_pdf.view(B, N, 1, 1, V), utility_v_m_b.view(B, 1, V, V, V))  # B*N*V*1*V
    utility_v_b = utility_v_b.squeeze(-2)  # B*N*V*V

    # calculate current policy's expected utility on each player's value (B*N*V)
    utility_v = (bidding_strategy * utility_v_b).sum(dim=-1)

    # each player's optimal utility under different value (B*N*V)
    opt_utility_v, _ = utility_v_b.max(dim=-1)

    # expectation on each [valid] player's valuation distribution (B*N)
    utility = valid_player_mask_bn * (valuation_dist * utility_v).sum(dim=-1)
    opt_utility = valid_player_mask_bn * (valuation_dist * opt_utility_v).sum(dim=-1)

    return opt_utility - utility
