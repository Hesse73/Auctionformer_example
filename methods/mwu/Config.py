class MWUConfig:
    mechanism = 'first_price'
    overbid = False
    agt_values = [100, 50]
    player_num = 2
    valuation_range = [101, 51]
    bidding_range = [101, 51]
    entry_fee = 0
    max_rounds = 100000
    log_freq = 10000
    algo = 'MWU'
    lr = 0.1
    rho = 1
    exponential = 1

class OMWUConfig:
    mechanism = 'first_price'
    overbid = False
    agt_values = [100, 50]
    player_num = 2
    valuation_range = [101, 51]
    bidding_range = [101, 51]
    entry_fee = 0
    max_rounds = 100000
    log_freq = 10000
    algo = 'OWMU'
    eta = 0.1