from Algorithm import ContextualMWU


class BiddingAgent:

    def __init__(self,args,agt_id):
        self.valuation_range = args.valuation_range[agt_id]
        self.bidding_range = args.bidding_range[agt_id]
        self.overbid = args.overbid
        self.mechanism = args.mechanism
        self.agt_id = agt_id

        self.algorithm = ContextualMWU(self.valuation_range,self.bidding_range,self.overbid,args.lr,args.rho,
                                       self.mechanism, args.exponential)

        self.last_action = None
        self.last_state = None

    def generate_action(self,state,test=False):
        action = self.algorithm.take_action(value=state)

        if not test:
            # not testing -> save last time (s,a)
            self.last_state = state
            self.last_action = action

        return action
        
    def update_policy(self,last_bid,entry_fee):
        others_bid = last_bid.copy()
        others_bid[self.agt_id] = -1
        market_price = others_bid.max()
        market_num = (others_bid == market_price).sum()

        self.algorithm.update(agt_value=self.last_state,market_price=market_price,
                              market_num=market_num,entry_fee=entry_fee)

    def normalized_weights(self):
        weights = self.algorithm.weights.copy()
        for value in range(self.valuation_range):
            weights[value] = weights[value]/weights[value].sum()
        return weights