import numpy as np


class ContextualMWU():

    def __init__(self,valuation_range,bidding_range,overbid,lr,rho,mechanism,exponential):
        self.valuation_range = valuation_range
        self.bidding_range = bidding_range
        self.overbid = overbid
        self.lr = lr
        self.rho = rho
        self.mechanism = mechanism
        self.exponential = exponential

        if self.overbid:
            self.weights = np.ones([self.valuation_range, self.bidding_range])
        else:
            # set weights where bid > value with 0
            bids = np.arange(0,self.bidding_range)
            values = np.arange(0,self.valuation_range).reshape(-1,1)
            self.weights = np.where(bids <= values, 1.0, 0.0)

    def take_action(self, value):
        upper_bound = self.bidding_range if self.overbid else value + 1
        valid_weights = self.weights[value][:upper_bound]
        return np.random.choice(np.arange(upper_bound), size=1, p=valid_weights / valid_weights.sum())[0]

    def update(self, agt_value, market_price, market_num, entry_fee):
        agt_value = int(agt_value)
        upper_bound = self.bidding_range if self.overbid else agt_value + 1
        virtual_bid = np.arange(upper_bound)
        if self.mechanism == 'second_price':
            rewards = np.where(virtual_bid > market_price, agt_value-market_price,0.0)
        else:
            rewards = np.where(virtual_bid > market_price, agt_value-virtual_bid,0.0)
        # # bid == market price -> estimated reward = (V-p)/N
        # if market_price < upper_bound:
        #     rewards[market_price] = float(agt_value-market_price)/(market_num+1)
        # entry fee
        rewards -= entry_fee
        # do not enter -> reward = 0
        rewards[0] = 0

        if self.exponential:
            pos_rate = (1 + self.lr) ** (rewards / self.rho)
            neg_rate = (1 - self.lr) ** (-rewards / self.rho)
            self.weights[agt_value][:upper_bound] *= np.where(rewards > 0, pos_rate, neg_rate)
        else:
            self.weights[agt_value][:upper_bound] *= (1 + self.lr * rewards)

        # normalize
        self.weights[agt_value] /= self.weights[agt_value].sum()
