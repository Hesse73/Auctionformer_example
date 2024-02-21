import numpy as np
import torch

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


class ContextualOMWU:

    def __init__(self,valuation_range,bidding_range,overbid,eta,mechanism):
        self.valuation_range = valuation_range
        self.bidding_range = bidding_range
        self.overbid = overbid
        self.eta = eta
        self.mechanism = mechanism

        if self.overbid:
            self.weights = np.ones([self.valuation_range, self.bidding_range])
        else:
            # set weights where bid > value with 0
            bids = np.arange(0,self.bidding_range)
            values = np.arange(0,self.valuation_range).reshape(-1,1)
            self.weights = np.where(bids <= values, 1.0, 0.0)
        self.last_regrets = np.zeros([self.valuation_range, self.bidding_range])
        self.t = 0

    def take_action(self, value):
        upper_bound = self.bidding_range if self.overbid else value + 1
        valid_weights = self.weights[value][:upper_bound]
        return np.random.choice(np.arange(upper_bound), size=1, p=valid_weights / valid_weights.sum())[0]

    def update(self, agt_value, market_price, market_num, entry_fee):
        self.t += 1

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
        rewards -= entry_fee
        # do not enter -> reward = 0
        rewards[0] = 0
        regrets = rewards.max() - rewards

        # update weights according to OMWU e^{-\eta*\sum_{t=1}^{T-1}u_{ij}^t + 2u_{ij}^{T}}
        optimisitc_sum = 2*regrets + self.last_regrets[agt_value, :upper_bound]
        self.last_regrets[agt_value, :upper_bound] += regrets

        # adapted_eta = self.eta * np.sqrt(1/self.t)
        # self.weights[agt_value][:upper_bound] = np.exp(-adapted_eta * optimisitc_sum)
        # if self.weights[agt_value].sum() == 0:
        #     raise
        # # normalize
        # self.weights[agt_value] /= self.weights[agt_value].sum()
        # if np.isnan(self.weights).any():
        #     raise

        cur_weights = -self.eta * optimisitc_sum
        self.weights[agt_value][:upper_bound] = torch.from_numpy(cur_weights).softmax(dim=-1).numpy()
        if np.isnan(self.weights).any():
            raise OverflowError
