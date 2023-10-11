import numpy as np
from Mechanism import first_price_auction,second_price_auction
from Valuation import DistValuation
from Agent import BiddingAgent

class PrivateAuction():

    def __init__(self, args, agt_value_hist):
        # player num
        self.player_num = args.player_num
        # set agents
        self.agt_list = [BiddingAgent(args, agt_id=i) for i in range(self.player_num)]
        # game rounds
        self.max_rounds = args.max_rounds
        # save args
        self.args = args
        # set mechanism function
        self.mechanism = args.mechanism
        if self.mechanism == 'first_price':
            self.auction_func = first_price_auction
        elif self.mechanism == 'second_price':
            self.auction_func = second_price_auction
        else:
            raise NotImplementedError(f"Unknown mechanism: {self.mechanism}")
        # entry fee
        self.entry_fee = args.entry_fee
        # valuation
        self.agt_value_hist = agt_value_hist
        self.agent_valuations = [DistValuation(agt_value_hist[i]) for i in range(self.player_num)]
        # logging
        self.log_freq = args.log_freq

    def run(self):
        # results records
        agt_rewards = [[] for i in range(self.player_num)]
        revenue_list = []
        # record per round
        last_bids = None
        for round in range(self.max_rounds+1):
            bid_values, true_value = np.zeros(self.player_num,dtype=int), np.zeros(self.player_num,dtype=float)
            for agt_idx, agt in enumerate(self.agt_list):
                # 1. generate current state
                cur_state = self.agent_valuations[agt_idx].generate_value()
                true_value[agt_idx] = cur_state
                # 2. update with bidding of previous round
                if last_bids is not None:
                    agt.update_policy(last_bids,self.entry_fee)
                # 3. generate action for current round
                action = agt.generate_action(state=cur_state)
                bid_values[agt_idx] = action

            last_bids = bid_values.copy()

            # submit agents' bidding value to mechanism and get reward & revenue (for plot)
            utility, revenue = self.auction_func(bid_values=bid_values, true_values=true_value, entry_fee=self.entry_fee)
            for idx in range(self.player_num):
                agt_rewards[idx].append(utility[idx])
            revenue_list.append(revenue)

            # logging
            if round != 0 and round % self.log_freq == 0:
                recent_rewards = [sum(agt_rewards[i][-self.log_freq:])/self.log_freq for i in range(self.player_num)]
                recent_revenue = sum(revenue_list[-self.log_freq:])/self.log_freq
                print(f'Round {round}: rewards=', recent_rewards, 'revenue=', recent_revenue)

    def export_p0_strategy(self):
        return self.agt_list[0].normalized_weights()
