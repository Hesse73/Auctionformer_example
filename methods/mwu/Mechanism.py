import numpy as np


def first_price_auction(bid_values, true_values, entry_fee):
    """
    first price auction
    bid_values (N*bid), true_values(N*value) -> utility & revenue
    """
    utility = np.zeros(len(bid_values)) - entry_fee

    max_bidder = bid_values == bid_values.max()
    highest_price = bid_values.max()
    utility[max_bidder] += (true_values[max_bidder] - highest_price)/max_bidder.sum()

    utility[bid_values == 0] = 0

    revenue = highest_price + entry_fee * sum(bid_values != 0)

    return utility,revenue


def second_price_auction(bid_values, true_values, entry_fee):
    """
    second price auction
    bid_values (N*bid), true_values(N*value) -> utility
    """
    utility = np.zeros(len(bid_values)) - entry_fee

    max_bidder = bid_values == bid_values.max()
    second_price = np.partition(bid_values, -2)[-2]
    utility[max_bidder] += (true_values[max_bidder] - second_price) / max_bidder.sum()

    utility[bid_values == 0] = 0

    revenue = second_price + entry_fee * sum(bid_values != 0)

    return utility, revenue

# def first_price_auction(bid_values, entry_fee):
#     """
#     First price auction
#     bid_values (N*bids) -> allocation & payment
#     all stored as numpy array
#     """
#
#     win_bidder = np.random.choice(np.flatnonzero(bid_values == bid_values.max()))
#     allocation = np.arange(len(bid_values)) == win_bidder
#
#     payment = np.zeros(len(bid_values)) + entry_fee
#     payment[win_bidder] += bid_values.max()
#
#     # bid=0 -> do not enter
#     allocation[bid_values == 0] = False
#     payment[bid_values == 0] = 0
#
#     return allocation,payment
#
#
# def second_price_auction(bid_values, entry_fee):
#     """
#     Second price auction
#     bid_values (N*bids) -> allocation & payment
#     all store as numpy array
#     """
#
#     win_bidder = np.random.choice(np.flatnonzero(bid_values == bid_values.max()))
#     allocation = np.arange(len(bid_values)) == win_bidder
#
#     payment = np.zeros(len(bid_values)) + entry_fee
#     payment[win_bidder] += np.partition(bid_values, -2)[-2]
#
#     # bid=0 -> do not enter
#     allocation[bid_values == 0] = False
#     payment[bid_values == 0] = 0
#
#     return allocation,payment
#