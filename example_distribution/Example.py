class Example:

    def __init__(self, mechanism, entry, player_num, lower_values, upper_values, dist_type):
        assert mechanism in ['first', 'second']
        assert dist_type in ['uniform', 'gaussian']
        assert len(lower_values) == player_num
        assert len(upper_values) == player_num
        self.mechanism = mechanism
        self.entry = entry
        self.player_num = player_num
        self.lower_values = lower_values
        self.upper_values = upper_values
        self.dist_type = dist_type


example_1 = Example(
    mechanism='first',
    entry=0,
    player_num=3,
    lower_values=[0, 0, 0],
    upper_values=[20, 10, 10],
    dist_type='gaussian'
)

example_2 = Example(
    mechanism='first',
    entry=3,
    player_num=4,
    lower_values=[0, 5, 5, 5],
    upper_values=[20, 15, 15, 15],
    dist_type='gaussian'
)


example_3 = Example(
    mechanism='first',
    entry=0,
    player_num=3,
    lower_values=[0, 0, 0],
    upper_values=[20, 10, 10],
    dist_type='uniform'
)
