import numpy as np


class DistValuation:

    def __init__(self, value_hist):
        self.value_hist = value_hist

    def generate_value(self):
        return np.random.choice(len(self.value_hist),1,p=self.value_hist)[0]