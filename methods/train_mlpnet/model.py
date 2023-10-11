import torch
import torch.nn as nn


class MLPSolver(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.max_player = configs.max_player
        self.valuation_range = configs.valuation_range
        self.max_entry = configs.max_entry

        self.emb_size = 128
        self.mechanism_embedding = nn.Embedding(4, self.emb_size)
        self.entrance_embedding = nn.Embedding(self.max_entry + 1, self.emb_size)

        input_size = self.max_player * self.valuation_range + self.emb_size * 2
        output_size = self.max_player * self.valuation_range * self.valuation_range

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024, affine=False), nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        # input: (mechanism:B, entry:B, value_dists:B*N*V)
        mechanism, entry, value_dists = x

        mechanism = self.mechanism_embedding(mechanism)  # B*emb
        entry = self.entrance_embedding(entry)  # B*emb
        value_dists = value_dists.flatten(start_dim=1)  # B*(N*V)
        inputs = torch.cat((mechanism, entry, value_dists), dim=-1)
        y = self.model(inputs)  # B*(N*V*V)
        y = y.view(y.shape[0], self.max_player, self.valuation_range, self.valuation_range)  # B*N*V*V
        y = y.softmax(dim=-1)

        return y
