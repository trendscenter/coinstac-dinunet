import torch.nn as nn


class MSANNet(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, dropout_in=[]):
        super(MSANNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList()

        current_dim = self.in_size
        for i, h_dim in enumerate(hidden_sizes):
            if i in dropout_in:
                layers = [nn.Linear(current_dim, h_dim, bias=False), nn.BatchNorm1d(h_dim, track_running_stats=False),
                          nn.ReLU(),
                          nn.Dropout(p=0.5)]
            else:
                layers = [nn.Linear(current_dim, h_dim, bias=False), nn.BatchNorm1d(h_dim, track_running_stats=False),
                          nn.ReLU()]

            self.layers.append(nn.Sequential(*layers))
            current_dim = h_dim

        self.fc_out = nn.Linear(current_dim, out_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.fc_out(x)