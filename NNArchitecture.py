import torch
torch.backends.cudnn.benchmark = False
from torch import nn

class MultipleRegression(nn.Module):
    def __init__(self, input_size, hidden_size,num_var):
        super(MultipleRegression, self).__init__()
        self.input_size = input_size
        dropout = 0.2
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_var),
            nn.ReLU()

        )

    def forward(self, inputs):
        return self.model(inputs)
