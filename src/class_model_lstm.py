import torch

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size,
                                  hidden_size,
                                  num_layers,
                                  batch_first=True,
                                  dropout=dropout,
                                  dtype=torch.float64)
        self.linear = torch.nn.Linear(hidden_size,
                                      output_size,
                                      dtype=torch.float64)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out