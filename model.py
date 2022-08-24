from torch import nn
from tcn import TemporalConvNet
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        #orginal input shape [batch, time step, input feature] -> after transpose [batch, input feature, time step] -> after model [batch, hidden channel, time step] -> after transpose [batch, time step, hidden channel]
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        # [batch, time step, hidden channel] -> after linear [batch, time step, output feature]
        output = self.linear(output).double()
        return output
