import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, w_init_gain="relu", activation=nn.ReLU()):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain=w_init_gain,
            ),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.dropout = dropout

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training)

        enc_output = enc_output.contiguous().transpose(1, 2)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


# class ZoneOutBiLSTM(nn.Module):
#     """ ZoneOut Bi-LSTM """

#     def __init__(self, hidden_dim, zoneout_rate=0.1):
#         super(ZoneOutBiLSTM, self).__init__()

#         self.hidden_dim = hidden_dim

#         # Bi-LSTM
#         self.lstm_cell_forward = ZoneOutCell(
#             nn.LSTMCell(self.hidden_dim, self.hidden_dim), zoneout_rate
#         )
#         self.lstm_cell_backward = ZoneOutCell(
#             nn.LSTMCell(self.hidden_dim, self.hidden_dim), zoneout_rate
#         )
#         self.linear = LinearNorm(self.hidden_dim * 2, self.hidden_dim)

#     def forward(self, x):

#         batch_size, seq_len, device = x.size(0), x.size(1), x.device

#         # Bi-LSTM
#         hs_forward = torch.zeros(batch_size, self.hidden_dim, device=device) # [B, H]
#         cs_forward = torch.zeros(batch_size, self.hidden_dim, device=device) # [B, H]
#         hs_backward = torch.zeros(batch_size, self.hidden_dim, device=device)
#         cs_backward = torch.zeros(batch_size, self.hidden_dim, device=device)

#         # Weights initialization
#         torch.nn.init.kaiming_normal_(hs_forward)
#         torch.nn.init.kaiming_normal_(cs_forward)
#         torch.nn.init.kaiming_normal_(hs_backward)
#         torch.nn.init.kaiming_normal_(cs_backward)

#         forward = []
#         backward = []

#         x = x.view(seq_len, batch_size, -1)

#         # Forward
#         for i in range(seq_len):
#             hs_forward, cs_forward = self.lstm_cell_forward(x[i], (hs_forward, cs_forward))
#             forward.append(hs_forward)

#         # Backward
#         for i in reversed(range(seq_len)):
#             hs_backward, cs_backward = self.lstm_cell_backward(x[i], (hs_backward, cs_backward))
#             backward.append(hs_backward)

#         # Output
#         out = torch.cat([torch.stack(forward), torch.stack(backward)], dim=-1)
#         out = self.linear(out.view(batch_size, seq_len, -1))

#         return out


# class ZoneOutCell(nn.Module):
#     """ZoneOut Cell module.

#     This is a module of zoneout described in
#     `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
#     This code is modified from `eladhoffer/seq2seq.pytorch`_.

#     Examples:
#         >>> lstm = torch.nn.LSTMCell(16, 32)
#         >>> lstm = ZoneOutCell(lstm, 0.5)

#     .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
#         https://arxiv.org/abs/1606.01305

#     .. _`eladhoffer/seq2seq.pytorch`:
#         https://github.com/eladhoffer/seq2seq.pytorch

#     """

#     def __init__(self, cell, zoneout_rate=0.1):
#         """Initialize zone out cell module.

#         Args:
#             cell (torch.nn.Module): Pytorch recurrent cell module
#                 e.g. `torch.nn.Module.LSTMCell`.
#             zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.

#         """
#         super(ZoneOutCell, self).__init__()
#         self.cell = cell
#         self.hidden_size = cell.hidden_size
#         self.zoneout_rate = zoneout_rate
#         if zoneout_rate > 1.0 or zoneout_rate < 0.0:
#             raise ValueError(
#                 "zoneout probability must be in the range from 0.0 to 1.0."
#             )

#     def forward(self, inputs, hidden):
#         """Calculate forward propagation.

#         Args:
#             inputs (Tensor): Batch of input tensor (B, input_size).
#             hidden (tuple):
#                 - Tensor: Batch of initial hidden states (B, hidden_size).
#                 - Tensor: Batch of initial cell states (B, hidden_size).

#         Returns:
#             tuple:
#                 - Tensor: Batch of next hidden states (B, hidden_size).
#                 - Tensor: Batch of next cell states (B, hidden_size).

#         """
#         next_hidden = self.cell(inputs, hidden)
#         next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
#         return next_hidden


#     def _zoneout(self, h, next_h, prob):
#         # apply recursively
#         if isinstance(h, tuple):
#             num_h = len(h)
#             if not isinstance(prob, tuple):
#                 prob = tuple([prob] * num_h)
#             return tuple(
#                 [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
#             )

#         if self.training:
#             mask = h.new(*h.size()).bernoulli_(prob)
#             return mask * h + (1 - mask) * next_h
#         else:
#             return prob * h + (1 - prob) * next_h
