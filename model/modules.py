import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .blocks import (
    ZoneOutBiLSTM,
    LinearNorm,
    ConvBlock,
)
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    """ Text Encoder """

    def __init__(self, config):
        super(TextEncoder, self).__init__()

        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        d_model = config["transformer"]["encoder_hidden"]
        kernel_size = config["transformer"]["encoder_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]
        zoneout = config["transformer"]["encoder_zoneout"]

        self.d_model = d_model
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0
        )

        self.conv_stack = nn.ModuleList(
            [
                ConvBlock(
                    d_model, d_model, kernel_size=kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.lstm = ZoneOutBiLSTM(
            d_model, zoneout_rate=zoneout
        )

    def forward(self, src_seq, mask=None):

        enc_output = self.src_word_emb(src_seq)

        for conv in self.conv_stack:
            enc_output = conv(enc_output, mask=mask)

        enc_output = self.lstm(enc_output)

        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0.)

        return enc_output


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = DurationPredictor(model_config)
        self.gaussian_upsampling = GaussianUpsampling(model_config)

    def forward(
        self,
        x,
        src_mask,
        duration_target=None,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x, attn = self.gaussian_upsampling(x, duration_target, src_mask)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, attn = self.gaussian_upsampling(x, duration_rounded, src_mask)

        return (
            x,
            log_duration_prediction,
            duration_rounded,
            attn,
        )


class GaussianUpsampling(nn.Module):
    """ Gaussian Upsampling """

    def __init__(self, model_config):
        super(GaussianUpsampling, self).__init__()
        # self.range_param_predictor = RangeParameterPredictor(model_config)

    def forward(self, encoder_outputs, duration, mask):
        device = encoder_outputs.device

        # range_param = self.range_param_predictor(encoder_outputs, duration, mask)

        t = torch.sum(duration, dim=-1, keepdim=True) #[B, 1]

        e = torch.cumsum(duration, dim=-1).float() #[B, L]
        c = e - 0.5 * duration #[B, L]
        t = torch.arange(1, torch.max(t).item()+1, device=device) # (1, ..., T)
        t = t.unsqueeze(0).unsqueeze(1) #[1, 1, T]
        c = c.unsqueeze(2)

        # print(range_param, 0.1*(range_param ** 2))

        # w_1 = torch.exp(-0.1*(range_param.unsqueeze(-1) ** -2) * (t - c) ** 2)  # [B, L, T]
        # w_2 = torch.sum(torch.exp(-0.1*(range_param.unsqueeze(-1) ** -2) * (t - c) ** 2), dim=1, keepdim=True)  # [B, 1, T]
        w_1 = torch.exp(-0.1 * (t - c) ** 2)  # [B, L, T]
        w_2 = torch.sum(torch.exp(-0.1 * (t - c) ** 2), dim=1, keepdim=True)  # [B, 1, T]
        w_2[w_2==0.] = 1.

        # w_1 = self.normpdf(t, c, range_param.unsqueeze(-1))  # [B, L, T]
        # w_1 = torch.distributions.normal.Normal(c, 0.1).log_prob(t)  # [B, L, T]
        # w_2 = torch.sum(w_1, dim=1, keepdim=True)  # [B, 1, T]
        # w_2[w_2==0.] = 1.

        w = w_1 / w_2

        out = torch.matmul(w.transpose(1, 2), encoder_outputs)

        return out, w


class DurationPredictor(nn.Module):
    """ Duration Parameter Predictor """

    def __init__(self, model_config):
        super(DurationPredictor, self).__init__()
        encoder_hidden = model_config["transformer"]["encoder_hidden"]
        variance_hidden = model_config["variance_predictor"]["variance_hidden"]

        self.duration_lstm = nn.LSTM(
            encoder_hidden,
            int(variance_hidden / 2), 2,
            batch_first=True, bidirectional=True
        )
        self.duration_proj = nn.Sequential(
            LinearNorm(variance_hidden, 1),
            nn.ReLU(),
        )

    def forward(self, encoder_output, mask):
        duration_prediction, _ = self.duration_lstm(encoder_output)
        duration_prediction = self.duration_proj(duration_prediction)
        duration_prediction = duration_prediction.squeeze(-1) # [B, L]
        if mask is not None:
            duration_prediction = duration_prediction.masked_fill(mask, 0.0)

        return duration_prediction


# class RangeParameterPredictor(nn.Module):
#     """ Range Parameter Predictor """

#     def __init__(self, model_config):
#         super(RangeParameterPredictor, self).__init__()
#         encoder_hidden = model_config["transformer"]["encoder_hidden"]
#         variance_hidden = model_config["variance_predictor"]["variance_hidden"]

#         self.range_param_lstm = nn.LSTM(
#             encoder_hidden + 1,
#             int(variance_hidden / 2), 2,
#             batch_first=True, bidirectional=True
#         )
#         self.range_param_proj = nn.Sequential(
#             LinearNorm(variance_hidden, 1),
#             nn.Softplus(),
#         )

#     def forward(self, encoder_output, duration, mask):
#         range_param_input = torch.cat([encoder_output, duration.unsqueeze(-1)], dim=-1)
#         range_param_prediction, _ = self.range_param_lstm(range_param_input)
#         range_param_prediction = self.range_param_proj(range_param_prediction)
#         range_param_prediction = range_param_prediction.squeeze(-1) # [B, L]
#         if mask is not None:
#             range_param_prediction = range_param_prediction.masked_fill(mask, 0.0)

#         return range_param_prediction


class SamplingWindow(nn.Module):
    """ Sampling Window """

    def __init__(self, model_config, train_config):
        super(SamplingWindow, self).__init__()
        self.upsampling_rate = model_config["wavegrad"]["upsampling_rate"]
        self.segment_length_up = train_config["window"]["segment_length"]
        self.segment_length = train_config["window"]["segment_length"] // self.upsampling_rate

    def pad_seq(self, seq, segment_length):
        if len(seq.shape) > 2:
            return torch.nn.functional.pad(
                    seq.transpose(-2, -1), (0, segment_length - seq.shape[1]), 'constant'
                ).data.transpose(-2, -1)
        return torch.nn.functional.pad(
                seq, (0, segment_length - seq.shape[1]), 'constant'
            ).data

    def get_hidden_segment(self, hiddens, seq_starts):
        batch = list()
        for i, (hidden, seq_start) in enumerate(zip(hiddens, seq_starts)):
            batch.append(hidden[seq_start:seq_start+self.segment_length])
        return torch.stack(batch)

    def forward(self, encoder_output, audio, seq_starts=None, full_len=False):
        if full_len:
            return encoder_output, audio
        if encoder_output.shape[1] > self.segment_length:
            encoder_segment = self.get_hidden_segment(encoder_output, seq_starts)
        encoder_segment = self.pad_seq(encoder_output, self.segment_length)
        audio_segment = self.pad_seq(audio, self.segment_length_up)
        return encoder_segment, audio_segment
