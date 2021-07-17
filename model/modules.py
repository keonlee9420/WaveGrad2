import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .blocks import (
    # ZoneOutBiLSTM,
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
        # self.lstm = ZoneOutBiLSTM(
        #     d_model, zoneout_rate=zoneout
        # )
        self.lstm = nn.LSTM(
            d_model,
            d_model // 2,
            1,
            batch_first=True, bidirectional=True
        )

    def forward(self, src_seq, src_len, mask=None):

        enc_output = self.src_word_emb(src_seq)

        for conv in self.conv_stack:
            enc_output = conv(enc_output, mask=mask)

        # enc_output = self.lstm(enc_output)
        # if mask is not None:
        #     enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0.)

        src_len = src_len.cpu().numpy()
        enc_output = nn.utils.rnn.pack_padded_sequence(
            enc_output, src_len, batch_first=True)

        self.lstm.flatten_parameters()
        enc_output, _ = self.lstm(enc_output)
        enc_output, _ = nn.utils.rnn.pad_packed_sequence(
            enc_output, batch_first=True)

        return enc_output

    def inference(self, src_seq, mask=None):

        enc_output = self.src_word_emb(src_seq)

        for conv in self.conv_stack:
            enc_output = conv(enc_output, mask=mask)

        self.lstm.flatten_parameters()
        enc_output, _ = self.lstm(enc_output)

        return enc_output


class GaussianUpsampling(nn.Module):
    """ Gaussian Upsampling """

    def __init__(self, model_config):
        super(GaussianUpsampling, self).__init__()

    def get_alignment_energies(self, gaussian, frames):
        """
        See https://github.com/mindslab-ai/wavegrad2
        """
        energies = gaussian.log_prob(frames).exp()  # [B, L, T]
        return energies

    def forward(self, encoder_outputs, duration, range_param, mask):
        device = encoder_outputs.device

        t = torch.sum(duration, dim=-1, keepdim=True) #[B, 1]

        e = torch.cumsum(duration, dim=-1).float() #[B, L]
        c = e - 0.5 * duration #[B, L]
        t = torch.arange(1, torch.max(t).item()+1, device=device) # (1, ..., T)
        t = t.unsqueeze(0).unsqueeze(1) #[1, 1, T]
        c = c.unsqueeze(2)
        s = range_param.unsqueeze(-1)

        g = torch.distributions.normal.Normal(loc=c, scale=s)

        w = self.get_alignment_energies(g, t)  # [B, L, T]

        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1), 0.0)

        attn = w / (torch.sum(w, dim=1).unsqueeze(1) + 1e-8)  # [B, L, T]
        out = torch.bmm(attn.transpose(1, 2), encoder_outputs)

        return out, attn


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
            # nn.ReLU(),
        )

    def forward(self, encoder_output, output_len, mask):

        output_len = output_len.cpu().numpy()
        encoder_output = nn.utils.rnn.pack_padded_sequence(
            encoder_output, output_len, batch_first=True)

        self.duration_lstm.flatten_parameters()
        duration_prediction, _ = self.duration_lstm(encoder_output)  # [B, L, channels]
        duration_prediction, _ = nn.utils.rnn.pad_packed_sequence(
            duration_prediction, batch_first=True)

        duration_prediction = self.duration_proj(duration_prediction)  # [B, L, 1]
        duration_prediction = duration_prediction.squeeze(-1)  # [B, L]
        if mask is not None:
            duration_prediction = duration_prediction.masked_fill(mask, 0.0)

        return duration_prediction

    def inference(self, encoder_output):

        self.duration_lstm.flatten_parameters()
        duration_prediction, _ = self.duration_lstm(encoder_output)  # [B, L, channels]

        duration_prediction = self.duration_proj(duration_prediction)  # [B, L, 1]
        duration_prediction = duration_prediction.squeeze(-1)  # [B, L]

        return duration_prediction


class RangeParameterPredictor(nn.Module):
    """ Range Parameter Predictor """

    def __init__(self, model_config):
        super(RangeParameterPredictor, self).__init__()
        encoder_hidden = model_config["transformer"]["encoder_hidden"]
        variance_hidden = model_config["variance_predictor"]["variance_hidden"]

        self.range_param_lstm = nn.LSTM(
            encoder_hidden + 1,
            int(variance_hidden / 2), 2,
            batch_first=True, bidirectional=True
        )
        self.range_param_proj = nn.Sequential(
            LinearNorm(variance_hidden, 1),
            nn.Softplus(),
        )

    def forward(self, encoder_output, output_len, duration, mask):

        range_param_input = torch.cat([encoder_output, duration.unsqueeze(-1)], dim=-1)

        output_len = output_len.cpu().numpy()
        range_param_input = nn.utils.rnn.pack_padded_sequence(
            range_param_input, output_len, batch_first=True)

        self.range_param_lstm.flatten_parameters()
        range_param_prediction, _ = self.range_param_lstm(range_param_input)  # [B, L, channels]
        range_param_prediction, _ = nn.utils.rnn.pad_packed_sequence(
            range_param_prediction, batch_first=True)

        range_param_prediction = self.range_param_proj(range_param_prediction)  # [B, L, 1]
        range_param_prediction = range_param_prediction.squeeze(-1)  # [B, L]
        if mask is not None:
            range_param_prediction = range_param_prediction.masked_fill(mask, 1e-8)

        return range_param_prediction

    def inference(self, encoder_output, duration):

        range_param_input = torch.cat((encoder_output, duration.unsqueeze(-1)), dim=-1)

        self.range_param_lstm.flatten_parameters()
        range_param_prediction, _ = self.range_param_lstm(range_param_input)  # [B, L, channels]

        range_param_prediction = self.range_param_proj(range_param_prediction)  # [B, L, 1]
        range_param_prediction = range_param_prediction.squeeze(-1)  # [B, L]

        return range_param_prediction


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

    def forward(self, encoder_output, audio, seq_starts=None):
        if encoder_output.shape[1] > self.segment_length:
            encoder_segment = self.get_hidden_segment(encoder_output, seq_starts)
        else:
            encoder_segment = encoder_output
        encoder_segment = self.pad_seq(encoder_segment, self.segment_length)
        audio_segment = self.pad_seq(audio, self.segment_length_up)
        return encoder_segment, audio_segment
