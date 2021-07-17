import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    TextEncoder,
    DurationPredictor,
    RangeParameterPredictor,
    GaussianUpsampling,
    SamplingWindow,
)
from wavegrad import WaveGrad
from utils.tools import get_mask_from_lengths


class WaveGrad2(nn.Module):
    """ WaveGrad2 """

    def __init__(self, preprocess_config, model_config, train_config):
        super(WaveGrad2, self).__init__()
        self.model_config = model_config

        self.encoder = TextEncoder(model_config)
        self.duration_predictor = DurationPredictor(model_config)
        self.range_param_predictor = RangeParameterPredictor(model_config)
        self.gaussian_upsampling = GaussianUpsampling(model_config)
        self.sampling_window = SamplingWindow(model_config, train_config)
        self.decoder = WaveGrad(preprocess_config, model_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        self.encoder_seg = self.audio_seg = None

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        audios,
        d_targets,
        seq_starts,
        phones,
    ):
        # Text Encoding
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        output = self.encoder(texts, src_lens, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Resampling
        log_d_predictions = self.duration_predictor(output, src_lens, src_masks)
        range_param = self.range_param_predictor(output, src_lens, d_targets, src_masks)
        output, attns = self.gaussian_upsampling(output, d_targets, range_param, src_masks)
        d_rounded = d_targets

        # Sampling Window
        encoder_seg, audio_seg = self.sampling_window(output, audios, seq_starts)
        self.encoder_seg, self.audio_seg = encoder_seg, audio_seg # Save for sampling

        # Compute Noise Loss
        noise_loss = self.decoder.compute_loss(encoder_seg.transpose(-2, -1), audio_seg)

        return (
            noise_loss,
            log_d_predictions,
            d_rounded,
            src_masks,
            src_lens,
            attns,
        )

    def inference(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        d_control=1.0,
    ):
        # Text Encoding
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        output = self.encoder.inference(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Resampling
        log_d_predictions = self.duration_predictor.inference(output)
        d_rounded = torch.clamp(
            (torch.round(torch.exp(log_d_predictions) - 1) * d_control),
            min=0,
        )
        range_param = self.range_param_predictor.inference(output, d_rounded)
        output, attns = self.gaussian_upsampling(output, d_rounded, range_param, src_masks)

        # Decoding
        output = self.decoder.forward(
            output.transpose(-2, -1), store_intermediate_states=False
        )

        return output
