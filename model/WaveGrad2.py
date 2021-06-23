import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TextEncoder, VarianceAdaptor, SamplingWindow
from wavegrad import WaveGrad
from utils.tools import get_mask_from_lengths


class WaveGrad2(nn.Module):
    """ WaveGrad2 """

    def __init__(self, preprocess_config, model_config, train_config):
        super(WaveGrad2, self).__init__()
        self.model_config = model_config

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
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
        audios=None,
        d_targets=None,
        seq_starts=None,
        phones=None,
        d_control=1.0,
        full_len=False
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Resampling
        (
            output,
            log_d_predictions,
            d_rounded,
            attns,
        ) = self.variance_adaptor(
            output,
            src_masks,
            d_targets,
            d_control,
        )

        # Sampling Window
        encoder_seg, audio_seg = self.sampling_window(output, audios, seq_starts, full_len)
        self.encoder_seg, self.audio_seg = encoder_seg, audio_seg # Save for sampling

        # WaveGrad Decoding
        noise_loss = torch.tensor([0.], device=output.device)
        if not full_len:
            noise_loss = self.decoder.compute_loss(encoder_seg.transpose(-2, -1), audio_seg)

        return (
            noise_loss,
            log_d_predictions,
            d_rounded,
            src_masks,
            src_lens,
            attns,
        )