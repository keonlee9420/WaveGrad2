import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import audio as Audio
from scipy.io import wavfile
from matplotlib import pyplot as plt
from datetime import datetime
from benchmark import compute_rtf


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if len(data) == 10:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            audios,
            durations,
            seq_starts,
            phones,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        audios = torch.from_numpy(audios).float().to(device)
        durations = torch.from_numpy(durations).long().to(device)
        seq_starts = torch.from_numpy(seq_starts).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            audios,
            durations,
            seq_starts,
            phones,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/noise", losses[1], step)
        logger.add_scalar("Loss/duration_loss", losses[2], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(model, targets, predictions, STFT, noise_schedule=None):

    basename = targets[0][0]

    raw_text_full = targets[1][0]
    phone = targets[-1][0]
    seq_start = targets[8][0].item()
    duration = targets[7][0].detach().cpu()
    start_idx, end_idx = -1, -1
    d_sum = 0
    for i, d in enumerate(duration):
        d_sum += d.item()
        if start_idx < 0 and d_sum >= seq_start:
            start_idx = i
            continue
        if start_idx >= 0 and end_idx < 0 and d_sum > seq_start + 64:
            end_idx = i
            break
    phone_ = phone.strip("}{").split(" ")[start_idx:end_idx+1]
    # print("\n", raw_text_full)
    # print(phone)
    # print(f"Phone Seg {start_idx}:{end_idx} =", phone_)

    audio_len = predictions[2][0].sum().item()
    attention = predictions[5][0][:, :audio_len].detach().cpu().numpy()

    # Sample Audio
    if noise_schedule is not None:
        model.decoder.set_new_noise_schedule(
            init=torch.linspace,
            init_kwargs={
                'steps': noise_schedule["n_iter"],
                'start': noise_schedule["betas_range"][0],
                'end': noise_schedule["betas_range"][1]
            }
        )
    with torch.no_grad():
        start = datetime.now()
        audio_prediction = model.decoder.forward(
            model.encoder_seg.transpose(-2, -1), store_intermediate_states=False
        )[0].detach().cpu()
        end = datetime.now()
        generation_time = (end - start).total_seconds()
        print("Sample a single audio in {:.4f} sec".format(generation_time))

    # Draw Spectrogram
    audio_target = model.audio_seg[0].detach().cpu()
    mel_target, _ = Audio.tools.get_mel_from_wav(audio_target, STFT)
    mel_prediction, _ = Audio.tools.get_mel_from_wav(audio_prediction, STFT)
    fig = plot_mel(
        [
            mel_prediction,
            mel_target,
            attention,
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram", "Resampling Attention"],
        attention=True,
        phone=phone_
    )

    return fig, audio_target, audio_prediction, basename


def synth_samples(model, targets, STFT, preprocess_config, path):

    basenames = targets[0]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    rtfs = []
    for i in range(len(basenames)):
        basename = basenames[i]

        # Sample Audio
        start = datetime.now()
        output = model.decoder.forward(
            model.encoder_seg.transpose(-2, -1), store_intermediate_states=False
        )[i].cpu().squeeze()
        end = datetime.now()
        inference_time = (end - start).total_seconds()
        rtf = compute_rtf(output, inference_time, sample_rate=sampling_rate)
        rtfs.append(rtf)

        # Save Audio
        wavfile.write(
            os.path.join(path, "{}.wav".format(basename)), sampling_rate, output.numpy()
        )

        # Draw and Save Spectrogram
        mel_prediction, _ = Audio.tools.get_mel_from_wav(output, STFT)
        fig = plot_mel(
            [
                mel_prediction,
            ],
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    print(f'Synthesis Done. RTF estimate: {np.mean(rtfs)} ± {np.std(rtfs)}')


def plot_mel(data, titles, attention=False, phone=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        if i == len(data)-1 and attention:
            im = axes[i][0].imshow(data[i], origin='lower', aspect='auto')
            axes[i][0].set_xlabel('Audio timestep (downsampled)')
            axes[i][0].set_ylabel('Text timestep')
            axes[i][0].set_xlim(0, data[i].shape[1])
            axes[i][0].set_ylim(0, data[i].shape[0])
            axes[i][0].set_title(titles[i], fontsize="medium")
            axes[i][0].tick_params(labelsize="x-small")
            axes[i][0].set_anchor("W")
            fig.colorbar(im, ax=axes[i][0])
            break
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    if phone is not None:
        fig.suptitle(" ".join(phone), fontsize=16)

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
