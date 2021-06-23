import os

import torch
import numpy as np

from model import WaveGrad2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = WaveGrad2(preprocess_config, model_config, train_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def set_noise_schedule(model, noise_schedule_path):
    noise_schedule = torch.tensor(torch.load(noise_schedule_path))
    n_iter = noise_schedule.shape[-1]
    init_fn = lambda **kwargs: noise_schedule
    init_kwargs = {'steps': n_iter}
    model.decoder.set_new_noise_schedule(init_fn, init_kwargs)
