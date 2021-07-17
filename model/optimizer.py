import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.current_step = current_step
        self.init_lr = train_config["optimizer"]["init_lr"]
        self.decay_rate = train_config["optimizer"]["decay_rate"]
        self.decay_start = train_config["optimizer"]["decay_start"]
        self.decay_end = train_config["optimizer"]["decay_end"]

    def step_and_update_lr(self):
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def lr_lambda(self):
        progress = (self.current_step - self.decay_start) / (self.decay_end - self.decay_start)
        return self.decay_rate ** np.clip(progress, 0.0, 1.0)

    def _update_learning_rate(self):
        self.current_step += 1
        lr = self.init_lr * self.lr_lambda()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr
