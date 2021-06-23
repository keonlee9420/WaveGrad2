import torch
import torch.nn as nn


class WaveGrad2Loss(nn.Module):
    """ WaveGrad2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(WaveGrad2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            _,
            duration_targets,
            _,
            _,
        ) = inputs[6:]
        (
            noise_loss,
            log_duration_predictions,
            _,
            src_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        log_duration_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            noise_loss + duration_loss
        )

        return (
            total_loss,
            noise_loss,
            duration_loss,
        )
