import unittest
import tempfile
import shutil
from unittest import mock

import numpy as np
import scipy.stats
import torch
from torch.utils.data import Subset

import model
import dataset
import trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build dataset with only one batch
        self.data = dataset.MyMNIST()
        self.data.train_data = Subset(self.data.train_data, range(4))
        self.data.test_data = Subset(self.data.train_data, range(4))
        vae = model.CNNVAE(self.data.train_data[0][0].shape, bottleneck_dim=10)
        optim = torch.optim.Adam(vae.parameters())
        self.log_dir = tempfile.mkdtemp()
        self.vae_trainer = trainer.Trainer(vae, self.data, optim,
                                           batch_size=4,
                                           device='cpu',
                                           log_dir=self.log_dir,
                                           num_generated_images=1)

    def tearDown(self):
        shutil.rmtree(self.log_dir)

    @torch.no_grad()
    def test_kl_divergence(self):
        mu = np.random.randn(10) * 0.25
        sigma = np.random.randn(10) * 0.1 + 1.
        standard_normal_samples = np.random.randn(100000, 10)
        transformed_normal_sample = standard_normal_samples * sigma + mu

        # Calculate empirical pdfs for both distributions
        bins = 1000
        bin_range = [-2, 2]
        expected_kl_div = 0
        for i in range(10):
            standard_normal_dist, _ = np.histogram(standard_normal_samples[:, i], bins, bin_range)
            transformed_normal_dist, _ = np.histogram(transformed_normal_sample[:, i], bins, bin_range)
            expected_kl_div += scipy.stats.entropy(transformed_normal_dist, standard_normal_dist)

        actual_kl_div = self.vae_trainer._kl_divergence(torch.tensor(sigma).log(), torch.tensor(mu))

        self.assertAlmostEqual(expected_kl_div, actual_kl_div.numpy(), delta=0.05)

    def test_overfit_on_one_batch(self):
        # Overfit on single batch
        self.vae_trainer.train(500)

        # Overfitting a VAE is hard, so we do not choose 0. as a goal
        # 30 sum of squared errors would be a deviation of ~0,04 per pixel given a really small KL-Div
        self.assertGreaterEqual(30, self.vae_trainer.eval())

    def test_logging(self):
        with mock.patch.object(self.vae_trainer.summary, 'add_scalar') as add_scalar_mock:
            self.vae_trainer.train(1)

        expected_calls = [mock.call('train/recon_loss', mock.ANY, 0),
                          mock.call('train/kl_div_loss', mock.ANY, 0),
                          mock.call('train/loss', mock.ANY, 0),
                          mock.call('test/loss', mock.ANY, 0)]
        add_scalar_mock.assert_has_calls(expected_calls)