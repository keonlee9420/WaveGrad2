import unittest

import torch

import model
from tests import templates


class TestCNNVAE(unittest.TestCase, templates.ModelTestsMixin):
    def setUp(self):
        self.test_inputs = torch.randn(4, 1, 32, 32)
        self.net = model.CNNVAE(input_shape=(1, 32, 32), bottleneck_dim=16)


class TestMLPVAE(unittest.TestCase, templates.ModelTestsMixin):
    def setUp(self):
        self.test_inputs = torch.randn(4, 1, 32, 32)
        self.net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)