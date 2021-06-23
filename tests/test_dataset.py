import unittest

import torch

import dataset
from tests import templates

class TestMNIST(unittest.TestCase, templates.DatasetTestsMixin):
    def setUp(self):
        self.data = dataset.MyMNIST()
        self.data_shape = torch.Size((1, 32, 32))