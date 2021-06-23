import unittest

import torch
from torch.utils.data import DataLoader


class ModelTestsMixin:
    @torch.no_grad()
    def test_shape(self):
        outputs = self.net(self.test_inputs)
        self.assertEqual(self.test_inputs.shape, outputs.shape)

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_device_moving(self):
        net_on_gpu = self.net.to('cuda:0')
        net_back_on_cpu = net_on_gpu.cpu()

        torch.manual_seed(42)
        outputs_cpu = self.net(self.test_inputs)
        torch.manual_seed(42)
        outputs_gpu = net_on_gpu(self.test_inputs.to('cuda:0'))
        torch.manual_seed(42)
        outputs_back_on_cpu = net_back_on_cpu(self.test_inputs)

        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_gpu.cpu()))
        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_back_on_cpu))

    def test_batch_independence(self):
        inputs = self.test_inputs.clone()
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = inputs[0].shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        outputs = outputs * mask

        # Compute backward pass
        loss = outputs.mean()
        loss.backward()

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(inputs.grad):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        outputs = self.net(self.test_inputs)
        loss = outputs.mean()
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0., torch.sum(param.grad ** 2))


class DatasetTestsMixin:
    def test_shape(self):
        with self.subTest(split='train'):
            self._check_shape(self.data.train_data)
        with self.subTest(split='test'):
            self._check_shape(self.data.test_data)

    def _check_shape(self, dataset):
        sample, _ = dataset[0]
        self.assertEqual(self.data_shape, sample.shape)

    def test_scaling(self):
        with self.subTest(split='train'):
            self._check_scaling(self.data.train_data)
        with self.subTest(split='test'):
            self._check_scaling(self.data.test_data)

    def _check_scaling(self, data):
        for sample, _ in data:
            # Values are in range [-1, 1]
            self.assertGreaterEqual(1, sample.max())
            self.assertLessEqual(-1, sample.min())
            # Values are not only covering [0, 1] or [-1, 0]
            self.assertTrue(torch.any(sample < 0))
            self.assertTrue(torch.any(sample > 0))

    def test_augmentation(self):
        with self.subTest(split='train'):
            self._check_augmentation(self.data.train_data, active=True)
        with self.subTest(split='test'):
            self._check_augmentation(self.data.test_data, active=False)

    def _check_augmentation(self, data, active):
        are_same = []
        for i in range(len(data)):
            sample_1, _ = data[i]
            sample_2, _ = data[i]
            are_same.append(0 == torch.sum(sample_1 - sample_2))

        if active:
            self.assertTrue(not all(are_same))
        else:
            self.assertTrue(all(are_same))

    def test_single_process_dataloader(self):
        with self.subTest(split='train'):
            self._check_dataloader(self.data.train_data, num_workers=0)
        with self.subTest(split='test'):
            self._check_dataloader(self.data.test_data, num_workers=0)

    def test_multi_process_dataloader(self):
        with self.subTest(split='train'):
            self._check_dataloader(self.data.train_data, num_workers=2)
        with self.subTest(split='test'):
            self._check_dataloader(self.data.test_data, num_workers=2)

    def _check_dataloader(self, data, num_workers):
        loader = DataLoader(data, batch_size=4, num_workers=num_workers)
        for _ in loader:
            pass