#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train bilinear CNN.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    src/train.py fc --base_lr 1e-0 \
    --batch_size 16 --epochs 55 --weight_decay 1e-8 \
    --dataset aircraft \
    | tee "[fc-] base_lr_1.0-weight_decay_1e-8-epoch_.log"

This file is modified from:
    https://github.com/HaoMood/blinear-cnn.
"""


import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import distributed
import torchvision

import cub200
import aircraft
from bcnn import BCNN

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


__all__ = ['BCNNManager']
__author__ = 'Yuan Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2019-02-22'
__email__ = 'ewyuanzhang@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2019-02-22'
__version__ = '1.0'


class BCNNManager(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _options: Hyperparameters.
        _path: Useful paths.
        _net: Bilinear CNN.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
    """
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        # Network.
        if self._options['dataset'] == 'cub200':
            num_classes = 200
        elif self._options['dataset'] == 'aircraft':
            num_classes = 100
        else:
            raise NotImplementedError("Dataset "+self._options['dataset']+" is not implemented.")
        self._net = BCNN(num_classes=num_classes, pretrained=options['target'] == 'fc')
        # Load the model from disk.
        if options['target'] == 'all':
            self._net.load_state_dict(torch.load(self._path['model']))
        self._net = torch.nn.parallel.DistributedDataParallel(
            self._net.cuda(),
            device_ids=[self._options['local_rank']],
            output_device=self._options['local_rank'])
        if dist.get_rank() == 0:
            print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            self._net.module.trainable_params, lr=self._options['base_lr'] * dist.get_world_size(),
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=3, verbose=True,
            threshold=1e-4)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        if self._options['dataset'] == 'cub200':
            train_data = cub200.CUB200(
                root=self._path['dataset'], train=True, download=True,
                transform=train_transforms)
            test_data = cub200.CUB200(
                root=self._path['dataset'], train=False, download=True,
                transform=test_transforms)
        elif self._options['dataset'] == 'aircraft':
            train_data = aircraft.Aircraft(
                root=self._path['dataset'], train=True, download=True,
                transform=train_transforms)
            test_data = aircraft.Aircraft(
                root=self._path['dataset'], train=False, download=True,
                transform=test_transforms)
        else:
            raise NotImplementedError("Dataset "+self._options['dataset']+" is not implemented.")
        # Partition dataset among workers using DistributedSampler
        train_sampler = distributed.DistributedSampler(
            train_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        test_sampler = distributed.DistributedSampler(
            test_data, num_replicas=dist.get_world_size(), rank=dist.get_rank())

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True,
            sampler=train_sampler)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self._options['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True,
            sampler=test_sampler)

    def train(self):
        """Train the network."""
        best_acc = 0.0
        best_epoch = None
        if dist.get_rank() == 0:
            print('Training.')
            print('Epoch\tTrain loss\tTrain acc\tTest acc\tTrain time')
        for t in range(self._options['epochs']):
            t0 = time.time()
            self._train_loader.sampler.set_epoch(t)
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda(async=True))

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data)
                # Backward pass.
                loss.backward()
                self._solver.step()

            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)

            if dist.get_rank() == 0:
                train_acc = 100 * num_correct / num_total
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = t + 1
                    print('*', end='')
                    # Save model onto disk.
                    torch.save(self._net.module.state_dict(),
                               os.path.join(self._path['model_dir'],
                                            'vgg_16_epoch_%d.pth' % (t + 1)))
                print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2fs' %
                      (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc, time.time()-t0))

        if dist.get_rank() == 0:
            print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y in data_loader:
                # Data.
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda(async=True))

                # Prediction.
                score = self._net(X)
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # Set the model to training phase
        num_total = torch.tensor(num_total).cuda()
        num_correct = torch.tensor(num_correct).cuda()
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        return 100 * num_correct.data.item() / num_total.data.item()

    def getStat(self):
        """Get the mean and std value for a certain dataset."""
        print('Compute mean and variance for training data.')
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True,
            transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=4,
            pin_memory=True)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in train_loader:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        print(mean)
        print(std)


def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train bilinear CNN on CUB200.')
    parser.add_argument('target', choices=['fc', 'all'],
                        help='Target training layers.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        required=True, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    parser.add_argument('--model', dest='model', type=str, default="",
                        help='Model for fine-tuning.')
    parser.add_argument('--dataset', dest='dataset', type=str, default="cub200",
                        help='The dataset for training.')
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay < 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'target': args.target,
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset': args.dataset,
        'local_rank': args.local_rank
    }

    project_root = os.popen('pwd').read().strip()
    path = {
        'dataset': os.path.join(project_root, 'data', options['dataset']),
        'model_dir': os.path.join(project_root, 'model'),
    }
    if options['target'] == 'all':
        path['model'] = os.path.join(path['model_dir'], args.model)
    for d in path:
        if options['target'] == 'fc':
            if not os.path.isdir(path[d]):
                os.makedirs(path[d])
        else:
            if d == 'model':
                assert os.path.isfile(path[d])
            else:
                assert os.path.isdir(path[d])

    # Initialize process group
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(options['local_rank'])

    manager = BCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()
