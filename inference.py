from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import pdb
from datetime import datetime
import argparse
import pprint

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# local stuff
from dsets.mnist import MNIST
from mymodels.mnist_net import Net
from train_test import train, test


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset-root', default='data/mnist_easy', type=str,
                        help='root directory of the dataset')
    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--model-file', default='', type=str,
                        help='location of the model file')
    return parser

if __name__ == "__main__":
    args = argparser().parse_args()
    pprint.pprint(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    dataset_test = MNIST(args.dataset_root, subset='test', csv_file='test.csv', transform=data_transforms)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load(args.model_file))

    test(args, model, device, test_loader)
