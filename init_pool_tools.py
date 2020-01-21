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

def obtain_init_pool(args):
    '''
    Go to the dataset root. Get train.csv
    shuffle train.csv and get the first "init_size" samples.
    create three new csv files -> init_pool.csv, labeled.csv and unlabeled.csv
    '''
    init_pool_size = args.init_size

    train_file = os.path.join(args.dataset_root, 'train.csv')
    init_file = os.path.join(args.dataset_root, 'init_pool.csv')
    labeled_file = os.path.join(args.dataset_root, 'labeled.csv')
    unlabeled_file = os.path.join(args.dataset_root, 'unlabeled.csv')

    train_rows = np.genfromtxt(train_file, delimiter=',', dtype=str)

    np.random.shuffle(train_rows)

    labeled_rows = train_rows[:init_pool_size]
    unlabeled_rows = train_rows[init_pool_size:]

    np.savetxt(labeled_file, labeled_rows,'%s,%s',delimiter=',')
    np.savetxt(init_file, labeled_rows,'%s,%s',delimiter=',')
    np.savetxt(unlabeled_file, unlabeled_rows,'%s,%s',delimiter=',')

    return labeled_file, unlabeled_file
