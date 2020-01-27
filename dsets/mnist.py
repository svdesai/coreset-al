from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pdb

class MNIST(Dataset):

    def __init__(self, root_dir, subset, csv_file, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir,'images')
        
        if '/' not in csv_file:
            self.dataframe = pd.read_csv(os.path.join(root_dir,csv_file), header=None)
        else:
            self.dataframe = pd.read_csv(csv_file, header=None)
        self.transform = transform

        self.subset = subset # train or test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.subset, self.dataframe.iloc[idx,0])
        img_name_small = self.dataframe.iloc[idx, 0]
        image = io.imread(img_name)

        label = self.dataframe.iloc[idx,1]
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label, 'img_name': img_name_small}

        return sample
