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
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# local stuff
from dsets.mnist import MNIST
from mymodels.mnist_net import Net
from train_test import train, test
from init_pool_tools import obtain_init_pool
from coreset import Coreset_Greedy


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--al-batch-size', default=500, type=int,
                        help='number of samples to add in each iteration')
    parser.add_argument('--init-size', default=1000, type=int,
                        help='init pool size')
    parser.add_argument('--sampling-method', default='random', type=str,
                        help='one of random, coreset')
    parser.add_argument('--dataset-root', default='data/mnist_easy', type=str,
                        help='root directory of the dataset')
    parser.add_argument('--dataset-name', default='mnist', type=str,
                        help='dataset name')
    parser.add_argument('--output-dir', default='output', type=str,
                        help='dataset name')
    parser.add_argument('--max-eps', type=int, default=10, metavar='N',
                        help='max episodes of active learning')
    return parser

def remove_rows(perm, samp):

    len_perm = len(perm)
    len_samp = len(samp)

    perm = perm.tolist()
    samp = samp.tolist()

    result = [item for item in perm if item not in samp]

    assert len(result) == len_perm - len_samp
    return np.array(result)

def get_features(model, loader):
    features = []
    model.eval()

    count = 0
    with torch.no_grad():
        for sample in loader:
            data = sample['image']
            target = sample['label']
            img_name = sample['img_name'][0]

            data, target = data.to(device), target.to(device)
            output = model.get_features(data)
            # pdb.set_trace()

            count += 1
            # if count > 10000:
            #     break

            features.append(output.cpu().numpy())
            # features.append((img_name, output.cpu().numpy()))
    return features
def active_sample(unlabeled_rows, sample_size, method='random', model=None):
    if method == 'random':
        np.random.shuffle(unlabeled_rows)
        sample_rows = unlabeled_rows[:sample_size]

        return sample_rows
    # if method == 'coreset':

    #     #create unlabeled loader
    #     data_transforms = transforms.Compose([
    #                            transforms.ToTensor(),
    #                            transforms.Normalize((0.1307,), (0.3081,))
    #                        ])

    #     unlab_dset = MNIST(args.dataset_root, subset='train',csv_file='unlabeled.csv',transform=data_transforms)
    #     unlab_loader = DataLoader(unlab_dset, batch_size=1, shuffle=False, **kwargs)

    #     #labeled dataloader
    #     lab_dset = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
    #     lab_loader = DataLoader(lab_dset, batch_size=1, shuffle=False, **kwargs)

    #     # get labeled features
    #     labeled_features = get_features(model, lab_loader) # (img_name, features)
    #     # pdb.set_trace()
    #     # get unlabeled features
    #     unlabeled_features = get_features(model, unlab_loader)# (img_name, features)

    #     # find closest pairs
    #     closest_pairs = [] # (unlabeled_index, labeled_index)

    #     for u_idx, u in enumerate(unlabeled_features):

    #         u_rep = u[1]

    #         l_rep = labeled_features[0][1]
    #         min_dist = np.linalg.norm(u_rep - l_rep)

    #         closest_pair = (u_idx, 0, min_dist) # init
    #         for l_i in range(1,len(labeled_features)):
    #             l = labeled_features[l_i]
    #             l_rep = l[1]
    #             curr_dist = np.linalg.norm(u_rep - l_rep)
    #             if min_dist > curr_dist:
    #                 min_dist = curr_dist
    #                 closest_pair = (u_idx, l_i, min_dist)

    #         closest_pairs.append(closest_pair)
    #     # pdb.set_trace()

    #     # after obtaining closest pairs
    #     # find closest pairs which are the farthest
    #     closest_pairs = sorted(closest_pairs, key=lambda x:x[2], reverse=True) #sort by distance

    #     sampled_items = closest_pairs[:sample_size]

    #     # extract indices
    #     unlab_indices = [x[0] for x in sampled_items]

    #     # finally
    #     sample_rows = []
    #     for idx, u in enumerate(unlabeled_rows):
    #         if idx in unlab_indices:
    #             sample_rows.append(u)

    #     # pdb.set_trace()
    #     assert len(sample_rows) == sample_size
    #     return np.array(sample_rows)
    
    if method == 'coreset':
        #create unlabeled loader
        data_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])

        unlab_dset = MNIST(args.dataset_root, subset='train',csv_file='unlabeled.csv',transform=data_transforms)
        unlab_loader = DataLoader(unlab_dset, batch_size=1, shuffle=False, **kwargs)

        #labeled dataloader
        lab_dset = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
        lab_loader = DataLoader(lab_dset, batch_size=1, shuffle=False, **kwargs)

        # get labeled features
        labeled_features = get_features(model, lab_loader) # (img_name, features)
        # pdb.set_trace()
        # get unlabeled features
        unlabeled_features = get_features(model, unlab_loader)# (img_name, features)

        all_features = labeled_features + unlabeled_features
        labeled_indices = np.arange(0,len(labeled_features))

        

        coreset = Coreset_Greedy(all_features)
        new_batch, max_distance = coreset.sample(labeled_indices, sample_size)
        
        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labeled_features) for i in new_batch]
        
        sample_rows = unlabeled_rows[new_batch]
        # pdb.set_trace()

        return sample_rows


def log(dest_dir, episode_id, sample_method, sample_time, accuracy, labeled_rows):
    log_file = os.path.join(dest_dir, 'log.csv')
    if not os.path.exists(log_file):
        log_rows = [['Episode Id','Sample Method','Sampling Time (s)','Labeled Pool','Accuracy']]
    else:
        log_rows = np.genfromtxt(log_file, delimiter=',', dtype=str).tolist()

    log_rows.append([episode_id,sample_method, sample_time, len(labeled_rows), accuracy])
    np.savetxt(log_file,log_rows,'%s,%s,%s,%s,%s',delimiter=',')


if __name__ == "__main__":
    args = argparser().parse_args()
    pprint.pprint(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Obtaining init pool
    labeled_csv, unlabeled_csv = obtain_init_pool(args)
    print("Initial labeled pool created.")

    # initial setup
    data_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    dataset_train = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
    dataset_test = MNIST(args.dataset_root, subset='test', csv_file='test.csv', transform=data_transforms)

    # initial training
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device) # initialize the model.
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # setup the optimizer
    scheduler = StepLR(optimizer, step_size = 1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        model = train(args, model, device, train_loader, optimizer, epoch)
        
        scheduler.step()

    accuracy = test(args, model, device, test_loader)
    # save model
    dest_dir = os.path.join(args.output_dir, args.dataset_name)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    now = datetime.now()
    dest_dir_name = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
    dest_dir_name = os.path.join(dest_dir, dest_dir_name)
    if not os.path.exists(dest_dir_name):
        os.mkdir(dest_dir_name)
    save_path = os.path.join(dest_dir_name,'init.pth')
    torch.save(model.state_dict(), save_path)
    print("initial pool model saved in: ",save_path)

    log(dest_dir_name, 0, args.sampling_method, 0, accuracy, [0]*args.init_size)


    # start the active learning loop.
    episode_id = 1
    while True:

        if episode_id > args.max_eps:
            break


        # read the unlabeled file
        unlabeled_rows = np.genfromtxt(unlabeled_csv, delimiter=',', dtype=str)
        labeled_rows = np.genfromtxt(labeled_csv, delimiter=',', dtype=str)

        print("Episode #",episode_id)


        # sanity checks
        if len(unlabeled_rows) == 0:
            break

        # set the sample size
        sample_size = args.al_batch_size
        if len(unlabeled_rows) < sample_size:
            sample_size = len(unlabeled_rows)

        # sample
        sample_start = time.time()
        sample_rows = active_sample(unlabeled_rows, sample_size, method=args.sampling_method, model=model)
        sample_end = time.time()

        sample_time = sample_end - sample_start

        # update the labeled pool
        labeled_rows = np.concatenate((labeled_rows,sample_rows),axis=0)
        np.savetxt(labeled_csv, labeled_rows,'%s,%s',delimiter=',')


        # update the unlabeled pool
        unlabeled_rows = remove_rows(unlabeled_rows, sample_rows)
        np.savetxt(unlabeled_csv, unlabeled_rows, '%s,%s', delimiter=',')

        print("Unlabeled pool size: ",len(unlabeled_rows))
        print("Labeled pool size: ",len(labeled_rows))


        #train the model
        dataset_train = MNIST(args.dataset_root, subset='train',csv_file='labeled.csv',transform=data_transforms)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)

        model = Net().to(device) # initialize the model.
        optimizer = optim.Adam(model.parameters(), lr=args.lr) # setup the optimizer
        # scheduler = StepLR(optimizer, step_size = 1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            model = train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(args, model, device, test_loader)
            # scheduler.step()

        # save model
        save_path = os.path.join(dest_dir_name, 'ep_'+str(episode_id)+'.pth')
        torch.save(model.state_dict(), save_path)

        log(dest_dir_name, episode_id, args.sampling_method, sample_time, accuracy, labeled_rows)

        episode_id += 1
