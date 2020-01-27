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
import csv

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
    parser.add_argument('--dropout-iterations', type=int, default=5, metavar='N',
                        help='dropout iterations for bald method')
    parser.add_argument('--nclasses', type=int, default=10, metavar='N',
                        help='number of classes in the dataset')
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

def get_probs(model, loader, stochastic=False):
    probs = []
    if stochastic:
        model.train()
    else:
        model.eval()

    count = 0
    with torch.no_grad():
        for sample in loader:
            data = sample['image']
            target = sample['label']
            img_name = sample['img_name'][0]

            data, target = data.to(device), target.to(device)

            if stochastic:
                output = model.stochastic_pred(data)
            output = model(data)

            # convert log softmax into softmax outputs
            prob = output.cpu().numpy()
            prob = np.exp(prob[0])

            probs.append(prob)

            count += 1

    return np.array(probs)

def active_sample(args, unlabeled_rows, sample_size, method='random', model=None):
    if method == 'random':
        np.random.shuffle(unlabeled_rows)
        sample_rows = unlabeled_rows[:sample_size]
        return sample_rows
    
    if method == 'prob_uncertain' or method == 'prob_margin' or method == 'prob_entropy':
        # unlabeled loader
        data_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])

        unlab_dset = MNIST(args.dataset_root, subset='train',csv_file='unlabeled.csv',transform=data_transforms)
        unlab_loader = DataLoader(unlab_dset, batch_size=1, shuffle=False, **kwargs)

        probabilities = get_probs(model, unlab_loader)  
        
        if method == 'prob_uncertain':
            max_probs = np.max(probabilities, axis=1)
        
            # kind of a heap sort.
            argsorted_maxprobs = np.argpartition(max_probs, sample_size)
            # least probabilities
            sample_indices = argsorted_maxprobs[:sample_size]
        
        elif method == 'prob_margin':
            # find the top two probabilities
            top2_sorted = -1 * np.partition(-probabilities, 2, axis=1)
            margins = [x[0]-x[1] for x in top2_sorted]
            margins = np.array(margins)

            # find the ones with highest margin
            argsorted_margins = np.argpartition(-margins, sample_size)
            sample_indices = argsorted_margins[:sample_size]

        
        elif method == 'prob_entropy':
            entropy_arr = (-probabilities*np.log2(probabilities)).sum(axis=1)

            # find the ones with the highest entropy
            argsorted_ent = np.argpartition(-entropy_arr, sample_size)
            sample_indices = argsorted_ent[:sample_size]
           
        sample_rows = unlabeled_rows[sample_indices]
        return sample_rows
    
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

        return sample_rows
    
    if method == 'dbal_bald':
        # according to BALD implementation by Riashat Islam
        # first randomly sample 2000 points
        dropout_pool_size = 2000
        unl_rows = np.copy(unlabeled_rows)

        if len(unl_rows) >= dropout_pool_size:
            np.random.shuffle(unl_rows)
            dropout_pool = unl_rows[:dropout_pool_size]
            temp_unlabeled_csv = 'unlabeled_temp.csv'
            np.savetxt(os.path.join(args.dataset_root, temp_unlabeled_csv), dropout_pool,'%s,%s',delimiter=',')
            csv_file = temp_unlabeled_csv
        else:
            dropout_pool = unl_rows
            csv_file = 'unlabeled.csv'
        
        

        #create unlabeled loader
        data_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])   

        unlab_dset = MNIST(args.dataset_root, subset='train',csv_file=csv_file,transform=data_transforms)
        unlab_loader = DataLoader(unlab_dset, batch_size=1, shuffle=False, **kwargs)

        scores_sum = np.zeros(shape=(len(dropout_pool), args.nclasses))
        entropy_sum = np.zeros(shape=(len(dropout_pool)))

        for _ in range(args.dropout_iterations):
            probabilities = get_probs(model, unlab_loader, stochastic=True)

            

            entropy = - np.multiply(probabilities, np.log(probabilities))
            entropy = np.sum(entropy, axis=1)

            entropy_sum += entropy
            scores_sum += probabilities
            
        
        avg_scores = np.divide(scores_sum, args.dropout_iterations)
        entropy_avg_sc = - np.multiply(avg_scores, np.log(avg_scores))
        entropy_avg_sc = np.sum(entropy_avg_sc, axis=1)

        avg_entropy = np.divide(entropy_sum, args.dropout_iterations)

        bald_score = entropy_avg_sc - avg_entropy

        # partial sort
        argsorted_bald = np.argpartition(-bald_score, sample_size)
        # get the indices
        sample_indices = argsorted_bald[:sample_size]
        sample_rows = dropout_pool[sample_indices]

        return sample_rows





def log(dest_dir, episode_id, sample_method, sample_time, accuracy, labeled_rows):
    log_file = os.path.join(dest_dir, 'log.csv')
    if not os.path.exists(log_file):
        log_rows = [['Episode Id','Sample Method','Sampling Time (s)','Labeled Pool','Accuracy']]
    else:
        log_rows = np.genfromtxt(log_file, delimiter=',', dtype=str).tolist()

    log_rows.append([episode_id,sample_method, sample_time, len(labeled_rows), accuracy])
    np.savetxt(log_file,log_rows,'%s,%s,%s,%s,%s',delimiter=',')

def log_picked_samples(dest_dir, samples, ep_id=0):
    dest_file = os.path.join(dest_dir, 'picked.txt')

    with open(dest_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode ID", str(ep_id)])
        for s in samples:
            writer.writerow(s.tolist())
            


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



    # copy labeled csv and unlabeled csv to dest_dir
    # pdb.set_trace()

    # save config
    with open(dest_dir_name + '/config.json', 'w') as f:
        import json
        json.dump(vars(args),f)
    # save logs

    # pdb.set_trace()
    log(dest_dir_name, 0, args.sampling_method, 0, accuracy, [0]*args.init_size)
    log_picked_samples(dest_dir_name, np.genfromtxt(labeled_csv, delimiter=',', dtype=str))


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
        sample_rows = active_sample(args, unlabeled_rows, sample_size, method=args.sampling_method, model=model)
        sample_end = time.time()

        # log picked samples
        log_picked_samples(dest_dir_name, sample_rows, episode_id)



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
