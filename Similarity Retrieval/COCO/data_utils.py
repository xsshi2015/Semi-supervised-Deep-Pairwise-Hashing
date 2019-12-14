from __future__ import print_function, division
import numpy as np
import torch
from torch.autograd import Variable
import argparse
import os
import torch.nn.functional as F 

parser = argparse.ArgumentParser(description='PyTorch Deep Noisy Label')
parser.add_argument('-ch', '--checkpoint', metavar='DIR', help='path to checkpoint (default: ./checkpoint)', default='./checkpoint')
parser.add_argument('-w', '--workers', default=0, type=int, metavar='N', help='number of workers for data processing (default: 4)')
parser.add_argument('-nb', '--num_bits', default=10, type=int, metavar='N', help='Number of binary bits to train (default: 8)')

def split(imageData, rate=0.02, v_rate=0.02):
    trainData = np.array(imageData.train_data)
    trainLabel = np.squeeze(imageData.train_labels)

    u_label = np.squeeze(np.unique(trainLabel))
    train_idx = []
    dict_idx = []
    val_idx = []

    for iter in range(u_label.size):
        idx = np.squeeze(np.where(trainLabel==u_label[iter]))
        sn = int(rate*idx.size)
        s_idx = np.squeeze(np.random.choice(idx, sn, replace=False))
        dict_idx.extend(s_idx)

        r_idx = np.squeeze(np.setdiff1d(idx, s_idx))
        if v_rate>0:
            srn = int(v_rate*idx.size)
            sr_idx = np.squeeze(np.random.choice(r_idx, srn, replace=False))
            val_idx.extend(sr_idx)
            train_idx.extend(np.setdiff1d(r_idx, sr_idx))
        else:
            train_idx.extend(r_idx)

    train_idx = np.squeeze(train_idx)
    dict_idx = np.squeeze(dict_idx)
    val_idx = np.squeeze(val_idx)
    # import pdb; pdb.set_trace()

    return train_idx, dict_idx, val_idx


def split_idx(train_labels, select_num=1000):
    trainLabel = np.squeeze(train_labels)

    u_label = np.squeeze(np.unique(trainLabel))
    train_idx = []
    dict_idx = []

    for iter in range(u_label.size):
        idx = np.squeeze(np.where(trainLabel==u_label[iter]))
        sn = int(select_num/u_label.size)
        s_idx = np.squeeze(np.random.choice(idx, sn, replace=False))
        dict_idx.extend(s_idx)

        r_idx = np.squeeze(np.setdiff1d(idx, s_idx))
        train_idx.extend(r_idx)

    train_idx = np.squeeze(train_idx)
    dict_idx = np.squeeze(dict_idx)

    return train_idx, dict_idx

def generate_anchor_idx(train_labels, labeled_idx, anchor_num=105):
    trainLabel = np.squeeze(train_labels)
    c = train_labels.shape[1]
    sn = int(anchor_num/c)


    dict_idx = []

    r_idx = labeled_idx
    for iter in range(c):
        idx = np.squeeze(np.where(trainLabel[r_idx,iter]==1))
        s_idx = np.squeeze(np.random.choice(r_idx[idx], sn, replace=False))
        dict_idx.extend(s_idx)

        r_idx = np.squeeze(np.setdiff1d(r_idx, s_idx))

    dict_idx = np.squeeze(dict_idx)

    return dict_idx

def split_multi_idx(train_labels, select_num=1000):
    n = train_labels.shape[0]
    c = train_labels.shape[1]
    sn = int(select_num/c)
    index = np.squeeze(np.arange(n))
    dict_idx = []
    for iter in range(c):
        idx = np.squeeze(np.where(train_labels[index,iter]==1))
        s_idx = np.squeeze(np.random.choice(index[idx], sn, replace=False))
        dict_idx.extend(s_idx)    
        index = np.squeeze(np.setdiff1d(index, s_idx))


    dict_idx = np.squeeze(dict_idx)
    train_idx = np.squeeze(np.setdiff1d(np.squeeze(np.arange(n)), dict_idx))


    # import pdb; pdb.set_trace()

    return train_idx, dict_idx





