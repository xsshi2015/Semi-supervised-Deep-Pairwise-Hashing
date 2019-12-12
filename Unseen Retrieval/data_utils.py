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


def split_class(Data, Label, selected_class):
    select_index = []
    index = np.squeeze(np.arange(np.squeeze(Label).size))

    for iter in range(selected_class.size):
        idx = np.squeeze(np.where(Label==selected_class[iter]))
        select_index.extend(idx)

    select_index = np.squeeze(select_index)
    r_index = np.squeeze(np.setdiff1d(index, select_index))


    return Data[select_index,:,:,:], Label[select_index], Data[r_index,:,:,:], Label[r_index]


def split_multi_label_class(Data, Label, selected_class):
    c =Label.shape[1]
    select_index = []
    index = np.squeeze(np.arange(Label.shape(0)))

    for iter in range(selected_class.size):
        idx = np.squeeze(np.where(Label[index,selected_class[iter]]==1))
        select_index.extend(idx)

    select_index = np.squeeze(select_index)
    index = np.squeeze(np.setdiff1d(index, select_index))

    r_class = np.squeeze(np.setdiff1d(np.squeeze(np.arange(Label.size(1))), selected_class))

    return Data[select_index,:,:,:], Label[select_index,selected_class], Data[index,:,:,:], Label[index,r_class]


def select_multilabel_anchors_1(trainLabel, anchor_num):
    labeled_idx = np.squeeze(np.arange(trainLabel.shape(0)))
    s_an = int(anchor_num/trainLabel.shape(1))

    dict_idx = []

    for iter in range(trainLabel.shape(1)):
        idx = np.squeeze(np.where(trainLabel[labeled_idx,iter]==1))
        s_idx = np.squeeze(np.random.choice(labeled_idx[idx], s_an, replace=False))
        dict_idx.extend(s_idx)

        labeled_idx = np.squeeze(np.setdiff1d(labeled_idx, s_idx))

    return np.squeeze(dict_idx)
    

def select_anchors_1(anchor_labels, anchor_num=1000):
    labeled_idx = np.squeeze(np.arange(np.squeeze(anchor_labels).size))
    u_labels = np.squeeze(np.unique(anchor_labels))
    s_an = int(anchor_num/u_labels.size)

    dict_idx = []

    for iter in range(u_labels.size):
        idx = np.squeeze(np.where(anchor_labels==u_labels[iter]))
        s_idx = np.squeeze(np.random.choice(labeled_idx[idx], s_an, replace=False))
        dict_idx.extend(s_idx)


    return np.squeeze(dict_idx)



def split_data(imageData, select_num=1000):
    trainData = np.array(imageData.data)
    trainLabel = np.squeeze(imageData.labels)

    u_label = np.squeeze(np.unique(trainLabel))
    train_idx = []
    dict_idx = []

    for iter in range(u_label.size):
        idx = np.squeeze(np.where(trainLabel==u_label[iter]))
        sn = int(select_num/u_label.size)
        s_idx = idx[:sn] #np.squeeze(np.random.choice(idx, sn, replace=False))
        dict_idx.extend(s_idx)

        r_idx = np.squeeze(np.setdiff1d(idx, s_idx))
        train_idx.extend(r_idx)

    train_idx = np.squeeze(train_idx)
    dict_idx = np.squeeze(dict_idx)
    # import pdb; pdb.set_trace()

    return trainData[train_idx,:,:,:], trainLabel[train_idx], trainData[dict_idx,:,:,:], trainLabel[dict_idx] 

def split_idx(train_labels, select_num=1000):
    trainLabel = np.squeeze(train_labels)

    u_label = np.squeeze(np.unique(trainLabel))
    train_idx = []
    dict_idx = []

    for iter in range(u_label.size):
        idx = np.squeeze(np.where(trainLabel==u_label[iter]))
        sn = int(select_num/u_label.size)
        s_idx = idx[:sn] #np.squeeze(np.random.choice(idx, sn, replace=False))
        dict_idx.extend(s_idx)

        r_idx = np.squeeze(np.setdiff1d(idx, s_idx))
        train_idx.extend(r_idx)

    train_idx = np.squeeze(train_idx)
    dict_idx = np.squeeze(dict_idx)
    # import pdb; pdb.set_trace()

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

def select_anchors(trainLabel, labeled_idx, anchor_num=500):
    anchor_labels = trainLabel[labeled_idx]
    u_labels = np.squeeze(np.unique(anchor_labels))
    s_an = int(anchor_num/u_labels.size)

    dict_idx = []

    for iter in range(u_labels.size):
        idx = np.squeeze(np.where(anchor_labels==u_labels[iter]))
        s_idx = np.squeeze(np.random.choice(labeled_idx[idx], s_an, replace=False))
        dict_idx.extend(s_idx)


    return np.squeeze(dict_idx)
    



