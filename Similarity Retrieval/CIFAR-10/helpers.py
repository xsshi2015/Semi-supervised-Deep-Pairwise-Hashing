import time
import shutil
import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
from torch.autograd import Variable

from scipy import io

def EncodingOnehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcMap(qB, rB, queryL, retrievalL):
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query

    return map

def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    num_query = queryL.shape[0]
    topkmap = 0

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def validate(model, train_loader, test_loader, use_gpu=True, batch_size=100, non_linear=True):
    mAP = 0.0

    train_bin = []
    test_bin = []
    train_labels = []
    test_labels = []

    model.eval()

    # Some stats to monitor the loss
    for iteration, data in enumerate(train_loader, 0):

        inputs, labels = data['image'], data['labels']

        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        #forward
        y = model(inputs)
        
        labels = EncodingOnehot(labels)
        train_bin.extend(y.data.cpu().numpy())
        train_labels.extend(labels.numpy())


    for iteration, data in enumerate(test_loader, 0):

        inputs, labels = data['image'], data['labels']

        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        #forward
        y = model(inputs)

        labels = EncodingOnehot(labels)
        test_bin.extend(y.data.cpu().numpy())
        test_labels.extend(labels.numpy())

    train_bin = np.sign(np.array(train_bin))
    test_bin = np.sign(np.array(test_bin))

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    mAP = CalcMap(test_bin, train_bin, test_labels, train_labels)

    return mAP


