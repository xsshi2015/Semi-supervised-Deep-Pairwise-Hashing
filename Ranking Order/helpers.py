import time
import shutil
import numpy as np
from pprint import pprint
from evaluation.cap_rank import cap_rank
from evaluation.compactbit import compactbit
from evaluation.hammingDist import hammingDist
import torch
import torch.nn as nn
from torch.autograd import Variable

from scipy import io

def validate(model, dict_loader, train_loader, test_loader, use_gpu=True, batch_size=100, non_linear=True):

    dict_bin = []
    dict_labels =[]
    train_bin = []
    test_bin = []
    train_labels = []
    test_labels = []

    model.eval()


        # Some stats to monitor the loss
    for iteration, data in enumerate(dict_loader, 0):

        inputs, labels = data['image'], data['labels']


        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        #forward
        y = model(inputs)


        dict_bin.extend(y.data.cpu().numpy())
        dict_labels.extend(labels.numpy())


    # Some stats to monitor the loss
    for iteration, data in enumerate(train_loader, 0):

        inputs, labels = data['image'], data['labels']


        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        #forward
        y = model(inputs)

        train_bin.extend(y.data.cpu().numpy())
        train_labels.extend(labels.numpy())

    # Some stats to monitor the loss
    for iteration, data in enumerate(test_loader, 0):


        inputs, labels = data['image'], data['labels']

        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        #forward
        y = model(inputs)

        test_bin.extend(y.data.cpu().numpy())
        test_labels.extend(labels.numpy())

    dict_bin = np.sign(np.array(dict_bin)) 
    train_bin = np.sign(np.array(train_bin))
    test_bin = np.sign(np.array(test_bin))

    dict_labels = np.array(dict_labels)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    train_bin = np.sign(np.matmul(train_bin, np.transpose(dict_bin)))+np.ones((train_bin.shape[0], dict_bin.shape[0]))
    test_bin = np.sign(np.matmul(test_bin, np.transpose(dict_bin)))+np.ones((test_bin.shape[0], dict_bin.shape[0]))
    
    train_bin = train_bin/2
    test_bin = test_bin/2
    

    # train_bin = (np.matmul(train_bin, np.transpose(dict_bin))>0).astype(int)
    # test_bin = (np.matmul(test_bin, np.transpose(dict_bin))>0).astype(int)

    hammTrainTest = np.matmul(train_bin, np.transpose(test_bin))
    hammingRank = np.argsort(-1*hammTrainTest, axis=0)

    Rel= np.array(np.matmul(train_labels, np.transpose(test_labels)))

    NDCG = cap_rank(Rel, hammingRank, 100)
    
    return NDCG

