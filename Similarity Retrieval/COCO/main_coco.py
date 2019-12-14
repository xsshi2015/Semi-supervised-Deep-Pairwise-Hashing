from __future__ import print_function
'''

Resources:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd
https://discuss.pytorch.org/t/convert-numpy-to-pytorch-dataset/743
http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#compose-transforms

https://github.com/jcjohnson/pytorch-examples#pytorch-custom-nn-modules
http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd
http://pytorch.org/docs/master/nn.html#convolution-layers
'''

import os
import sys
import time
#import matplotlib.pyplot as plt
import shutil
import visdom
# import helpers
import argparse
import torchvision
import numpy as np
from scipy import io

import torch

import torch.nn as nn
import data_utils as du
from pprint import pprint
import torch.optim as optim
import torch.nn.functional as F

from stat_utils import AverageMeter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torchvision.models.resnet import model_urls
from torch.utils.data.sampler import SubsetRandomSampler

from networks import CNN
from networks import Custom_Loss as custom_loss
from data_transform import DataTransform as DT
from utils import AverageMeterSet
from adam import Adam

import helpers
import data
from multilabel_imgfolder import ImageFolder
import folder


parser = argparse.ArgumentParser(description='PyTorch CoadjutantHashing Training')
parser.add_argument('-d', '--data', metavar='DIR', help='path to dataset (default: ./data)', default='./data')
parser.add_argument('-ch', '--checkpoint', metavar='DIR', help='path to checkpoint (default: ./checkpoint)', default='./checkpoint')
parser.add_argument('-lg', '--log', metavar='DIR', help='path to log (default: ./log)', default='./log')
parser.add_argument('-ds', '--dataset', metavar='FILE', help='dataset to use [cifar100, nuswide, coco, cocosent] (default: cifar10)', default='coco')
parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-st', '--step_size', default=10, type=int, metavar='N', help='step size to decay the learning rate (default: 10)')
parser.add_argument('-b', '--batch_size', default=100, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N', help='number of workers for data processing (default: 4)')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('-sz', '--image_size', default=224, type=int, metavar='N', help='Size of input to use (default: 32)')
parser.add_argument('-c', '--channels', default=3, type=int, metavar='N', help='Number of channels of the input, which could be different for sentences (default: 3)')
parser.add_argument('-nb', '--num_bits', default=48, type=int, metavar='N', help='Number of binary bits to train (default: 8)')
parser.add_argument('-nc', '--num_class', default=80, type=int, metavar='N', help='Number of classes to train (default: 10)')
parser.add_argument('-an', '--anchor_num', default=400, type=int, metavar='N', help='Number of anchors (default: 100)')
parser.add_argument('-gamma', '--gamma', default=0.1, type=float, metavar='F', help='initialize parameter lambda (default: 0.1)')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')



def save_checkpoint(state, is_best, prefix='', num_bits=48, filename='./checkpoint/checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        print("====> saving the new best model")
        path = "/".join(filename.split('/')[:-1])
        best_filename = os.path.join(path, prefix+'model_best_'+'coco_'+str(num_bits)+'.pt')
        shutil.copyfile(filename, best_filename)

def rampup(global_step, rampup_length=20):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)


def rampdown(epoch, num_epochs=50, rampdown_length=20):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return np.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0

def step_rampup(epoch, rampup_length=20):
    if epoch<=rampup_length:
        return 1.0
    else:
        return 0.0

def generate_anchor_vectors(dict_loader_train):
    anchors_data = []
    anchor_Label = []

    for jteration, anchor_data in enumerate(dict_loader_train, 0):
        anchor_inputs, anchor_labels, _ =  anchor_data['image'], anchor_data['labels'], anchor_data['index'].numpy()
        if anchor_inputs.size(3)==3:
            anchor_inputs = anchor_inputs.permute(0,3,1,2)
        
        anchors_data.extend(anchor_inputs.numpy())
        anchor_Label.extend(anchor_labels.numpy())

    anchors_data = torch.from_numpy(np.array(anchors_data)).type(torch.FloatTensor).cuda()
    anchor_Label = torch.from_numpy(np.array(anchor_Label)).type(torch.FloatTensor)

    return anchors_data, anchor_Label

def CalcSim(batch_label, train_label, eps=1e-3):
    S = (batch_label.mm(train_label.t())>0).type(torch.FloatTensor).cuda()
    W = S.clone()
    W[W==0]=-1.0
    return S, W


def pre_train(dataloader, test_loader, dict_loader, dataloader_test, mask_labels, total_epochs = 50, learning_rate=1e-4, load_flag=0, use_gpu=True, seed=123):
    
    args = parser.parse_args()
    pprint(args)    

    num_bits = args.num_bits

    model = CNN(model_name='alexnet', bit = num_bits, class_num= args.num_class)

    criterion = custom_loss(num_bits = num_bits)

    arch = 'cnn_'
    filename = arch + args.dataset+'_'+str(num_bits)+"bits"
    checkpoint_filename = os.path.join(args.checkpoint, filename+'.pt')


    
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        torch.cuda.manual_seed(seed)
    


    running_loss = 0.0

    start_epoch=0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    best_prec = -99999



    k = 8000
    ramp_up_mult = -5
    n_samples = 183734
    alpha = 0.4
    alpha_1 = 0.99
    
    n_classes = 10


    mask_labels = torch.from_numpy(mask_labels).long().cuda()

    Z_h1 = torch.zeros(n_samples, num_bits).float().cuda()        # intermediate values
    z_h1 = torch.zeros(n_samples, num_bits).float().cuda()        # temporal outputs
    h1 = torch.zeros(n_samples, num_bits).float().cuda()  # current outputs


    Z_h2 = torch.zeros(args.anchor_num, num_bits).float().cuda()        # intermediate values
    z_h2 = torch.zeros(args.anchor_num, num_bits).float().cuda()        # temporal outputs
    h2 = torch.zeros(args.anchor_num, num_bits).float().cuda()  # current outputs



    for epoch in range(start_epoch, total_epochs):    
        model.train(True)
        
        rampup_value = rampup(epoch)
        rampdown_value = rampdown(epoch)
        learning_rate =  rampup_value*rampdown_value*0.00005
        adam_beta1 = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
        adam_beta2 =  step_rampup(epoch) * 0.99 + (1- step_rampup(epoch))* 0.999

        if epoch==0:
            u_w = 0.0    
        else:
            u_w = rampup_value
        
        u_w_m = u_w*1.0


        u_w_m =  torch.autograd.Variable(torch.FloatTensor([u_w_m]).cuda(), requires_grad=False)

        optimizer = Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=1e-8, amsgrad=True)


        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        anchors_data, anchor_Label = generate_anchor_vectors(dict_loader)

        for iteration, data in enumerate(dataloader, 0):
            anchor_index = np.arange(args.anchor_num)
            np.random.shuffle(anchor_index)
    
            anchor_index = anchor_index[:100]


            anchor_index = torch.from_numpy(anchor_index).long().cuda()

            anchor_inputs = anchors_data[anchor_index,:,:,:]
            anchor_labels = anchor_Label[anchor_index,:]


            inputs, labels, index = data['image'], data['labels'], data['index']

            labels = labels.float()
            
            mask_flag = Variable(mask_labels[index], requires_grad=False)
            idx = (mask_flag>0)

            if index.shape[0] == args.batch_size:
                anchor_batch_S, anchor_batch_W = CalcSim(labels[idx,:].cuda(), anchor_labels.cuda())
                
                if inputs.size(3)==3:
                    inputs = inputs.permute(0,3,1,2)
                inputs = inputs.type(torch.FloatTensor)


                zcomp_h1 = z_h1[index.cuda(),:]
                zcomp_h2 = z_h2[anchor_index,:]

                labeled_batch_S, labeled_batch_W = CalcSim(labels[idx,:].cuda(), labels[idx,:].cuda())

                if use_gpu:
                    inputs = Variable(inputs.cuda(), requires_grad=False)
                    anchor_batch_S = Variable(anchor_batch_S.cuda(), requires_grad=False)
                    anchor_batch_W = Variable(anchor_batch_W.cuda(), requires_grad=False)
                    labeled_batch_S = Variable(labeled_batch_S.cuda(), requires_grad=False)
                    labeled_batch_W = Variable(labeled_batch_W.cuda(), requires_grad=False)


                # zero the parameter gradients
                optimizer.zero_grad()
                

                y_h1 = model(inputs)
                y_h2 = model(anchor_inputs)

                y = F.sigmoid(0.5*torch.matmul(y_h1, y_h2.permute(1,0)))


                loss,l_batch_loss, m_loss = criterion(y, y_h1, y_h2, anchor_batch_S, anchor_batch_W, labeled_batch_S, labeled_batch_W, zcomp_h1, zcomp_h2, mask_flag, u_w_m, epoch, num_bits)

                
                h1[index,:] = y_h1.data.clone()
                h2[anchor_index,:] = y_h2.data.clone()
                # backward+optimize
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()

                Z_h2 = alpha_1 * Z_h2+ (1. - alpha_1) * h2
                z_h2 = Z_h2 * (1. / (1. - alpha_1 ** (epoch + 1)))

        print("Epoch[{}]({}/{}) Loss_H: {:.4f}".format(epoch, iteration, len(dataloader),loss.item()))

        batch_time.update(time.time() - end)        
        end = time.time()

        print("batch time is:{}".format(batch_time.val))

        
        Z_h1 = alpha * Z_h1 + (1. - alpha) * h1
        z_h1 = Z_h1 * (1. / (1. - alpha ** (epoch + 1)))



  
        if epoch % 1 ==0:
            MAP = helpers.validate(model, dataloader_test, test_loader)

            print("Test image map is:{}".format(MAP))

            is_best = MAP > best_prec
            best_prec = max(best_prec, MAP)

            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            }, is_best, prefix=arch, num_bits = num_bits, filename=checkpoint_filename)


    return model


def main():
    args = parser.parse_args()
    pprint(args)

    # check and create directories
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    if not os.path.exists(args.log):
        os.makedirs(args.log)    
    


    print('==> Preparing data..')
    transformations_train = transforms.Compose([
        data.RandomTranslateWithReflect(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.472, 0.448, 0.410], [0.265, 0.261, 0.277])  # coco
        ])

    transformations_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.472, 0.448, 0.410], [0.265, 0.261, 0.277])  # coco
        ])
    
    
    dataset = ImageFolder()

    Label = dataset.imgs_labels
    
    retrieval_idx, test_idx = du.split_multi_idx(Label, select_num=1600)

    
    train_path = [dataset.imgs_path[jter] for jter in retrieval_idx]
    trainLabel = dataset.imgs_labels[retrieval_idx,:]

    test_path = [dataset.imgs_path[jter] for jter in test_idx]
    testLabel =  dataset.imgs_labels[test_idx,:]

    unlabeled_idx, labeled_idx = du.split_multi_idx(trainLabel, select_num=8000)

    n = trainLabel.shape[0]

    mask_labels = np.squeeze(np.zeros((n,1)))
    mask_labels[labeled_idx]=1


    dict_idx = du.generate_anchor_idx(trainLabel, labeled_idx, anchor_num=args.anchor_num)
    dict_path = [train_path[jter] for jter in dict_idx]
    dict_data_transform = folder.dataBags(Path=dict_path, Label=trainLabel[dict_idx,:], transform=transformations_train)


    train_data_transform = folder.dataBags(Path=train_path, Label=trainLabel, transform=transformations_train)
    #labeled_train_path = [train_path[jter] for jter in labeled_idx]
    #train_data_transform =  folder.dataBags(Path= labeled_train_path, Label=trainLabel[labeled_idx,:],transform=transformations_train)
    train_data_transform_test =  folder.dataBags(Path= train_path, Label=trainLabel, transform=transformations_test)
    test_data_transform = folder.dataBags(Path=test_path, Label=testLabel, transform=transformations_test)

    


    dict_loader = torch.utils.data.DataLoader(dict_data_transform, batch_size=args.anchor_num,
                                              shuffle=False, num_workers=args.workers)

    batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idx, labeled_idx, args.batch_size, 40)

    train_loader = torch.utils.data.DataLoader(train_data_transform,
                                               batch_sampler = batch_sampler,
                                               num_workers = args.workers,
                                               pin_memory=True)


    train_loader_test = torch.utils.data.DataLoader(train_data_transform_test, batch_size=100,
                                              shuffle=False, num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(test_data_transform, batch_size=100,
                                              shuffle=False, num_workers=args.workers)


    model = pre_train(train_loader, test_loader, dict_loader, train_loader_test, mask_labels, total_epochs = 50, learning_rate=1e-4, load_flag=0, use_gpu=True, seed=args.seed)

    
if __name__ == "__main__":
    sys.exit(main())