import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from utils import GaussianNoise, savetime, save_exp



class CNN(nn.Module):
    def __init__(self, model_name, bit, class_num):
        super(CNN, self).__init__()
        if model_name == 'alexnet':
            original_model = models.alexnet(pretrained=True)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl1.weight = original_model.classifier[1].weight
            cl1.bias = original_model.classifier[1].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[4].weight
            cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, bit),
            )
            self.model_name = 'alexnet'

        if model_name == 'resnet18':
            original_model = models.resnet18(pretrained=True)
            self.conv1 = original_model.conv1
            self.bn1 = original_model.bn1
            self.relu = original_model.relu
            self.maxpool = original_model.maxpool

            self.layer1 = original_model.layer1
            self.layer2 = original_model.layer2
            self.layer3 = original_model.layer3
            self.layer4 = original_model.layer4

            self.avg = original_model.avgpool
            self.classifier = nn.Linear(512, bit)
            self.model_name = 'resnet18'

    def forward(self, x):
        if self.model_name=='alexnet':
            f = self.features(x)
            f = f.view(f.size(0), 256 * 6 * 6)
        else:
            f = self.conv1(x)
            f = self.bn1(f)
            f = self.relu(f)
            f = self.maxpool(f)

            f = self.layer1(f)
            f = self.layer2(f)
            f = self.layer3(f)
            f = self.layer4(f)
            f = self.avg(f)
            f = f.view(f.size(0), -1)
        y_h = F.tanh(self.classifier(f))

        return y_h



class Custom_Loss(torch.nn.Module):
    def __init__(self, num_bits=128):
        super(Custom_Loss, self).__init__()
        self.num_bits = num_bits

    def forward(self, out, out_h1, out_h2, target, W, batch_target, batch_W, z_h1,z_h2, mask_flag, w_mse, epoch, bit):
        
        def cross_entropy_objective(Y_prob, Y, W, mask_flag):
            cond = (mask_flag > 0)
            nnz = torch.nonzero(cond)
            nbsup = len(nnz)
            if nbsup > 0:
                s_Y_prob = torch.clamp(Y_prob[cond,:], min=1e-5, max=1. - 1e-5)
                neg_log_likelihood = -1. * Y * torch.log(s_Y_prob) + W*(1. - Y) * torch.log(1. - s_Y_prob)  # negative log bernoulli 
                loss = neg_log_likelihood.sum()/neg_log_likelihood.data.nelement()

                if torch.isnan(loss):
                    import pdb; pdb.set_trace()

                return loss

        def l_cross_entropy_pair(Y_h, S,W, mask_flag, epoch, bit):
            if epoch>0:
                cond = (mask_flag > 0)
                nnz = torch.nonzero(cond)
                nbsup = len(nnz)
                if nbsup > 0:
                    bn= nbsup//2
                    Y_prob = F.sigmoid(48/bit*0.1*torch.matmul(Y_h[cond,:], Y_h[cond,:].permute(1,0)))

                    s_Y_prob = torch.clamp(Y_prob[:bn,nbsup-bn:], min=1e-5, max=1. - 1e-5)
                    S1 = S[:bn,nbsup-bn:]
                    W1 = W[:bn,nbsup-bn:]

                    neg_log_likelihood = -1. * S1* torch.log(s_Y_prob) + W1*(1. - S1) * torch.log(1. - s_Y_prob)  # negative log bernoulli 
                    neg_log_likelihood = neg_log_likelihood**2
                    loss0 = neg_log_likelihood.sqrt().sum()/neg_log_likelihood.data.nelement()

                    return loss0        
            else:
                return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)


        def e_cross_entropy_mse(out_h1, out_h2, z_h1, z_h2, epoch, bit):
            if epoch>0:
                bn_1 = out_h1.size(1)//2
                bn_2 = out_h2.size(1)//2
                
                Y_prob = F.sigmoid(48/bit*0.1*torch.matmul(out_h1, out_h2.permute(1,0)))
                S_prob = F.sigmoid(48/bit*0.1*torch.matmul(z_h1, z_h2.permute(1,0)))

                loss0 = F.mse_loss(Y_prob, S_prob, size_average=True)

                return loss0        
            else:
                return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False)



        sup_loss = cross_entropy_objective(out, target, W, mask_flag)
        m_loss = e_cross_entropy_mse(out_h1, out_h2, z_h1, z_h2, epoch, bit)
        l_sup_loss_pair = l_cross_entropy_pair(out_h1, batch_target, batch_W, mask_flag, epoch, bit)

        return sup_loss + w_mse*m_loss + l_sup_loss_pair, l_sup_loss_pair, m_loss   
