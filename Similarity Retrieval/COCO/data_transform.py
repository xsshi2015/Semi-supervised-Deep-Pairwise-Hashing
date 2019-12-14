from __future__ import print_function
from PIL import Image
import os
import os.path
import cv2
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class DataTransform(data.Dataset):
    def __init__(self, trainData=None, trainLabel=None, trainIndex= None,
                 transform=None, target_transform=None,
                 download=False):
 
        self.train_data = trainData
        self.train_labels = trainLabel
        self.train_index = trainIndex
        self.transform = transform

        # self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train_index is not None:
            img, target, idx = self.train_data[index], self.train_labels[index], self.train_index[index]
        else:
            img, target, idx = self.train_data[index], self.train_labels[index], index


        # import pdb; pdb.set_trace()

        
        if self.transform is not None:    
            img = cv2.resize(img, (224, 224))
            img = Image.fromarray(img)
            img = self.transform(img)
        
        sample = {'image': img, 'labels': target, 'index': idx}

        return sample


    def __len__(self):
        return len(self.train_data)

    
    