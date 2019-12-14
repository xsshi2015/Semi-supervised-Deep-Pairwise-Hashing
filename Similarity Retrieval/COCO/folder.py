"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import cv2
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from multilabel_imgfolder import default_loader

from PIL import Image


class dataBags(data_utils.Dataset):
    def __init__(self, Path=None, Label=None, transform=None, seed=1):
        self.path = Path 
        self.labels = Label
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        img = default_loader(self.path[index])
        label = self.labels[index]

        img = np.array(img)
        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(img)
        if self.transform is not None:    
            img = self.transform(img)
        
        sample = {'image': img, 'labels': label, 'index': index}

        return sample


