from PIL import Image
import glob
import torch
import torch.utils.data as data
# torch.utils.data.dataset is an abstract class representing a dataset
from torch.utils.data.dataset import Dataset
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html
import os
import torch
import numpy as np
import pandas as pd
import sys
import csv



'''
Pytorch uses datasets and has a very handy way of creating dataloaders in your main.py
Make sure you read enough documentation.
'''
class CIFAR(Dataset):
    """
    CIFAR dataset
    Implements Dataset (torch.utils.data.dataset)
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir (string): Directory with all the images
        """
        #gets the data from the directory
        self.image_list = glob.glob(data_dir)
        #calculates the length of image_list
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        """
        Lazily get the item at the index.
        """
        # Get image name from the pandas df
        single_image_path = self.image_list[index]

        
        # Open image
        image = Image.open(single_image_path)
        
        # Convert to numpy, dim = 28x28
        image_np = np.asarray(image)/255 
        
        # Do some operations on image
        
        image_0 = rotate_img(image_np, 0)
        
        image_90 = rotate_img(image_np, 90)
        
        image_180 = rotate_img(image_np, 180)
        
        image_270 = rotate_img(image_np, 270)
        
        # print(image_270.shape)
        
        image_stack = np.stack((image_0, image_90, image_180, image_270))
        
        # print(image_stack.shape)
        
        # One hot encoding for the labels
        label_stack = np.stack((np.array([1,0,0,0]),
                                np.array([0,1,0,0]),
                                np.array([0,0,1,0]),
                                np.array([0,0,0,1])))
        
        # print(label_stack.shape)
        
        # Convert numpy to a tensor
        image_tensor = torch.from_numpy(image_np).float()
        
        label_tensor = torch.from_numpy(label_stack).float()
        
        return (image_tensor, label_tensor)

    def __len__(self):
        return self.data_len