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
        
        image_0 = rotate_img(image_np, 0).reshape(3,32,32)
        image_90 = rotate_img(image_np, 90).reshape(3,32,32)
        image_180 = rotate_img(image_np, 180).reshape(3,32,32)
        image_270 = rotate_img(image_np, 270).reshape(3,32,32)
        
        # print(image_270.shape)
        
        image_stack = np.stack((image_0, image_90, image_180, image_270))
        # print(image_stack.shape)
        
        # print(image_stack.shape)
        
        # One hot encoding for the label
        label_stack = np.stack((0,1,2,3))
        # label_stack = np.stack((np.array([1,0,0,0]), np.array([0,1,0,0]), np.array([0,0,1,0]), np.array([0,0,0,1])))
        
        # print(label_stack.shape)
        
        
        # Convert numpy to a tensor
        image_tensor = torch.from_numpy(image_stack).float()
        
        label_tensor = torch.from_numpy(label_stack).float()
        label_tensor = label_tensor.type(torch.LongTensor)

        
        # print(image_tensor.shape)
        # print(label_tensor.shape)
        return (image_tensor, label_tensor)

    def __len__(self):
        return self.data_len



"""
Takes in an image and a rotation. Returns the the image with the rotation applied.
"""
def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')