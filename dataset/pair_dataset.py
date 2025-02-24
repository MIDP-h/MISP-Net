import os
import sys
import random

# import cv2
import numpy as np

import torch
from torch.utils.data import Dataset 


class PairDataset(Dataset):

    def __init__(self, backend_left, backend_right=None):
        backend_right = backend_left
        # self.symmetry = True
        self.backend_left = backend_left
        self.backend_right = backend_right
        self.data_len_left = len(backend_left)
        self.data_len_right = len(backend_right)
  
    def __len__(self):
        return int(self.data_len_left*9)

    def to_pair_index(self, index):
        index_left = int(index // 9)
        idx = index - index_left * 9
        index_right = (index_left//10)*10 + idx
        index_right = index_right if index_right < index_left else index_right+1
        return index_left, index_right
    

    def __getitem__(self, index):
        index_left, index_right = self.to_pair_index(index)
        data_left = self.backend_left[index_left]
        data_right = self.backend_right[index_right]
        
        data = [data_left, data_right]
        data = [ torch.from_numpy(d).float() for d in data ]

        data_left = data[0]
        data_right = data[1]
        
        return data_left, data_right