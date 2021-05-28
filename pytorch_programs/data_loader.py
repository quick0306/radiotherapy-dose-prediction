import torch
import os
from torch.utils import data
import numpy as np
from config import *

class PredictionDataSet(data.Dataset):
    def __init__(self, 
                 data_folder,
                 transofrm = None):
        self.datafolder = data_folder
        self.batch_dirs = os.listdir(data_folder)
        self.transform = transofrm
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.dimention = len(standard_name)
    
    def __len__(self):
        return len(self.batch_dirs)
    
    def __getitem__(self, index: int):
        # select the sample
        batch_XY = np.load(self.datafolder+'/'+self.batch_dirs[index])
        x = np.squeeze(batch_XY[:,:,:,:,0:self.dimention], axis=0)
        y = np.squeeze(batch_XY[:,:,:,:,self.dimention:self.dimention+1], axis=0)
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        x = x.permute(3, 0, 1, 2)
        y = y.permute(3, 0, 1, 2)
        return x, y
