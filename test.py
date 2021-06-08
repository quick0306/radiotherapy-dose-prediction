import os
import numpy as np
from os import listdir
import argparse
import logging
import os
import random
import numpy as np
from .networks.unet_model import UNet
from .networks.AttUnet_model import Att_UNet
from .networks.trainer import Trainer
from torchsummary import summary
from .networks.data_loader import PredictionDataSet
from torch.utils import data
import torch
from .networks.inference import predict_batch, metrics_summary, predict_evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str,
                    default='../Data/npy_dataset/test/', help='test data path folder')

parser.add_argument('--model_weight_path', type=str,
                    default='../models/best_AttUnet_model.pt', help='model weight path')

parser.add_argument('--epochs', type=int,
                    default=200, help='the epochs for traning')
parser.add_argument('--lr_scheduler', type=int,
                    default=None, help='learning rate schedular')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size for training and validation')

args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(args.test_data_path):
        raise SystemExit('dicom folder not exist')

    if args.test_data_path[-1] != '/':
        test_path = args.test_data_path + '/'
    data_folder = test_path
    data_dirs = listdir(data_folder)
    # load model once 
    model = Att_UNet(n_channels=12, n_classes=1).to(device)
    model_weights = torch.load(args.model_weight_path)
    model.load_state_dict(model_weights)
    # start to go-over all patients in the test_path subfolders
    metrics_all = {}
    for folder in data_dirs:
        if os.path.isdir(data_folder+folder):
            print('work on test patient ', folder)
            test_patient_path = data_folder+folder
            metrics = predict_evaluation(model, test_patient_path)
            metrics_all[folder] = metrics


    metrics_all = predict_batch(args.model_weight_path, args.test_data_path)
    df_mean, df_std  = metrics_summary(metrics_all)
    