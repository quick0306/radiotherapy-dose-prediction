import os
import numpy as np
from os import listdir
import argparse
import logging
import os
import random
import numpy as np
from preprocessing.generate_data import get_plans

parser = argparse.ArgumentParser()
parser.add_argument('--dicom_path', type=str,
                    default='../Data/dicom_data/', help='dicom data folder')
parser.add_argument('--plan_save_path', type=str,
                    default='..Data/plans', help='The generated plan save path')
parser.add_argument('--dataset_save_path', type=str,
                    default='./Data/npy_dataset', help='the processed numpy array save path')
parser.add_argument('--section', type=str,
                    default='Lungs', help='which part of TMI plan the training focus on')
parser.add_argument('--section_size', type=tuple,
                    default=(27, 37.5, 50), help='the img dimention that we want to cut')
parser.add_argument('-- matrix_size', type=tuple,
                    default=(32, 96, 128), help='the processed data matrix size')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size for each generated dataset')

args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(args.dicom_path):
        raise SystemExit('dicom folder not exist')

    if not os.path.exists(args.plan_save_path):
        os.makedirs(args.plan_save_path)

    if not os.path.exists(args.dataset_save_path):
        os.makedirs(args.dataset_save_path)

    scans, dose_imgs =  get_plans(args.dicom_path, 
                                  section = args.section, 
                                  section_size = args.section_size, 
                                  matrix_size = args.matrix_size, 
                                  plan_save_path = args.plan_save_path, 
                                  dataset_save_path = args.dataset_save_path, 
                                  save_npy = True, 
                                  batch_size = args.batch_size)
    