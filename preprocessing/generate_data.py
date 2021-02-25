import os
import numpy as np
from os import listdir
from scipy.io import loadmat
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import h5py
import pydicom as dicom
import matplotlib.pyplot as plt
import pydicom as dicom
import dicom_contour.contour as dcm
from dicom_contour.contour import get_contour_file,get_roi_names, coord2pixels, cfile2pixels, plot2dcontour, slice_order, get_contour_dict, get_data,  create_image_mask_files, get_dose_on_ct
from dicom_contour.contour import get_mask
from collections import defaultdict 
from dicom_contour.dose import ArrayVolume, build_dose_volume
from fuzzywuzzy import fuzz, process
import random
import math
import time
import pickle
from preprocessing.get_plan_from_dicom import Plan
from preprocessing.timer_class import Timer
from config import *
from util import *


def get_plans(dicom_path, section = 'Lungs', section_size = (27, 37.5, 50), matrix_size = (16, 96, 128), plan_save_path = 'Data/plans', dataset_save_path = 'Data/npy_dataset', save_npy = True, batch_size = 1):
    """ Get the dataset from the folder of dicom files, for each patient, we will 
        get the plan class, and resample and re-cut the plan, then same the plan to a folder
        section size and matrix_size --> z, x, y 
        section size: tuple in unit cm, it is just a guess of patient body size. (z,x,y)=(27,37.5,75)cm and pixel = (7.5, 1.5625,1.5625)mm --> matrix = 36x240x480
        resample the matrix size to -> (16, 64, 128) for each patient
    """
    # Create dateset:
    data_folder = dicom_path+'/'
    data_dirs = listdir(data_folder)
    scans = []  # scans is used to store the masks 
    dose_imgs = [] # dose_imgs is used to store the doses
    i = 0
    for folder in data_dirs:
        if os.path.isdir(data_folder+folder):
            print('work on patient ', folder)
            subfolder = data_folder+folder +'/'
            
            plan = Plan()
            plan.get_plan_mask(subfolder)
            plan.rename(standard_name)
            # get the z, x, y range of the body part and the focus secation (Lungs, liver...)
            zmin_body,zmax_body,xmin_body,xmax_body,ymin_body,ymax_body = plan.structure_range('BODY')
            zmin,zmax,xmin,xmax,ymin,ymax = plan.structure_range(section)
            origin = [math.floor((zmin+zmax)/2), math.floor((xmin_body+xmax_body)/2), math.floor((ymin_body+ymax_body)/2)]
            z_scope = int(section_size[0]*10 / plan.slice_thickness)
            x_scope = int(section_size[1]*10 / plan.Row_Spacing)
            y_scope = int(section_size[2]*10 / plan.Col_Spacing)

            # we need to do some adjustment before image cut to make sure we cut correctly
            if(x_scope >= np.shape(plan.img_volume)[1]):
                x_dim = [0,np.shape(plan.img_volume)[1]]
            else:
                x_dim = [int(origin[1] - x_scope/2), int(origin[1] + x_scope/2)]

            if(y_scope >= np.shape(plan.img_volume)[2]):
                y_dim = [0,np.shape(plan.img_volume)[2]]
            else:
                y_dim = [int(origin[2] - y_scope/2), int(origin[2] + y_scope/2)]
            
            z_dim = [int(origin[0] - z_scope/2), int(origin[0] + z_scope/2)]

            print('x_dim, y_dim, z_dim', x_dim, y_dim, z_dim)

            plan.img_cut(x_dim, y_dim, z_dim)

            #resize the matrix after cut the image to appropriate size
            plan.resample(x_dim = matrix_size[1], y_dim = matrix_size[2], z_dim = matrix_size[0])

            # save the processed plan to pickle file to plan save_path
            if not os.path.exists(plan_save_path):
                os.makedirs(plan_save_path)
           # file_name = plan_save_path + '/' + folder + '.pickle'
           # with open(file_name, "wb") as file_:
           #     pickle.dump(plan, file_, -1)

            # create a npy matrix for structures, return [z, x, y, channel]
            structure_masks = get_masks(plan, standard_name)
            structure_masks = np.expand_dims(structure_masks, axis = 0)
            # get the dose matrix (1, z, x, y)
            dose_img = np.array(plan.dose_volume).astype('float32')
            dose_img = np.expand_dims(dose_img, axis = 3)
            dose_img = np.expand_dims(dose_img, axis = 0)
            print('scan shape and dose shape=',structure_masks.shape, dose_img.shape)
            del plan
            # conbine the patients data
            if scans == []:
                scans = structure_masks
                dose_imgs = dose_img
            else:
                print('add new patient data')
                scans= np.concatenate((scans,structure_masks),axis=0)
                dose_imgs=np.concatenate((dose_imgs,dose_img),axis=0)

    print('Structure Masks Data Shape: ' + str(scans.shape))
    print('Dose Data Shape: ' + str(dose_imgs.shape))
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
    if save_npy:
        np.save(dataset_save_path+'/structures.npy', scans)
        np.save(dataset_save_path+'/dose.npy', dose_imgs)
        print('NPY dataset saved!')

    # save all patient data to different batches for training and saved to save folder, the masks and dose are concate to 
    # final data is [batch_size, z,x,y, n_s+1] --> the '+1' is dose   
    for batch_i in range(0, dose_imgs.shape[0], batch_size):
        batch_npy = np.concatenate((scans[batch_i:batch_i+batch_size],dose_imgs[batch_i:batch_i+batch_size]),axis=4)
        batch_npy = np.array(batch_npy)
        np.save(dataset_save_path+'/batch_{0}.npy'.format(batch_i), batch_npy)

    return scans, dose_imgs


def get_masks(plan, standard_name):
    masks = []
    for s in standard_name:
        mask = plan.structures[s]['mask']
        mask = np.expand_dims(mask,axis=3) # extend the mask dimention to 4
        if masks == []:
            masks = mask
        else:
            masks =  np.concatenate((masks, mask), axis = 3)
    print('one structure Mask Data Shape: ' + str(masks.shape))
    return masks



def get_plans_unit_test():

    dicom_path = './dicom_data_test/'
    plan_save_path = './Data/plans_test'
    dataset_save_path = './Data/npy_dataset_test'

    get_plans(dicom_path, section = 'Lungs', plan_save_path = 'Data/plans_test', dataset_save_path = 'Data/npy_dataset_test', save_npy = True, batch_size = 1)
    scans, dose_imgs = get_plans(dicom_path, section = 'Lungs', section_size = section_size, matrix_size = matrix_size, plan_save_path = plan_save_path, dataset_save_path = dataset_save_path, save_npy = True, batch_size = 1)
    return scans, dose_imgs



if __name__ == '__main__':
    plan_save_path = 'Data/plans'
    dataset_save_path = 'Data/npy_dataset'
    