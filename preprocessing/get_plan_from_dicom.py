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
from preprocessing.timer_class import Timer
from config import *
from util import *


class Plan(object):
    def __init__(self, Origin = None,X_grid= None, Y_grid= None, Z_grid = None,\
                 Col_Spacing= None, Row_Spacing= None, slice_thickness = None,\
                 img_volume = None, structures = None, dose_volume = None):
        self.Origin = Origin
        self.X_grid = X_grid
        self.Y_grid = Y_grid
        self.Z_grid = Z_grid
        self.Col_Spacing = Col_Spacing
        self.Row_Spacing = Row_Spacing
        self.slice_thickness = slice_thickness
        self.img_volume= img_volume
        self.structures= structures
        self.dose_volume= dose_volume
        
    def get_plan_mask(self, patient_path):
        """ read the folder...
        """
        path = patient_path
        contour_file = get_contour_file(path)
        if path[-1] != '/': path += '/'
        # get the dose_volume
        f = dicom.dcmread(path + contour_file)
        RS_name = f.StructureSetLabel
        print("work on RS structure ",RS_name)
        structures = defaultdict()
        
        
        for s in os.listdir(path):
            img = dicom.dcmread(path + '/' + s)
            if hasattr(img, 'pixel_array'):  # to ensure not to read contour file
                img_arr = img.pixel_array
                # physical distance between the center of each pixel
                x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
                slice_thickness = float(img.SliceThickness)
                self.Col_Spacing = x_spacing
                self.Row_Spacing = y_spacing
                self.slice_thickness = slice_thickness
                break
        print('start structure extraction')
        t = Timer()
        t.start()
        roi_seq_names = [roi_seq.ROIName for roi_seq in list(f.StructureSetROISequence)]
        for i in range(0,len(roi_seq_names)):
            roi_name = roi_seq_names[i]
            ## check if this contour is empty contour, if empty, skip it
            RTV_temp = f.ROIContourSequence[i]
            if not hasattr(RTV_temp, 'ContourSequence'):
                continue
            structures[roi_name] = {}
            img_voxel, contour_outline, mask_voxel = get_data(path, contour_file, i)
            structures[roi_name]["mask"] = mask_voxel
            structures[roi_name]["contour"] = contour_outline
            structures[roi_name]["index"] = i
            self.img_volume = img_voxel
        print('finish extract contour')
        t.stop()
        print('start extract dose')
        t.start()
        for i in range(0,len(roi_seq_names)):
            roi_name = roi_seq_names[i] 
            RTV_temp = f.ROIContourSequence[i]
            if not hasattr(RTV_temp, 'ContourSequence'):
                continue
            img_voxel, mask_voxel, dose_voxel = get_dose_on_ct(path, contour_file, i)
            self.dose_volume = dose_voxel
            break
        print('finish extract dose')
        t.stop()
        self.structures = structures
        return

    def rename(self, standard_name = standard_name):
        dict_name = {}
        dict_origin = {}
        standard_name_lower = []
        print(PTV_VExP_Bone)
        is_GI_Upper = False
        for structure in list(self.structures.keys()):
            if structure == 'GI_Upper': is_GI_Upper = True
            print(structure)
            if(structure in PTV_VExP_Bone):
                print('find PTV_VExP')
                self.structures['PTV_VExP'] = self.structures.pop(structure)
            if(structure == 'Lungs_Total'):
                print('Lungs')
                self.structures['Lungs'] = self.structures.pop(structure)

        if not is_GI_Upper:
            self.structures['GI_Upper'] = self.structures.pop('Stomach')


        for s in standard_name:
            standard_name_lower.append(s.lower())
            dict_name[s.lower()] = s

        if (self.Col_Spacing == None):
            raise ValueError
         
        origin_names = list(self.structures.keys())
        origin_names_lower = []
        for s in origin_names:
            origin_names_lower.append(s.lower())
            dict_origin[s.lower()] = s

        for s in standard_name:
            cand = process.extractOne(s.lower(), origin_names_lower, scorer=fuzz.ratio)
            if(cand[1]>=80):
                self.structures[s] = self.structures.pop(dict_origin[cand[0]])
                print('{} <--------> {}'.format(s, dict_origin[cand[0]]))
            else:
                if (s == 'Breasts'):
                    self.structures['Breasts'] = {}
                    self.structures['Breasts']['mask'] = np.zeros(np.shape(self.img_volume))
                    self.structures['Breasts']['contour'] = np.zeros(np.shape(self.img_volume))
                else:
                    raise ValueError("cannot find the organ matchs", s)
        return

    def plot_3d_img(self, organ = 'Lungs'):
        if (self.img_volume.any() == None):
            return
        fig, (ax1, ax2, ax3, ax4)= plt.subplots(4, 1)
        max_img = np.max(self.img_volume.flatten())
        max_dose = np.max(self.dose_volume.flatten())
        tracker1 = IndexTracker(ax1, self.img_volume, fig,0,max_img)
        fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
        tracker2 = IndexTracker(ax2, self.dose_volume, fig,0,max_dose)
        fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
        tracker3 = IndexTracker(ax3, self.structures[organ]['mask'], fig,0,1)
        fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)
        tracker4 = IndexTracker(ax4, self.structures[organ]['contour'], fig,0,1)
        fig.canvas.mpl_connect('scroll_event', tracker4.onscroll)
        plt.show()

    def plot_DVH(self, structure_list):
        dose_flat = self.dose_volume.flatten()
        max_dose = np.max(dose_flat)
        # define the metrics that we want to display
        Dmean = {}; Dmax = {}; D95 = {}; D5 = {}; D98 = {}; D2 = {}
        DVH_all = defaultdict()
        # define the parameter for DVH
        DVH_bin = 2000
        DVH_inv = max_dose*1.0/DVH_bin
        dose_bin = np.zeros(DVH_bin)
        dose_bin = np.arange(0,DVH_bin)*DVH_inv
        dose_bin1 = np.arange(-1,DVH_bin)*DVH_inv
        for s in structure_list:
            self.structures[s]['mask'] = self.structures[s]['mask']>0
            # generate the mask for specific organ
            mask_organ = np.squeeze(self.structures[s]['mask'])
            mask_organ = mask_organ.flatten()
            volume = len(mask_organ==True)
            # make as dose_organ
            dose_masked =  np.ma.masked_where(mask_organ==False, dose_flat)
            dose_organ = dose_masked.compressed()
            if np.asarray(dose_organ).size == 0:
                print(s, ' not exist')
                continue
            Dmean[s] = dose_organ.mean()
            Dmax[s] = dose_organ.max()
            # start to calculate the DVH
            DVH = np.zeros(DVH_bin)
            DVH_diff, bin_edges = np.histogram(dose_organ,dose_bin1)
            DVH = np.cumsum(DVH_diff)
            DVH = 1 - DVH/DVH.max()
            index = np.argmin(np.abs(DVH-0.95))
            D95[s] = index*DVH_inv
            index = np.argmin(np.abs(DVH-0.98))
            D98[s] = index*DVH_inv
            index = np.argmin(np.abs(DVH-0.05))
            D5[s] = index*DVH_inv
            index = np.argmin(np.abs(DVH-0.02))
            D2[s] = index*DVH_inv
            DVH_all[s] = DVH
            print(s)
            print('True mean organ dose is: ', dose_organ.mean())
            print('True max organ dose is: ', dose_organ.max())
        fig = plt.figure()
        for s in structure_list:
            if s not in DVH_all.keys():
                continue
            r = random.uniform(0, 1); g = random.uniform(0, 1); b = random.uniform(0, 1)
            plt.plot(dose_bin,DVH_all[s]*100, color = (r,g,b), linewidth=1)
        
        plt.ylabel('volume %')
        plt.legend(structure_list,bbox_to_anchor=(1.1, 1.05),prop={'size': 6}) 
        plt.show()
        return dose_bin, DVH_all, Dmean, Dmax, D95, D5, D98, D2

    def resample(self, x_dim, y_dim, z_dim):
        x_size = np.shape(self.img_volume)[1]
        y_size = np.shape(self.img_volume)[2]
        z_size = np.shape(self.img_volume)[0]
        x_ratio = x_dim/x_size; y_ratio = y_dim/y_size; z_ratio = z_dim/z_size
        self.Col_Spacing = self.Col_Spacing * y_ratio
        self.Row_Spacing = self.Row_Spacing * x_ratio
        self.slice_thickness = self.slice_thickness * z_ratio
        self.img_volume = zoom(self.img_volume, (z_ratio, x_ratio, y_ratio))
        self.dose_volume = zoom(self.dose_volume, (z_ratio, x_ratio, y_ratio))
        for s in list(self.structures.keys()):
            self.structures[s]['mask'] = zoom(self.structures[s]['mask'], (z_ratio, x_ratio, y_ratio))>0
            self.structures[s]['contour'] = zoom(self.structures[s]['contour'], (z_ratio, x_ratio, y_ratio))>0
    
        return 
    
    def structure_range(self, structure):
        if(structure not in self.structures.keys()):
            print('The structure is not in list!')
            return
        mask =  self.structures[structure]['mask']>0
        (z,x,y) = np.where(mask)
        return min(z), max(z), min(x), max(x), min(y), max(y)

    def img_cut(self, x_dim, y_dim, z_dim):
        # origin = [z, x, y]
       # x_dim = math.floor(x_dim/2)*2; y_dim = math.floor(y_dim/2)*2; z_dim = math.floor(z_dim/2)*2
        self.img_volume = self.img_volume[z_dim[0]:z_dim[1], x_dim[0]:x_dim[1], y_dim[0]:y_dim[1]]
    
        self.dose_volume = self.dose_volume[z_dim[0]:z_dim[1], x_dim[0]:x_dim[1], y_dim[0]:y_dim[1]]
    
        for s in list(self.structures.keys()):
            self.structures[s]['mask'] = self.structures[s]['mask'][z_dim[0]:z_dim[1], x_dim[0]:x_dim[1], y_dim[0]:y_dim[1]]
            self.structures[s]['contour'] = self.structures[s]['contour'][z_dim[0]:z_dim[1], x_dim[0]:x_dim[1], y_dim[0]:y_dim[1]]
        return
    

    def normalize(self, structure, normal_mean):
        structure_list = [structure]
        dose_bin, DVH_all, Dmean, Dmax, D95, D5, D98, D2 = self.plot_DVH(structure_list)
        scale = normal_mean*1.0/Dmean[structure]
        self.dose_volume = self.dose_volume * scale
        return


### Below are the test functions for used in Plan class

def plot_3d_img(plan, organ = 'Lungs'):
        if (plan.img_volume.any() == None):
            return
        fig, (ax1, ax2, ax3, ax4)= plt.subplots(4, 1)
        max_img = np.max(plan.img_volume.flatten())
        max_dose = np.max(plan.dose_volume.flatten())
        tracker1 = IndexTracker(ax1, plan.img_volume, fig,0,max_img)
        fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
        tracker2 = IndexTracker(ax2, plan.dose_volume, fig,0,max_dose)
        fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
        tracker3 = IndexTracker(ax3, plan.structures[organ]['mask'], fig,0,1)
        fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)
        tracker4 = IndexTracker(ax4, plan.structures[organ]['contour'], fig,0,1)
        fig.canvas.mpl_connect('scroll_event', tracker4.onscroll)
        plt.show()

def plot_DVH(plan, structure_list):
    dose_flat = plan.dose_volume.flatten()
    max_dose = np.max(dose_flat)
    # define the metrics that we want to display
    Dmean = {}; Dmax = {}; D95 = {}; D5 = {}; D98 = {}; D2 = {}
    DVH_all = defaultdict()
    # define the parameter for DVH
    DVH_bin = 2000
    DVH_inv = max_dose*1.0/DVH_bin
    dose_bin = np.zeros(DVH_bin)
    dose_bin = np.arange(0,DVH_bin)*DVH_inv
    dose_bin1 = np.arange(-1,DVH_bin)*DVH_inv

    for s in structure_list:
        plan.structures[s]['mask'] = plan.structures[s]['mask']>0
        # generate the mask for specific organ
        mask_organ = np.squeeze(plan.structures[s]['mask'])
        mask_organ = mask_organ.flatten()
        volume = len(mask_organ==True)
        # make as dose_organ
        dose_masked =  np.ma.masked_where(mask_organ==False, dose_flat)
        dose_organ = dose_masked.compressed()
        Dmean[s] = dose_organ.mean()
        Dmax[s] = dose_organ.max()

        # start to calculate the DVH
        DVH = np.zeros(DVH_bin)
        DVH_diff, bin_edges = np.histogram(dose_organ,dose_bin1)
        DVH = np.cumsum(DVH_diff)
        DVH = 1 - DVH/DVH.max()
        index = np.argmin(np.abs(DVH-0.95))
        D95[s] = index*DVH_inv
        index = np.argmin(np.abs(DVH-0.98))
        D98[s] = index*DVH_inv
        index = np.argmin(np.abs(DVH-0.05))
        D5[s] = index*DVH_inv
        index = np.argmin(np.abs(DVH-0.02))
        D2[s] = index*DVH_inv
        DVH_all[s] = DVH
        print(s)
        print('True mean organ dose is: ', dose_organ.mean())
        print('True max organ dose is: ', dose_organ.max())
    fig = plt.figure()
    for s in structure_list:
        r = random.uniform(0, 1); g = random.uniform(0, 1); b = random.uniform(0, 1)
        plt.plot(dose_bin,DVH_all[s]*100, color = (r,g,b), linewidth=1)
        
    plt.ylabel('volume %')
    plt.legend(structure_list,bbox_to_anchor=(1.1, 1.05),prop={'size': 6}) 
    plt.show()
    return DVH_all, Dmean, Dmax, D95, D5, D98, D2
        

def structure_range(plan, structure):
    if(structure not in plan.structures.keys()):
        print('The structure is not in list!')
        return
    mask =  plan.structures[structure]['mask']>0
    (z,x,y) = np.where(mask)
    return min(z), max(z), min(x), max(x), min(y), max(y)


def img_cut(plan, x_dim, y_dim, z_dim, origin):
    # origin = [z, x, y]
    x_dim = math.floor(x_dim/2)*2; y_dim = math.floor(y_dim/2)*2; z_dim = math.floor(z_dim/2)*2
    plan.img_volume = plan.img_volume[int(origin[0]-z_dim/2):int(origin[0]+z_dim/2), int(origin[1]-x_dim/2):int(origin[1]+x_dim/2),\
        int(origin[2]-y_dim/2):int(origin[2]+y_dim/2)]
    
    plan.dose_volume = plan.dose_volume[int(origin[0]-z_dim/2):int(origin[0]+z_dim/2), int(origin[1]-x_dim/2):int(origin[1]+x_dim/2),\
        int(origin[2]-y_dim/2):int(origin[2]+y_dim/2)]
    
    for s in list(plan.structures.keys()):
        plan.structures[s]['mask'] = plan.structures[s]['mask'][int(origin[0]-z_dim/2):int(origin[0]+z_dim/2), int(origin[1]-x_dim/2):int(origin[1]+x_dim/2),\
            int(origin[2]-y_dim/2):int(origin[2]+y_dim/2)]
        
        plan.structures[s]['contour'] = plan.structures[s]['contour'][int(origin[0]-z_dim/2):int(origin[0]+z_dim/2), int(origin[1]-x_dim/2):int(origin[1]+x_dim/2),\
            int(origin[2]-y_dim/2):int(origin[2]+y_dim/2)]
    
    return plan
    
    
def resample(plan, x_dim=128, y_dim=128, z_dim=64):
    x_size = np.shape(plan.img_volume)[1]
    y_size = np.shape(plan.img_volume)[2]
    z_size = np.shape(plan.img_volume)[0]
    x_ratio = x_dim/x_size; y_ratio = y_dim/y_size; z_ratio = z_dim/z_size
    plan.Col_Spacing = plan.Col_Spacing * y_ratio
    plan.Row_Spacing = plan.Row_Spacing * x_ratio
    plan.slice_thickness = plan.slice_thickness * z_ratio
    plan.img_volume = zoom(plan.img_volume, (z_ratio, x_ratio, y_ratio))
    plan.dose_volume = zoom(plan.dose_volume, (z_ratio, x_ratio, y_ratio))
    for s in list(plan.structures.keys()):
        plan.structures[s]['mask'] = zoom(plan.structures[s]['mask'], (z_ratio, x_ratio, y_ratio))>0
        plan.structures[s]['contour'] = zoom(plan.structures[s]['contour'], (z_ratio, x_ratio, y_ratio))>0
    
    return plan


def plan_unit_test():
    path = './dicom_data/TMI_REAM'
    plan = Plan()
    plan.get_plan_mask(path)
    plan.rename(standard_name = standard_name)
    print(plan.structures.keys())
  #  plan.plot_3d_img(organ = 'Ring_1')
    return plan




    
