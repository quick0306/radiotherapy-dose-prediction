import os
import sys
import numpy as np
#from get_dataset import get_scan, scan_pading, save_seg_imgs
from tensorflow.keras.models import model_from_json, load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from preprocessing.get_plan_from_dicom import Plan
from config import *
import pandas as pd
from IPython.display import display
from collections import defaultdict
from util import * 

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def predict_evaluation(model, test_patient_path):
    ''' predict the dose and evaluation for individual patient
    '''
    if test_patient_path[-1] != '/':
        test_patient_path = test_patient_path + '/'
    subfolder = test_patient_path
    test_npy = listdir(subfolder)
    file_name = test_npy
    plan = Plan()
    plan.structures = defaultdict()
    plan_hat = Plan() # plan_hat is the predicted dose of plan
    plan_hat.structures = defaultdict()

    # start to assign the metrix to plan attributes
    batch_XY = np.load(subfolder+test_npy[0])
    s_n = len(standard_name)  # s_n is the number of structures
    X =  batch_XY[:,:,:,:,0:s_n]
    Y =  batch_XY[:,:,:,:,s_n:s_n+1]

    # load model and predict
    Y_hat = model.predict(X)
   # Y_hat = np.maximum(np.subtract(Y, 3), 0)  # shift dose 3 Gy
    dose_hat = np.squeeze(Y_hat)
    plan_hat.dose_volume = dose_hat
    dose_true = np.squeeze(Y) # get the dose matrix of [Z,X,Y]
    plan.dose_volume = dose_true
    for i in range(0,s_n):
        mask = np.squeeze(X[:,:,:,:,i])
        s = standard_name[i]
        plan.structures[s] = defaultdict()
        plan.structures[s]['mask'] = mask
        plan_hat.structures[s] = defaultdict()
        plan_hat.structures[s]['mask'] = mask    
    
    
    batch_XY_hat = np.concatenate((X, Y_hat), axis = -1)
    np.save(test_patient_path+'hat_'+ test_npy[0], batch_XY_hat)
    metrics = evaluate(plan, plan_hat)
    return metrics

def evaluate(plan, plan_hat, structure_list = standard_name):
    ''' display the dose difference and DVH difference
        return several metrics, D95, D2, Dmax, Dmean
    '''
    # get dose diffrerence
    dose = plan.dose_volume
    dose_hat = plan_hat.dose_volume
    dose_diff = np.add(np.multiply(-1,dose), dose_hat)
    # get DVH metrics
    dose_bin, DVH_all, Dmean, Dmax, D95, D5, D98, D2 = plan.plot_DVH(structure_list)
    dose_bin_hat, DVH_all_hat, Dmean_hat, Dmax_hat, D95_hat, D5_hat, D98_hat, D2_hat = plan_hat.plot_DVH(structure_list)


    ## start display the dose difference    
    fig, (ax1, ax2, ax3)= plt.subplots(3, 1)
    max_dose = np.max(dose.flatten())
    tracker1 = IndexTracker(ax1, dose, fig,0,max_dose)
    fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
    tracker2 = IndexTracker(ax2, dose_hat, fig,0,max_dose)
    fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
    tracker3 = IndexTracker(ax3, dose_diff, fig,dose_diff.min(),dose_diff.max())
    fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)
    plt.show()

    ## show DVHs on one image
    fig = plt.figure()
    structure_legend = []
    for s in structure_list:
        if s not in DVH_all.keys():
            continue
        r = random.uniform(0, 1); g = random.uniform(0, 1); b = random.uniform(0, 1)
        plt.plot(dose_bin,DVH_all[s]*100, color = (r,g,b), linewidth=1)
        plt.plot(dose_bin_hat,DVH_all_hat[s]*100, color = (r,g,b), linewidth=1, linestyle='dashed')
        structure_legend.append(s)
        structure_legend.append(s + '_hat')
    plt.ylabel('volume %')
    plt.legend(structure_legend,bbox_to_anchor=(1.1, 1.05),prop={'size': 6}) 
    plt.show()

    # get metric difference into dataframe and display
    column_names = ["Organ", "Dmean", "Dmax", "D95", "D98", "D5", "D2"]
    df = pd.DataFrame(columns = column_names)
    for s in structure_list:
        if s not in DVH_all.keys():
            continue
        Dmean, Dmax, D95, D5, D98, D2
        df = df.append({'Organ' : s, 'Dmean' : (Dmean_hat[s]-Dmean[s])/Dmean[s], 'Dmax' : (Dmax_hat[s]-Dmax[s])/Dmax[s], \
                        'D95':(D95_hat[s]-D95[s])/D95[s], 'D98' : (D98_hat[s]-D98[s])/D98[s], 'D5' : (D5_hat[s]-D5[s])/D5[s], \
                        'D2' : (D2_hat[s]-D2[s])/D2[s]}, ignore_index = True) 
    display(df)
    return df



def predict_batch(model_path, test_path):
    ''' predict the dose for a npy matrix, when generate the data, we main need to maintain 
        a patient to batch_i list to excel

    '''
    # setup the test_path
    if test_path[-1] != '/':
        test_path = test_path + '/'
    data_folder = test_path
    data_dirs = listdir(data_folder)
    
    # load model once 
    if model_path[-1] != '/':
        model_path = model_path + '/'
    model =  load_model(model_path + 'final_model.h5' )
    model.load_weights(model_path + 'best_weights.h5')
   # model = None

    # start to go-over all patients in the test_path subfolders

    metrics_all = {}
    for folder in data_dirs:
        if os.path.isdir(data_folder+folder):
            print('work on test patient ', folder)
            test_patient_path = data_folder+folder
            metrics = predict_evaluation(model, test_patient_path)
            metrics_all[folder] = metrics
    
    return metrics_all
            

## test functoin run in jupyter
def predict_unit_test():
    ''' test function run in jupyter
    '''

    test_path = 'Data/npy_dataset/test/'
    model_path = 'Data/Model_UnetDense/'
    test_patient_path = 'Data/npy_dataset/test/patient_2'
    model = None
   # predict_evaluation(model = None, test_patient_path = test_patient_path)
    predict_batch(model_path, test_path)


    



