import os
import sys
import numpy as np
#from get_dataset import get_scan, scan_pading, save_seg_imgs
from tensorflow.keras.models import model_from_json, load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def predict(save_result = False, path):
    
    ## Load prediction model
    model =  load_model(path + '/Model_UnetDense/model.h5')
    model.load_weights(path + '/Model_UnetDense/best_weights.h5')

    # 
    patient = 'Colon'
    test_path = 'Data_v4/test/'
    save_path = test_path+patient+'_predict_step1.npy'
    compare_dose(model, test_path, patient,True, save_path)
    # define the organ that we want to compare dose on organ_index
    organ_index = [1,2,3,4,5,6,7,8,9,10]
    organ_name = ['Body', 'PTV_Ribs','PTV_VExP','PTV_SpinalCord','PTV_LN','PTV_Spleen','PTV_Liver','Lungs','Heart','Esophagus','GI_Upper','Breasts','Avoid1']
    compare_DVH(model, test_path, patient, organ_index, organ_name)
    
    # Predict the dose after step2
    if(step2_finish == True):
        save_path_step2=test_path+patient+'step2.npy'
        model =  load_model('Data_v4/Model_UnetDense/step2/step2_model.h5')
        model.load_weights('Data_v4/Model_UnetDense/step2/best_weights.h5')
        step2_feature_file = test_path+patient+'_feature_step2.npy'
        
        compare_dose2(model, test_path, step2_feature_file, patient, True, save_path_step2)
        organ_index = [1,2,3,4,5,6,7,8,9,10]
        organ_name = ['Body', 'PTV_Ribs','PTV_VExP','PTV_SpinalCord','PTV_LN','PTV_Spleen','PTV_Liver','Lungs','Heart','Esophagus','GI_Upper','Breasts','Avoid1']
        compare_DVH2(model, test_path, step2_feature_file,patient, organ_index, organ_name,normalize = True)




