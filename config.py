import os
import numpy as np
from os import listdir

# define the plan matrix and training data size
standard_name = [ 'BODY', 'PTV_Ribs', 'PTV_VExP', 'PTV_SpCord', 'PTV_LN', 'PTV_Spleen', 'PTV_Liver', 'Lungs', 'Heart', 'Esophagus', 'GI_Upper', 'Breasts']
PTV_VExP_Bone = ['.PTV2_Bone', 'PTV2_Bone', 'PTV_Bone', '.PTV_Bone']
section_size = (27, 37.5, 50)
matrix_size = (32, 120, 160)
section = 'Lungs'
# define the training parameter

epochs = 1000
batch_size = 5
test_size = 0.1
input_size = (matrix_size[0], matrix_size[1], matrix_size[2], len(standard_name))

# define folder locations
parent_path = 'Data'
training_npy_path = 'Data/npy_dataset/training/'
validation_npy_path = 'Data/npy_dataset/validation/'
plan_save_path = 'Data/plans'
dataset_save_path = 'Data/npy_dataset'

# define the model name
final_model = 'final_model.h5'
best_weight = 'best_weights.h5'




