import os
import numpy as np
from os import listdir

# define the plan matrix and training data size
standard_name = [ 'BODY', 'PTV_Ribs', 'PTV_VExP', 'PTV_SpCord', 'PTV_LN', 'PTV_Spleen', 'PTV_Liver', 'Lungs', 'Heart', 'Esophagus', 'GI_Upper', 'Breasts']
PTV_VExP_Bone = ['PTV_Bone_Total', '.PTV2_Bone', 'PTV2_Bone', 'PTV_Bone']
section_size = (27, 37.5, 75)
matrix_size = (16, 64, 128)

# define the training parameter

epochs = 10
batch_size = 4
test_size = 0.1
imput_size = (16, 64, 128, len(standard_name))

# define folder locations
parent_path = 'Data'




