# radiotherapy-dose-prediction
In radiotherapy treatment planning, the planner usually need to try different optimization parameters to achieve the best plan with optimal dose distribution. This is an trial and error process and the planner not always find the best dose ditribution.

In this project, we developed an deep-learning model for optimal dose prediction. Using this model, the planner can use only **CT** and **organ structure** to predict the best plan dose distribution . 


# Requirements
- [dicom-contour](https://github.com/dudongsu/dicom-contour) (modified version by Dongsu Du)
- [TensorFlow](https://tensorflow.org/) (An open source deep learning platform) 
- [pydicom](https://github.com/pydicom/pydicom) (An open source dicom package)

# Table Of Contents
-  [Training data](#Training-Data)
-  [Data preprocessing](#Data-preprocessing)
-  [Models](#Models)
-  [Programs](#Programs)
-  [Future Work](#futurework)

# Training-Data  
We have obtained 30 Total marrow irradiation plan data as training data. Each plan has been optimized to the best distribution with same evaluation criteria, and it includes:
- CT images (dicom files)
- Structures file (dicom file)
- Dose file (dicom files) 


# Data-preprocessing
The CT images, structures and dose need to be registered together to generate masks. 

In `preprocesing/get_plan_from_dicom.py`. 
```python
class Plan(Object):
    def get_plan_mask():
        # resampled all CT, structures, dose to same grid
    def rename():
        # rename all structures to standard
    def resample():
        # resample the matrix
    def structure_range():
        # return the structure range in matrix
    def img_cut():
        # cut the whole images
```


# Models

We have implemented 3 models 
- Unet 
- Attention gated UNet
- GAN 

![alt text](https://github.com/dudongsu/radiotherapy-dose-prediction/blob/master/Attention_UNet.png?raw=true)

# Programs

```
├──  config
│    └──  config parameters  
├──  data  
│    └── datasets, please email dudongsu@gmail.com to request the data
    
├──  preprocessing
│   ├── generate_data.py     - this file contains the program to generate training and testing data

│   ├── get_plan_from_dicom.py   - this file contains the Plan class
│
├── model_trainig_evaluation             - this folder contains trainig models and evaluation program.
│   └── get_models.py              - includes all neuro network models
│
├── test.ipynb            - this script contains the whole trainig and tesing process
```


# Future Work

I will developed a new network strucutre based on transformer