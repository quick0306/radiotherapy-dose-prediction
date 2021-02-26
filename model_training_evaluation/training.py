import os
import numpy as np
import datetime
from os import listdir
import tensorflow as tf
import random
#from get_dataset import read_npy_dataset, split_npy_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from model_training_evaluation.get_models import unet_dense, get_Discriminator, get_GAN, get_Generator, save_model
from util import mse, dice_coeff 
from config import *


# define training hyperparameters
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
epochs = 1000

def data_gen(splitted_npy_dataset_path, batch_size):
    c = 0
    batch_dirs = listdir(splitted_npy_dataset_path)
    n = os.listdir(splitted_npy_dataset_path) #List of training images
    random.shuffle(n)
    while True:
        X_batch = np.zeros((batch_size, input_size[0], input_size[1], input_size[2], input_size[3])).astype('float')
        Y_batch = np.zeros((batch_size, input_size[0], input_size[1], input_size[2], 1)).astype('float')
        for i in range(c,c+batch_size):
            
            batch_XY = np.load(splitted_npy_dataset_path+'/'+n[i])
            #print('batch_XY size', batch_XY.shape)
            dimention = len(standard_name)
            X_batch[i-c] =  batch_XY[:,:,:,:,0:dimention]
            Y_batch[i-c] =  batch_XY[:,:,:,:,dimention:dimention+1]    
        c = c + batch_size
        if(c+batch_size>=len(os.listdir(splitted_npy_dataset_path))):
            c=0
            random.shuffle(n)
        yield X_batch, Y_batch

def train_nn_model(model, training_npy_path, validation_npy_path, epochs, save_path):
   # test_XY = np.load(test_path+'/test.npy')
   # X_test, Y_test = test_XY[:,:,:,:,0:2],test_XY[:,:,:,:,2:3]
    
    batch_dirs = listdir(training_npy_path)
    len_batch_dirs = len(batch_dirs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save the best weights
    best_weights_path = save_path+'best_weights.h5'
    checkpoints = []
    checkpoints.append(EarlyStopping(monitor='val_loss', patience=500))
    checkpoints.append(ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False))
    model.fit_generator(data_gen(training_npy_path, batch_size), steps_per_epoch=int(len_batch_dirs/batch_size)+1, epochs=epochs, validation_data=data_gen(validation_npy_path, batch_size = batch_size), 
                          validation_steps=5, callbacks=checkpoints)
    #scores = model.evaluate(X_test, Y_test)        
    #print(Y_predict.shape)
    #dice_score = dice_coefficient(model.predict(X_test), Y_test)
        
  #  print('Test loss:', scores[0], '\nTest accuracy:', scores[1], '\nDice Coefficient Accuracy:', dice_score)
    return model

def run_training(train_gan_model = False, input_size = (16, 96, 128, 12), parent_path='Data', training_npy_path = 'Data/npy_dataset/training/' , validation_npy_path = 'Data/npy_dataset/validation/'):
    
    if train_gan_model:
        Generator = unet_dense(input_size)
        Encoder = unet_dense(input_size)
        Discriminator = get_Discriminator(input_shape_1=input_size, input_shape_2=(16, 64, 128, 1))
        GAN = get_GAN(input_size, Generator, Discriminator)

        # Saving non-trained models:
        save_model(Generator, path = parent_path + '/GAN-Models/Generator/', model_name = 'model', weights_name = 'weights')
        save_model(Encoder, path = parent_path + '/GAN-Models/Encoder/', model_name = 'model', weights_name = 'weights')
        save_model(Discriminator, path= parent_path +'/GAN-Models/Discriminator/', model_name = 'model', weights_name = 'weights')
        print('Non-Trained model saved to "Data/GAN-Models"!')
        ## Train
        Generator, Encoder, Discriminator = train_gan(Generator, Encoder, Discriminator, GAN, splitted_npy_dataset_path='Data_v4/npy_dataset/training_npy_dataset', test_path = 'Data_v4/npy_dataset/validation_npy', epochs = epochs)

        # Saving trained models:
        save_model(Generator, path=parent_path + 'GAN-Models/Generator/', model_name = 'model', weights_name = 'weights')
        save_model(Encoder, path=parent_path + 'GAN-Models/Encoder/', model_name = 'model', weights_name = 'weights')
        save_model(Discriminator, path= parent_path +'/GAN-Models/Discriminator/', model_name = 'model', weights_name = 'weights')
        print('Trained model saved to', parent_path +'/GAN-Models')
        return Generator
        
    else:
        save_path_checkpoints = parent_path + '/Checkpoints_UnetDense/'
        model_save_path = parent_path + '/Model_UnetDense'
        pred_model = unet_dense(input_size = input_size)
        save_model(pred_model, path = model_save_path, model_name = 'unet_model', weights_name = 'weights')
        print('Non-Trained model saved to',  model_save_path)
        model = train_nn_model(pred_model, training_npy_path = training_npy_path, validation_npy_path = validation_npy_path, epochs = epochs, save_path = save_path_checkpoints)
        save_model(model, path= model_save_path, model_name = final_model, weights_name = 'weights')
        print('Trained model saved to', model_save_path)
        return model


def training_unit_test():
    run_training(train_gan_model = False, input_size = input_size, parent_path= parent_path, training_npy_path =  training_npy_path , validation_npy_path = validation_npy_path)


if __name__ == '__main__':
    run_training(train_gan_model = False, input_size = input_size, parent_path= parent_path, training_npy_path =  training_npy_path , validation_npy_path = validation_npy_path)