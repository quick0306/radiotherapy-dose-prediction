# Dongsu DU， City of hope cancer center
# Define the network structure for several models
import os
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adadelta, Adam
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input, Conv3D, Dense, UpSampling3D, Activation, MaxPooling3D, Dropout, concatenate, Flatten, Multiply, Subtract, Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization

def save_model(model, path='Data/Model/', model_name = 'model', weights_name = 'weights'):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path+model_name+'.h5',overwrite= True,include_optimizer = True)
    return

def get_pred_model(model_path, weights_path):
    if not os.path.exists(model_path):
        print('Model file not exists!')
        return None
    elif not os.path.exists(weights_path):
        print('Weights file not exists!')
        return None

    # Getting model:
    with open(model_path, 'r') as model_file:
        model = model_file.read()
    model = model_from_json(model)
    # Getting weights
    model.load_weights(weights_path)
    return model

# Loss Function:
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# prediction model 1

def unet(pretrained_weights = None,input_size = (16, 64, 128, 12)):
    ''' unet structure with different dimension metrix 
    '''
    
    inputs = Input(input_size)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 4)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2, 2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv3D(1, 1, activation = 'linear')(conv9)

    print('output shape=',conv10.shape)
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics=['mean_squared_error', 'acc'])
    
    #model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    print('Dose prediction Model Architecture:')
    print(model.summary())
    
    return model


# model of unet_v2


def unet_v2(pretrained_weights = None,input_size = (128,256,10)):
    
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  #  conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  #  conv1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.25)(conv1)     # drop for 1
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1) # work on drop 
   # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  #  conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  #  conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.25)(conv2)     # drop for 2
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2) # work on drop 
   # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
   # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
   # conv3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.25)(conv3)     # drop for 3
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # work on drop 
   # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
   # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
   # conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.25)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
   # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
   # conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.25)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
   # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
   # conv6 = BatchNormalization()(conv6)
    drop6 = Dropout(0.25)(conv6)    # add drop 6

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
   # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
   # conv7 = BatchNormalization()(conv7)
    drop7 = Dropout(0.25)(conv7)    # add drop 7

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
   # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
   # conv8 = BatchNormalization()(conv8)
    drop8 = Dropout(0.25)(conv8)    # add drop 8

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
   # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
   # conv9 = BatchNormalization()(conv9)
    drop9 = Dropout(0.25)(conv9)    # add drop 9
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop9)
   # conv9 = BatchNormalization()(conv9)
    
    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)
    print('output shape=',conv10.shape)
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics=['mean_squared_error', 'acc'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    print('Dose prediction Model Architecture:')
    print(model.summary())
    
    return model

############################# model 3 ###############################################################

def unet_v3(pretrained_weights = None,input_size = (128,256,13)):
    
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)

    print('output shape=',conv10.shape)
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4, beta_1=0.9, beta_2=0.999), loss = 'mean_squared_error', metrics=['mean_squared_error', 'acc'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    print('Dose prediction Model Architecture:')
    print(model.summary())
    
    return model


########################################### model 4#######################################################

def unet_dense(pretrained_weights = None,input_size = (16, 64, 128, 12)):
    
    inputs = Input(input_size)
    conv1_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    input_1 = concatenate([inputs,conv1_1], axis = 4)    
    conv1_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_1)
    conv1 = concatenate([input_1,conv1_2], axis = 4)
    
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    input_2 = concatenate([pool1,conv2_1], axis = 4)    
    conv2_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_2)
    conv2 = concatenate([input_2,conv2_2], axis = 4)
    
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    input_3 = concatenate([pool2,conv3_1], axis = 4) 
    conv3_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_3)
    conv3 = concatenate([input_3,conv3_2], axis = 4)
    
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    input_4 = concatenate([pool3,conv4_1], axis = 4) 
    conv4_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_4)
    conv4 = concatenate([input_4,conv4_2], axis = 4)
    drop4 = Dropout(0.5)(conv4)
    
    
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
    conv5_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    input_5 = concatenate([pool4,conv5_1], axis = 4) 
    conv5_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_5)
    conv5 = concatenate([input_5,conv5_2], axis = 4)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 4)
    conv6_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    input_6 = concatenate([merge6,conv6_1], axis = 4)
    conv6_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_6)
    conv6 = concatenate([input_6,conv6_2], axis = 4)

    up7 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    input_7 = concatenate([merge7,conv7_1], axis = 4)
    conv7_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_7)
    conv7 = concatenate([input_7,conv7_2], axis = 4)
    
    up8 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    input_8 = concatenate([merge8,conv8_1], axis = 4)
    conv8_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_8)
    conv8 = concatenate([input_8,conv8_2], axis = 4)

    up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (3,3))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9_1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    input_9 = concatenate([merge9,conv9_1], axis = 4)
    conv9_2 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_9)
    conv9 = concatenate([input_9,conv9_2], axis = 4)
    
    conv9 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv3D(1, 1, activation = 'linear')(conv9)

    print('output shape=',conv10.shape)
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4, beta_1=0.9, beta_2=0.999), loss = 'mean_squared_error', metrics=['mean_squared_error', 'acc'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    print('Dose prediction Model Architecture:')
    print(model.summary())
    
    return model


#################################################################################################################

def get_GAN(input_shape, Generator, Discriminator):
    
    Discriminator.trainable = False
    input_gan = Input(shape=input_shape) # segmentes  [128,256,13]
    fake_dose = Generator(input_gan)
    
    dis_out = Discriminator([input_gan,fake_dose])
    
   # gan_output = Discriminator([input_gan, generated_dose])

    # Compile GAN:
    gan = Model(input_gan, outputs = [dis_out,fake_dose])
    gan.compile(optimizer=Adam(lr = 0.0002,beta_1=0.5), loss=['mse', 'mae'], loss_weights=[1,10], metrics=['mean_squared_error', 'acc'])

    print('GAN Architecture:')
    print(gan.summary())
    return gan


########################################Generator for GAN #########################################################

def unet_Generator(pretrained_weights = None,input_size = (128,256,13)):
    
    inputs = Input(input_size)
    conv1_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    input_1 = concatenate([inputs,conv1_1], axis = 3)    
    conv1_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_1)
    conv1 = concatenate([input_1,conv1_2], axis = 3)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    input_2 = concatenate([pool1,conv2_1], axis = 3)    
    conv2_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_2)
    conv2 = concatenate([input_2,conv2_2], axis = 3)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    input_3 = concatenate([pool2,conv3_1], axis = 3) 
    conv3_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_3)
    conv3 = concatenate([input_3,conv3_2], axis = 3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    input_4 = concatenate([pool3,conv4_1], axis = 3) 
    conv4_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_4)
    conv4 = concatenate([input_4,conv4_2], axis = 3)
    drop4 = Dropout(0.5)(conv4)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    input_5 = concatenate([pool4,conv5_1], axis = 3) 
    conv5_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_5)
    conv5 = concatenate([input_5,conv5_2], axis = 3)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    input_6 = concatenate([merge6,conv6_1], axis = 3)
    conv6_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_6)
    conv6 = concatenate([input_6,conv6_2], axis = 3)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    input_7 = concatenate([merge7,conv7_1], axis = 3)
    conv7_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_7)
    conv7 = concatenate([input_7,conv7_2], axis = 3)
    
    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    input_8 = concatenate([merge8,conv8_1], axis = 3)
    conv8_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_8)
    conv8 = concatenate([input_8,conv8_2], axis = 3)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9_1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    input_9 = concatenate([merge9,conv9_1], axis = 3)
    conv9_2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_9)
    conv9 = concatenate([input_9,conv9_2], axis = 3)
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'linear')(conv9)

    print('output shape=',conv10.shape)
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4, beta_1=0.9, beta_2=0.999), loss = 'mean_squared_error', metrics=['mean_squared_error', 'acc'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    print('Dose prediction Model Architecture:')
    print(model.summary())
    
    return model





##################################################################################################################

def get_Generator(input_shape):
    Generator, _ = unet_dense(input_shape)
    print('Generator Architecture:')
    print(Generator.summary())
    return Generator


#################################################################################################


def get_Discriminator(input_shape_1, input_shape_2):

    dis_inputs_1 = Input(shape=input_shape_1) # segmentes  [128,256,13]
    dis_inputs_2 = Input(shape=input_shape_2) # dose [128,256,1]
    combined_imgs = concatenate([dis_inputs_1, dis_inputs_2], axis=-1)
  #  print('shape of combined imgs', combined_imgs.shape())
   # mul_1 = Multiply()([dis_inputs_1, dis_inputs_2]) # Getting segmented part
   # encoder_output_1 = Encoder(dis_inputs_1)
   # encoder_output_2 = Encoder(mul_1)
   # subtract_dis = Subtract()([encoder_output_1, encoder_output_2])

    d = Conv2D(32, (4,4), strides=(2,2), padding='same')(combined_imgs)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(64, (4,4), strides=(2,2), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(64, (4,4), strides=(2,2), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(128, (4,4), strides=(2,2), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(128, (4,4), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # non-patch output
    
#    d = Conv2D(1, (4,4), padding='same')(d)
    flat_1 = Flatten()(d)
    dis_fc_1 = Dense(128)(flat_1)
    dis_fc_1 = Activation('relu')(dis_fc_1)

  #  dis_drp_1 = Dropout(0.5)(dis_fc_1)

    dis_fc_2 = Dense(56)(dis_fc_1)
    dis_fc_2 = Activation('relu')(dis_fc_2)

  #  dis_drp_2 = Dropout(0.5)(dis_fc_2)

    dis_fc_3 = Dense(1)(dis_fc_2)
    dis_similarity_output = Activation('sigmoid')(dis_fc_3)
    # define model
    Discriminator = Model(inputs=[dis_inputs_1, dis_inputs_2],outputs = dis_similarity_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    Discriminator.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    print('Discriminator Architecture:')
    print(Discriminator.summary())
    
    return Discriminator
   


###############################################################################################



def get_segment_model(data_shape):
    # U-Net:
    inputs = Input(shape=(data_shape))
    print('matrix size =',inputs.shape)
    conv_block_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    conv_block_1 = Activation('relu')(conv_block_1)
 #   conv_block_1 = Dropout(0.2) (conv_block_1)
    conv_block_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(conv_block_1)
    conv_block_1 = Activation('relu')(conv_block_1)
    pool_block_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_1)
    
   # pool_block_1 = Dropout(0.2) (pool_block_1)
    conv_block_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(pool_block_1)
    conv_block_2 = Activation('relu')(conv_block_2)
  #  conv_block_2 = Dropout(0.2) (conv_block_2)
    conv_block_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(conv_block_2)
    conv_block_2 = Activation('relu')(conv_block_2)
    pool_block_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_2)

    conv_block_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pool_block_2)
    conv_block_3 = Activation('relu')(conv_block_3)
   # conv_block_3 = Dropout(0.2) (conv_block_3)
    conv_block_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(conv_block_3)
    conv_block_3 = Activation('relu')(conv_block_3)
    pool_block_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_3)

    conv_block_4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(pool_block_3)
    conv_block_4 = Activation('relu')(conv_block_4)
   # conv_block_4 = Dropout(0.2) (conv_block_4)
    conv_block_4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(conv_block_4)
    conv_block_4 = Activation('relu')(conv_block_4)
    pool_block_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block_4)

    conv_block_5 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(pool_block_4)
    conv_block_5 = Activation('relu')(conv_block_5)
   # conv_block_5 = Dropout(0.2) (conv_block_5)
    conv_block_5 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(conv_block_5)
    conv_block_5 = Activation('relu')(conv_block_5)

    encoder = Model(inputs=inputs, outputs=conv_block_5)

    up_block_1 = UpSampling2D((2, 2))(conv_block_5)
    up_block_1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(up_block_1)

    merge_1 = concatenate([conv_block_4, up_block_1])

    conv_block_6 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(merge_1)
    conv_block_6 = Activation('relu')(conv_block_6)
   # conv_block_6 = Dropout(0.2) (conv_block_6)
    conv_block_6 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(conv_block_6)
    conv_block_6 = Activation('relu')(conv_block_6)

    up_block_2 = UpSampling2D((2, 2))(conv_block_6)
    up_block_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(up_block_2)

    merge_2 = concatenate([conv_block_3, up_block_2])

    conv_block_7 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(merge_2)
    conv_block_7 = Activation('relu')(conv_block_7)
   # conv_block_7 = Dropout(0.2) (conv_block_7)
    conv_block_7 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(conv_block_7)
    conv_block_7 = Activation('relu')(conv_block_7)

    up_block_3 = UpSampling2D((2, 2))(conv_block_7)
    up_block_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(up_block_3)

    merge_3 = concatenate([conv_block_2, up_block_3])

    conv_block_8 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(merge_3)
    conv_block_8 = Activation('relu')(conv_block_8)
   # conv_block_8 = Dropout(0.2) (conv_block_8)
    conv_block_8 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(conv_block_8)
    conv_block_8 = Activation('relu')(conv_block_8)

    up_block_4 = UpSampling2D((2, 2))(conv_block_8)
    up_block_4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(up_block_4)

    merge_4 = concatenate([conv_block_1, up_block_4])

    conv_block_9 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(merge_4)
    conv_block_9 = Activation('relu')(conv_block_9)
   # conv_block_9 = Dropout(0.2) (conv_block_9)
    conv_block_9 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(conv_block_9)
    conv_block_9 = Activation('relu')(conv_block_9)

    conv_block_10 = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(conv_block_9)
    
    
    outputs = Activation('linear')(conv_block_10)
    print('output shape=',outputs.shape)
    model = Model(inputs=inputs, outputs=outputs)

    """
    # For Multi-GPU:

    try:
        model = multi_gpu_model(model)
    except:
        pass
    """

    model.compile(optimizer = Adam(lr=0.0001), loss='mean_squared_error', metrics=['mean_squared_error', 'acc'])

    print('Dose prediction Model Architecture:')
    print(model.summary())

    return model


##################

if __name__ == '__main__':
    segment_model = get_segment_model((128,256,13))
    generator = get_Generator((128,256,1))
    discriminator = get_Discriminator((128,256,1), (128,256,1), encoder)
    gan = get_GAN((128,256,1), generator, discriminator)
