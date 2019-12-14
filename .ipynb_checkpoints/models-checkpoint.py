import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os,sys
from PIL import Image
import random
import albumentations as albu
import cv2
import helper
import pandas as pd 
import keras
import h5py
from keras import datasets, layers, models
from keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

class lRelu(Activation):
    
    def __init__(self, activation, **kwargs):
        super(lRelu, self).__init__(activation, **kwargs)
        self.__name__ = 'lrelu'

def lrelu(x):
    return tf.keras.activations.relu(x, alpha=0.1)

get_custom_objects().update({'lrelu': lRelu(lrelu)})

"""Model 1 - baseline"""
def model1(name, inputs = (16,16,3)): 
    model = models.Sequential()
    
    if name == 'SoftmaxBinary': 
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='softmax'))
    
    elif name == 'SigmoidBinary':
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
    elif name == 'SoftmaxCategorical':
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = 'same'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(2, activation='softmax'))
        
    elif name == 'SigmoidCategorical':
                  
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
    
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    
        model.add(layers.Flatten())
        
        model.add(layers.Dense(2, activation='sigmoid'))
        
    return model


"""Model 2 - leaky relu"""
def model2(name, inputs = (16,16,3)): 
    #lrelu = layers.Lambda(lambda x: tf.keras.activations.relu(x, alpha=0.1))
    model = models.Sequential()
    
    if name == 'SoftmaxBinary': 
        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='softmax'))
    
    elif name == 'SoftmaxCategorical':
        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(2, activation='softmax'))
        
    elif name == 'SigmoidBinary': 

        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
    elif name == 'SigmoidCategorical':

        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding ='same' ,input_shape=inputs))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same'))

        model.add(layers.Flatten())
                  
        model.add(layers.Dense(2, activation='sigmoid'))
    return model

"""Model 3 - """
def model3(name, sha = (16,16,3)): 
    
    model_input = keras.Input(shape=sha)

    if name == 'SoftmaxBinary':
                              
        x = layers.Conv2D(64,(7,7), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)


        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
                              
        model_output = layers.Dense(1, activation='softmax')(x)
        
        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 3, SoftmaxBinary')
    elif name == 'SoftmaxCategorical':
        x = layers.Conv2D(64,(7,7), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)


        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
        model_output = layers.Dense(2, activation='softmax')(x)
        
        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 3, SoftmaxCategorical')
        
    elif name == 'SigmoidBinary':
        x = layers.Conv2D(64,(7,7), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)


        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
        model_output = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 3, SigmoidBinary')
        
    elif name == 'SigmoidCategorical':
        x = layers.Conv2D(64,(7,7), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)


        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
        model_output = layers.Dense(2, activation='sigmoid')(x)
        
        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 3, SigmoidCategorical')
    
    return model

"""Model 4 - same as 3 but different windows size"""
def model4(name, sha = (16,16,3)): 


    model_input = keras.Input(shape=sha)
    
    if name == 'SoftmaxBinary':
       
        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(64,(3,3), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)

        x= layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
        x = layers.Dropout(0.5)(x)

        model_output = layers.Dense(2, activation='softmax')(x)
        
        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 4, SoftmaxBinary')
    elif name == 'SoftmaxCategorical':
        
        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)
        
        x = layers.Conv2D(64,(3,3), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)

        x= layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
        x = layers.Dropout(0.5)(x)

        model_output = layers.Dense(2, activation='softmax')(x)

        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 4, SoftmaxCategorical')
    
    elif name == 'SigmoidBinary':
        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)
        
        x = layers.Conv2D(64,(3,3), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)
        
        x= layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
        x = layers.Dropout(0.5)(x)
        
        model_output = layers.Dense(2, activation='sigmoid')(x)
        
        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 4, SigmoidBinary')
    elif name == 'SigmoidCategorical':
        
        x = layers.Conv2D(64,(5,5), activation = lrelu, padding = 'same')(model_input)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(64,(3,3), activation = lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2,2))(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation=lrelu, padding = 'same')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=lrelu)(x)
        x = layers.Dropout(0.5)(x)

        model_output = layers.Dense(2, activation='sigmoid')(x)
  
        model = keras.Model(inputs=model_input, outputs=model_output, name='Model 4, SigmoidCategorical')
    
    return model

def generate_minibatch(X, Y, batch_size, window_size):
    """
    Procedure for real-time minibatch creation and image augmentation.
     This runs in a parallel thread while the model is being trained.
    """
    Y = tf.keras.utils.to_categorical(Y, 2)
    while 1:
        # Generate one minibatch
        X_batch = np.empty((batch_size, window_size,window_size, 3))
        Y_batch = np.empty((batch_size, 2))
        for i in range(batch_size):
            # Select a random image
            idx = np.random.choice(X.shape[0])
            shape = X[idx].shape
            # The label does not depend on the image rotation/flip (provided that the rotation is in steps of 90°)
            X_batch[i] = X[idx]
            Y_batch[i] = Y[idx]
        yield (X_batch, Y_batch)