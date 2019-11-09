import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
import cv2

from imgaug import augmenters as iaa

import skimage
import scipy
from skimage import measure
from skimage.transform import resize

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score

import tensorflow as tf
import keras

import glob
import pickle
import tqdm
from PIL import Image

from matplotlib import pyplot as plt

IMAGE_SIZE = 512

# Codes to load CVsets
CVdict = pickle.load(open("CVdict.p", "rb"))
trainSet = CVdict['CV1']+CVdict['CV2']+CVdict['CV3']+CVdict['CV4']+CVdict['CV5']
validSet = CVdict['test']

print(len(trainSet))
print(len(validSet))
#print(len(testSet))

# retrieve PTX labels
ptxDict = pickle.load(open("ptxDict.p", "rb"))

ptxLabels = {}

for eachID in list(ptxDict.keys()):
    hasPTX = False
    if(len(ptxDict[eachID])>0):
        hasPTX = True
    ptxLabels[eachID] = hasPTX

metaDataDict  = pickle.load(open("metaDataDict.p", "rb"))

# # Codes for building the generator
import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, RandomSizedCrop,
    RandomBrightnessContrast, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion
)
 
AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    #optionas below emulate different positioning and patient geometry
    ShiftScaleRotate(rotate_limit=(-25, 25)),
    #options below emulate different X-ray penetration
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
        Blur(blur_limit=2),
         ], p=0.3),
    #ToFloat(max_value=1)
],p=1)
 
AUGMENTATIONS_TEST = Compose([
    #ToFloat(max_value=1)
],p=1)

class generator(keras.utils.Sequence):
    
    def __init__(self, idList, metaDataDict, ptxDict, batch_size=8, image_size=512, shuffle=True, augment=False, predict=False):
        self.idList = idList
        self.metaDataDict = metaDataDict
        self.ptxDict = ptxDict
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, thisID):
        # load NPY file
        img = np.array(Image.open('512train/'+thisID+'.png'))
        # load auxillary information
        aux_info = metaDataDict[thisID]
        # if image contains pneumonia
        hasPTX = False
        if(len(ptxDict[thisID])>0):
            hasPTX = True
        # Convert bool to categorical. e.g. True -> [0,1]
        hasPTX = keras.utils.to_categorical(hasPTX, num_classes=2)
        if self.augment:
            img = AUGMENTATIONS_TRAIN(image=img)['image']
        # expand to 3 dim
        if len(img.shape)==2:
            img = np.repeat(img[...,None],3,2)
        return img, aux_info, hasPTX
    
    def __loadpredict__(self, thisID):
        # load NPY file
        img = np.array(Image.open('512train/'+thisID+'.png'))
        # load auxillary information
        aux_info = metaDataDict[thisID]
        # expand to 3 dim
        if len(img.shape)==2:
            img = np.repeat(img[...,None],3,2)
        return img, aux_info
        
    def __getitem__(self, index):
        # select batch
        batchIDlist = self.idList[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            items = [self.__loadpredict__(eachID) for eachID in batchIDlist]
            # zip images and masks
            imgs, aux_info = zip(*items)
             # create numpy batch
            imgs = np.array(imgs)
            aux_info = np.array(aux_info)
            return [imgs, aux_info]
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(eachID) for eachID in batchIDlist]
            # zip images and masks
            imgs, aux_info, hasPTXs = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            aux_info = np.array(aux_info)
            hasPTXs = np.array(hasPTXs)
            return [imgs, aux_info], hasPTXs
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.idList)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.idList) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.idList) / self.batch_size)

# Model Architecture

# Code to clear GPU cache if needed i.e. run this before creating new model
keras.backend.clear_session()

def myMobileNet():
    #imageInput = keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3,), dtype='float32', name='imageInput')
    #output = keras.layers.normalization.BatchNormalization()(imageInput)
    pretrained_model = keras.applications.densenet.DenseNet121(input_shape=(512,512,3), weights='imagenet', include_top = False)
    output = pretrained_model.output
    output = keras.layers.core.Flatten()(output)

    auxInput = keras.layers.Input(shape=(3,), dtype='float32', name='auxInput')
    #output = keras.layers.Concatenate(axis=-1)([output, auxInput])

    output = keras.layers.normalization.BatchNormalization()(output)
    output = keras.layers.core.Dropout(0.3)(output)
    output = keras.layers.core.Dense(32, activation='relu')(output)
    output = keras.layers.normalization.BatchNormalization()(output)
    output = keras.layers.core.Dropout(0.3)(output)
    output = keras.layers.core.Dense(16, activation='relu')(output)
    output = keras.layers.normalization.BatchNormalization()(output)
    output = keras.layers.core.Dropout(0.3)(output)
    output = keras.layers.core.Dense(2, activation='softmax')(output)
    model = keras.models.Model(inputs=pretrained_model.inputs+ [auxInput], outputs=[output])
    #model = keras.utils.multi_gpu_model(model, gpus=2)
    
    return model

model = myMobileNet()
model.summary()

myOptimizer = keras.optimizers.Adam(lr=3e-4)

model.compile(optimizer=myOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Useful callbacks if needed
save_model_callback = keras.callbacks.ModelCheckpoint('models-PTXdetect03-DenseNet/'+'190730_DenseNetDetect_weightsOnly_{epoch:02d}_{val_acc:.2f}.h5', monitor='val_acc',
                                                      verbose=2, save_best_only=True, save_weights_only=True, 
                                                      mode='auto', period=1)
                                                      
train_gen = generator(trainSet, metaDataDict, ptxDict, batch_size=8, 
                      image_size=512, shuffle=True, augment=True, predict=False)

valid_gen = generator(validSet, metaDataDict, ptxDict, batch_size=8, 
                      image_size=512, shuffle=False, augment=False, predict=False)

history = model.fit_generator(train_gen, validation_data=valid_gen, class_weight='auto', callbacks=[save_model_callback], epochs=100, verbose=2, shuffle=True)