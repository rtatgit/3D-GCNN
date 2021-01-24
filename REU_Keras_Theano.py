#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:15:51 2017

@author: RishabhMTigadoli
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import plot_model
plot_model(model, to_file='model.png')
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')

data,label = shuffle(train_data,train_label, random_state=2)    #Shuffle The Data

batch_size = 32
# number of output classes
nb_classes = 10
# number of epochs to train
nb_epoch = 10


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=4)
X_train = X_train.reshape(X_train.shape[0],1, 15,15,15)
X_test = X_test.reshape(X_test.shape[0], 1,15,15,15)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

model = Sequential()

model.add(Convolution3D(nb_filters, nb_conv, nb_conv,nb_conv,
                        border_mode='valid',                #First Conv Layer 
                        input_shape=( 1,15, 15, 15)))

convout1 = Activation('relu') #Activation Layer ReLU
model.add(convout1)

model.add(Convolution3D(nb_filters, nb_conv, nb_conv,nb_conv))#Second Conv Layer

convout2 = Activation('relu') #Activation Layer ReLU
model.add(convout2)

model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))#Max Pooling Layer(Subsampling)

model.add(Dropout(0.5)) #Dropout Layer

model.add(Flatten()) #Flattening the image to single matrix.Acts as input to Dense Layer

model.add(Dense(128))#Dense Layer 
model.add(Activation('relu'))#Activation  Unit RelU

model.add(Dropout(0.5))#Dropout Layer

model.add(Dense(nb_classes))#Output Layer Classes
model.add(Activation('softmax'))#Softmax Classifier Activation Function For Output Layer

#Compile The Model with different loss and optimisation methods
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1, validation_split=0.2)
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])#Score of the Model
print('Test accuracy:', score[1])#Accuracy Of the Model(Test Accuracy)

fname = "REU_adam_10_weights.hdf5"
model.save_weights(fname,overwrite=True)
model.load_weights(fname)
test = np.load('test_data.npy')
test = test.reshape(test.shape[0], 1,15,15,15)
test_l=np.load('test_label.npy')
test_l = np_utils.to_categorical(test_l, nb_classes)

from sklearn.metrics import classification_report,confusion_matrix

#Y_pred = model.predict(X_test)
#print(Y_pred)
#y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)
#  
#                       (or)

y_pred = model.predict_classes(test)
#y_pred_cross = model.predict_classes(X_test)

print(y_pred)

p=model.predict_proba(test) # to predict probability

target_names = ['class 0(Bathtub)', 'class 1(Bed)', 'class 2(Chair)','class 3(Desk)',\
                'class 4(Dresser)','class 5(Monitor)','class 6(Night_Stand)',\
                'class 7(Sofa)','class 8(Table)','class 9(Toilet)']

#target_names = ['class 0(airplane)','class 1(bathtub)','class 2(bed)','class 3(bench)','class 4(bookshelf)',\
#               'class 5(bottle)','class 6(bowl)','class 7(car)','class 8(chair)','class 9(cone)',\
#               'class 10(cup)','class 11(curtain)','class 12(desk)','class 13(door)','class 14(dresser)',\
#               'class 15(flower_pot)','class 16(glass_box)','class 17(guitar)','class 18(keyboard)','class 19(lamp)',\
#               'class 20(laptop)','class 21(mantel)','class 22(monitor)','class 23(night_stand)','class 24(person)',\
#               'class 25(piano)','class 26(plant)','class 27(radio)','class 28(range_hood)','class 29(sink)',\
#               'class 30(sofa)','class 31(stairs)','class 32(stool)','class 33(table)','class 34(tent)',\
#               'class 35(toilet)','class 36(tv_stand)','class 37(vase)','class 38(wardrobe)','class 39(xbox)']

print(classification_report(np.argmax(test_l,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(test_l,axis=1), y_pred))
conf_m = confusion_matrix(np.argmax(test_l,axis=1), y_pred)