#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 19:46:57 2017

@author: RishabhMTigadoli
"""
import os
import sys
import numpy as np
from numpy import size,array
#import tensorflow as tf
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import data_prep_util as dat
#import tf_util 
#phases = {'train':0,'test':1}
def get_train():
    
    classes = {'bathtub':0,'bed':1,'chair':2,'desk':3,'dresser':4,'monitor':5,\
               'night_stand':6,'sofa':7,'table':8,'toilet':9}
    label = []
    train_data = []
    train_label= []
    train_data = np.zeros([3991,2048,3],dtype='float32')
    train_label = np.array(train_label,dtype = int)
    folder = '/Users/RishabhMTigadoli/Desktop/REU 3D Deep Learning/Data-Sets/Model_samp' 
    k = 0
    for clas,label in sorted(classes.iteritems()):
        print ("Vectorising the %s Category " % clas)
        category_path = folder + '/' + clas
        
        phase = 'train'
        cloud_list = category_path + '/' + phase
    
        list_files = os.listdir(cloud_list)
    #            print(list_files)
        labels=np.ones(len(list_files,),dtype = int)
        labels[0:len(list_files)] = label
        labels = array(labels)
        train_label= np.append(train_label,labels)
        for i,file_n in enumerate(list_files):
            
            file_1 = cloud_list + '/' + file_n
            point_data = dat.load_ply_data(file_1,2048)
            
    #        point_data = tf.convert_to_tensor(point_data, dtype=tf.float32)
            for j in range(point_data.shape[0]):
                train_data[k,j,:] = (point_data[j][0],point_data[j][1],point_data[j][2])
                
    #        train_data = train_data.reshape(point_data.shape[0],3)
            k+=1
    #    training_data = [array(train_data,dtype = 'str')\
    #                     ,array(train_label,dtype='int32')] 
    np.save('train_pc.npy',train_data)
    np.save('train_pc_label',train_label)
    return train_data,train_label

def get_test():
    
        classes = {'bathtub':0,'bed':1,'chair':2,'desk':3,'dresser':4,'monitor':5,\
               'night_stand':6,'sofa':7,'table':8,'toilet':9}
        label = []
        test_data = []
        test_label= []
        test_data = np.zeros([908,2048,3],dtype='float32')
        test_label = np.array(test_label,dtype = int)
        folder = '/Users/RishabhMTigadoli/Desktop/REU 3D Deep Learning/Data-Sets/Model_samp' 

        k = 0
        for clas,label in sorted(classes.iteritems()):
            print ("Vectorising the %s Category " % clas)
            category_path = folder + '/' + clas
            
            phase = 'test'
            cloud_list = category_path + '/' + phase
        
            list_files = os.listdir(cloud_list)
        #            print(list_files)
            labels=np.ones(len(list_files,),dtype = int)
            labels[0:len(list_files)] = label
            labels = array(labels)
            test_label= np.append(test_label,labels)
            for i,file_n in enumerate(list_files):
                
                file_1 = cloud_list + '/' + file_n
                point_data = dat.load_ply_data(file_1,2048)
                
        #        point_data = tf.convert_to_tensor(point_data, dtype=tf.float32)
                for j in range(point_data.shape[0]):
                    test_data[k,j,:] = (point_data[j][0],point_data[j][1],point_data[j][2])
                    
        #        train_data = train_data.reshape(point_data.shape[0],3)
                k+=1
        #    training_data = [array(train_data,dtype = 'str')\
        #                     ,array(train_label,dtype='int32')] 
        np.save('test_pc.npy',test_data)
        np.save('test_pc_label',test_label)

        return test_data,test_label
        
 