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
#import REU_1
#import pyntcloud.utils.array as ar

#from pyntcloud import PyntCloud
#from pyntcloud.io import ply as py
#from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import data_prep_util as dat

#import sys
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)

def get_train():
    x_y_z = [15,15,15]

#    classes = {'bathtub':0,'bed':1,'chair':2,'desk':3,'dresser':4,'monitor':5,\
#               'night_stand':6,'sofa':7,'table':8,'toilet':9}
    classes = {'airplane':0,'bathtub':1,'bed':2,'bench':3,'bookshelf':4,\
               'bottle':5,'bowl':6,'car':7,'chair':8,'cone':9,\
               'cup':10,'curtain':11,'desk':12,'door':13,'dresser':14,\
               'flower_pot':15,'glass_box':16,'guitar':17,'keyboard':18,'lamp':19,\
               'laptop':20,'mantel':21,'monitor':22,'night_stand':23,'person':24,\
               'piano':25,'plant':26,'radio':27,'range_hood':28,'sink':29,\
               'sofa':30,'stairs':31,'stool':32,'table':33,'tent':34,\
               'toilet':35,'tv_stand':36,'vase':37,'wardrobe':38,'xbox':39            
               }    
    label = []
    train_data = []
    train_label= []
    train_data = np.zeros([9843,2048,3],dtype='float32')
    v1 = np.zeros([9843,15,15,15],dtype = 'float32')
    train_label = np.array(train_label,dtype = int)
    folder = '/Users/RishabhMTigadoli/Desktop/REU 3D Deep Learning/Data-Sets/Model_samp_40' 
    k = 0
    for clas,label in sorted(classes.iteritems()):
        print ("Voxelising the %s Category " % clas)
        category_path = folder + '/' + clas
        
        phase = 'train'
        cloud_list = category_path + '/' + phase
    
        list_files = os.listdir(cloud_list)
    #            print(list_files)
        labels=np.ones(len(list_files,),dtype = int)
        labels[0:len(list_files)] = label
        labels = array(labels)
        train_label= np.append(train_label,labels)
        for no,file_n in enumerate(list_files):
            
            file_1 = cloud_list + '/' + file_n
            point_data = dat.load_ply_data(file_1,2048)
            
    #        point_data = tf.convert_to_tensor(point_data, dtype=tf.float32)
            for j in range(point_data.shape[0]):
                train_data[k,j,:] = (point_data[j][0],point_data[j][1],point_data[j][2])
            xyzmin = train_data[k].min(0)
            xyzmax = train_data[k].max(0)
            margin = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
            xyzmin = xyzmin - margin / 2
            xyzmax = xyzmax + margin / 2
            segments = []
            shape = []

            for i in range(3):
                # note the +1 in num
                s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i]+1),
                                      retstep=True)
                segments.append(s)
                shape.append(step)
 
            voxel_x2 = np.clip(np.searchsorted(segments[0], train_data[k,:, 0]) - 1, 0,
                                           x_y_z[0])
            voxel_y2 = np.clip(np.searchsorted(segments[1], train_data[k,:, 1]) - 1, 0,
                                   x_y_z[1])
            voxel_z2 = np.clip(np.searchsorted(segments[2], train_data[k,:, 2]) - 1, 0,
                                   x_y_z[2])
            voxel_n2 = np.ravel_multi_index([voxel_x2, voxel_y2, voxel_z2],x_y_z)
            
            # compute center of each voxel
            
            n_voxels1 = x_y_z[0] * x_y_z[1] * x_y_z[2]
            
            set(voxel_n2)
            vector1 = np.zeros(n_voxels1)
            count1 = np.bincount(voxel_n2)
            vector1[:len(count1)] = count1
            vector1 /= len(voxel_n2)
            v1[k,:,:,:] = vector1.reshape(x_y_z)
            
            
            
    #        train_data = train_data.reshape(point_data.shape[0],3)
            k+=1
    #    training_data = [array(train_data,dtype = 'str')\
    #                     ,array(train_label,dtype='int32')] 
    np.save('train_data_40',v1)
    np.save('train_label_40',train_label)
    return v1,train_label

def get_test():
    x_y_z = [15,15,15]

#    classes = {'bathtub':0,'bed':1,'chair':2,'desk':3,'dresser':4,'monitor':5,\
#               'night_stand':6,'sofa':7,'table':8,'toilet':9}
    classes = {'airplane':0,'bathtub':1,'bed':2,'bench':3,'bookshelf':4,\
               'bottle':5,'bowl':6,'car':7,'chair':8,'cone':9,\
               'cup':10,'curtain':11,'desk':12,'door':13,'dresser':14,\
               'flower_pot':15,'glass_box':16,'guitar':17,'keyboard':18,'lamp':19,\
               'laptop':20,'mantel':21,'monitor':22,'night_stand':23,'person':24,\
               'piano':25,'plant':26,'radio':27,'range_hood':28,'sink':29,\
               'sofa':30,'stairs':31,'stool':32,'table':33,'tent':34,\
               'toilet':35,'tv_stand':36,'vase':37,'wardrobe':38,'xbox':39            
               }   
    label = []
    test_data = []
    test_label= []
    test_data = np.zeros([2468,2048,3],dtype='float32')
    v_test = np.zeros([2468,15,15,15],dtype = 'float32')
    test_label = np.array(test_label,dtype = int)
    folder = '/Users/RishabhMTigadoli/Desktop/REU 3D Deep Learning/Data-Sets/Model_samp_40' 
    k = 0
    for clas,label in sorted(classes.iteritems()):
        print ("Voxelising the %s Category " % clas)
        category_path = folder + '/' + clas
        
        phase = 'test'
        cloud_list = category_path + '/' + phase
    
        list_files = os.listdir(cloud_list)
    #            print(list_files)
        labels=np.ones(len(list_files,),dtype = int)
        labels[0:len(list_files)] = label
        labels = array(labels)
        test_label= np.append(test_label,labels)
        for no,file_n in enumerate(list_files):
            
            file_1 = cloud_list + '/' + file_n
            point_data = dat.load_ply_data(file_1,2048)
            
    #        point_data = tf.convert_to_tensor(point_data, dtype=tf.float32)
            for j in range(point_data.shape[0]):
                test_data[k,j,:] = (point_data[j][0],point_data[j][1],point_data[j][2])
            xyzmin = test_data[k].min(0)
            xyzmax = test_data[k].max(0)
            margin = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
            xyzmin = xyzmin - margin / 2
            xyzmax = xyzmax + margin / 2
            segments = []
            shape = []

            for i in range(3):
                # note the +1 in num
                s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i]+1),
                                      retstep=True)
                segments.append(s)
                shape.append(step)
 
            voxel_x2 = np.clip(np.searchsorted(segments[0], test_data[k,:, 0]) - 1, 0,
                                           x_y_z[0])
            voxel_y2 = np.clip(np.searchsorted(segments[1], test_data[k,:, 1]) - 1, 0,
                                   x_y_z[1])
            voxel_z2 = np.clip(np.searchsorted(segments[2], test_data[k,:, 2]) - 1, 0,
                                   x_y_z[2])
            voxel_n2 = np.ravel_multi_index([voxel_x2, voxel_y2, voxel_z2],x_y_z)
            
            # compute center of each voxel
            
            n_voxels1 = x_y_z[0] * x_y_z[1] * x_y_z[2]
            
            set(voxel_n2)
            vector1 = np.zeros(n_voxels1)
            count1 = np.bincount(voxel_n2)
            vector1[:len(count1)] = count1
            vector1 /= len(voxel_n2)
            v_test[k,:,:,:] = vector1.reshape(x_y_z)
            
            
            
    #        train_data = train_data.reshape(point_data.shape[0],3)
            k+=1
    #    training_data = [array(train_data,dtype = 'str')\
    #                     ,array(train_label,dtype='int32')] 
    np.save('test_data_40.npy',v_test)
    np.save('test_label_40.npy',test_label)

    return v_test,test_label
