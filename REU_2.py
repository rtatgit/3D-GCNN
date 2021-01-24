#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 12:41:07 2017

@author: RishabhMTigadoli
"""

from pyntcloud import PyntCloud
from pyntcloud.io import ply as py
import os
#import sys
from numpy import size

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)

def sample_objects(input_folder,out_folder):
    
    phases = {'train':0,'test':1}
#    classes = {'bathtub':0,'bed':1,'chair':2,'desk':3,'dresser':4,'monitor':5,'night_stand':6,'sofa':7,'table':8,'toilet':9}
    classes = {'airplane':0,'bathtub':1,'bed':2,'bench':3,'bookshelf':4,\
               'bottle':5,'bowl':6,'car':7,'chair':8,'cone':9,\
               'cup':10,'curtain':11,'desk':12,'door':13,'dresser':14,\
               'flower_pot':15,'glass_box':16,'guitar':17,'keyboard':18,'lamp':19,\
               'laptop':20,'mantel':21,'monitor':22,'night_stand':23,'person':24,\
               'piano':25,'plant':26,'radio':27,'range_hood':28,'sink':29,\
               'sofa':30,'stairs':31,'stool':32,'table':33,'tent':34,\
               'toilet':35,'tv_stand':36,'vase':37,'wardrobe':38,'xbox':39            
               }
#    classes ={'wardrobe':0,'xbox':1}
    for c in classes:
        print ("Converting the %s Category " % c)
        category_path = input_folder + '/' + c
        out = out_folder +'/'+ c
        if not os.path.exists(out):
            
            os.makedirs(out)
            
        for t in phases:
            phase = t
            mesh_list = category_path + '/' + phase
            output_folder = out +'/'+ phase
            if not os.path.exists(output_folder):
            
                os.makedirs(output_folder)
            
            list_files = os.listdir(mesh_list)
#            print(list_files)
            for i,file in enumerate(list_files):
                 
#                if (file =='.' or file == '..' or \
#                    os.path.isdir(file) or \
#                    os.path.splitext(file)[1] != 'ply'):
#                        continue
                file_1 = mesh_list + '/' + file
                mesh = PyntCloud.from_file(file_1)
                pointcloud = mesh.get_sample("mesh_random_sampling",n=2048,rgb=False,normals=False)
                outfile = output_folder + '/' + file
#                
##                print (outfile)
                py.write_ply(outfile,points = pointcloud,as_text=True)
    
    

