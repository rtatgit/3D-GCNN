#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:48:38 2017

@author: RishabhMTigadoli
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
import tf_util

def placeholder_inputs(batch_size, vx,vy,vz):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, vx,vy,vz))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
#    vx = point_cloud.get_shape()[1].value
#    vy = point_cloud.get_shape()[2].value
#    vz = point_cloud.get_shape()[3].value
    end_points = {}
    input_image = tf.expand_dims(point_cloud, -1)
    
    net = tf_util.conv3d(input_image, 32, [5,5,5],
                         scope='conv1', stride=[2,2,2],
                         bn=True, is_training=is_training,
                         padding='SAME', bn_decay=bn_decay)
    
    net = tf_util.conv3d(net, 32, [3,3,3],
                         scope='conv2', stride=[2,2,2],
                         bn=True, is_training=is_training,
                         padding='SAME', bn_decay=bn_decay)
   
    # Symmetric function: max pooling
    net = tf_util.max_pool3d(net, [2,2,2],
                             padding='VALID', scope='maxpool')
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 10, activation_fn=None, scope='fc2')

    return net, end_points

def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,15,15,15))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
