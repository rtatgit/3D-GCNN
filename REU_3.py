#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:21:35 2017

@author: RishabhMTigadoli
"""
import numpy as np
import REU_Voxelise_fix
#import REU_1
import REU_4 as model
import tf_util
import tensorflow as tf
import os
import sys
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.001
DECAY_STEP = 200000
DECAY_RATE = 0.8
MOMENTUM = 0.9
OPTIMIZER = 'momentum'
MAX_EPOCH = 100
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


#training_data,training_label = REU_1.get_train()    
#test_data,test_label = REU_1.get_test()
training_data,training_label = REU_Voxelise_fix.get_train()    
test_data,test_label = REU_Voxelise_fix.get_test()

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    
    file_size = training_data.shape[0]
    num_batches = file_size // BATCH_SIZE
        
    total_correct = 0
    total_seen = 0
    loss_sum = 0
       
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        # Augment batched point clouds by rotation and jittering
        feed_dict = {ops['pointclouds_pl']: training_data[0:BATCH_SIZE,:,:,:],
                     ops['labels_pl']: training_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        step, _, loss_val, pred_val = sess.run([ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == training_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        
    print('mean loss: %f' % (loss_sum / float(num_batches)))
    print('accuracy: %f' % (total_correct / float(total_seen)))

def test_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    NUM_CLASSES = 10
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
            
    file_size = test_data.shape[0]
    num_batches = file_size // BATCH_SIZE
        
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: test_data[start_idx:end_idx,:,:,:],
                     ops['labels_pl']: test_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        step, loss_val, pred_val = sess.run([ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == test_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = test_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)
        
    print('eval mean loss: %f' % (loss_sum / float(total_seen)))
    print('eval accuracy: %f'% (total_correct / float(total_seen)))
    print('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            pointclouds_pl, labels_pl = model.placeholder_inputs(BATCH_SIZE, 15,15,15)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            print 'bn_decay:',bn_decay

            # Get model and loss 
            pred, end_points = model.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = model.get_loss(pred, labels_pl, end_points)
            print 'loss:',loss

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            print 'accuracy:',accuracy

            # Get training operator
            learning_rate = get_learning_rate(batch)
            print 'learning_rate:',learning_rate
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
#             Add ops to save and restore all the variables.
#            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
#        config.gpu_options.allow_growth = True
#        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
#        merged = tf.summary.merge_all()
#        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
#                                  sess.graph)
#        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
#               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            print('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops)
            test_one_epoch(sess, ops)
            
            # Save the variables to disk.
#            if epoch % 10 == 0:
#                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
#                log_string("Model saved in file: %s" % save_path)
if __name__ == "__main__":
    train()
    