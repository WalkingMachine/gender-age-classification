from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf
#from data import distorted_inputs
import re
from tensorflow.contrib.layers import *

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base


TOWER_NAME = 'tower'


def get_checkpoint(checkpoint_path, requested_step=None, basename='checkpoint'):
    if requested_step is not None:

        model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, basename, requested_step)
        if os.path.exists(model_checkpoint_path) is None:
            print('No checkpoint file found at [%s]' % checkpoint_path)
            exit(-1)
            print(model_checkpoint_path)
        print(model_checkpoint_path)
        return model_checkpoint_path, requested_step

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inception_v3(nlabels, images, pkeep, is_training):

    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
    weight_decay = 0.00004
    stddev=0.1
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with tf.variable_scope("InceptionV3", "InceptionV3", [images]) as scope:

        with tf.contrib.slim.arg_scope(
                [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=True):
            with tf.contrib.slim.arg_scope(
                    [tf.contrib.slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params):
                net, end_points = inception_v3_base(images, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = tf.nn.dropout(net, pkeep, name='droplast')
                    net = flatten(net, scope="flatten")
    
    with tf.variable_scope('output') as scope:
        
        weights = tf.Variable(tf.truncated_normal([2048, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        _activation_summary(output)
    return output


def inception_v3_test(nlabels, images, pkeep, is_training):

    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
    weight_decay = 0.00004
    stddev=0.1
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with tf.variable_scope("InceptionV3", "InceptionV3", [images]) as scope:

        with tf.contrib.slim.arg_scope(
                [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=True):
            with tf.contrib.slim.arg_scope(
                    [tf.contrib.slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params):
                net, end_points = inception_v3_base(images, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = tf.nn.dropout(net, pkeep, name='droplast')
                    net = flatten(net, scope="flatten")
    
    with tf.variable_scope('output') as scope:
        
        weights = tf.Variable(tf.truncated_normal([2048, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        _activation_summary(output)
    return output,net
