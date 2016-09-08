# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
'''
Coding Just for Fun
Created by burness on 16/8/30.
'''
""" Finetuning Example. Using weights from model trained in
convnet_cifar10.py to retrain network for a new task (your own dataset).
All weights are restored except last layer (softmax) that will be retrained
to match the new task (finetuning).
"""


import tflearn
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_image_dataset_from_dir
import os
from config import  *
import tensorflow as tf



def vgg16(placeholderX=None):

    x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                           placeholder=placeholderX)

    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, 12, activation='softmax', scope='fc8',restore=False)

    return x



#
data_dir = data_path
from tflearn.data_utils import image_preloader
X,Y = image_preloader('files_list', image_shape = (224,224),mode='file',categorical_labels=True,normalize=True,files_extension=['.jpg', '.png'])


num_classes = 12
with tf.device('/gpu:2'):
    softmax = vgg16()
    regression = regression(softmax, optimizer='adam',
                            loss='categorical_crossentropy',
                            learning_rate=0.001,restore=False)
    model = tflearn.DNN(regression, checkpoint_path='finetuning-cv_bot',
                        max_checkpoints=3, tensorboard_verbose=2, tensorboard_dir="./logs")
    model_file = os.path.join(model_path, "vgg16.tflearn")
    model.load(model_file,weights_only=True)
    # Start finetuning
    model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_epoch=False, snapshot_step=200, run_id='finetuning')
    model.save('animal-classifier')
