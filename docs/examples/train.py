# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np

# Data loading and preprocessing
import load_data as ld 
X, Y, testX, testY = ld.load_data(one_hot=True)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 360, 3])
dense1 = tflearn.fully_connected(input_layer, 64, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.5)
dense2 = tflearn.fully_connected(dropout1, 64, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.5)
softmax = tflearn.fully_connected(dropout2, 8, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=50, validation_set=(testX, testY),
          show_metric=True, batch_size = 30, run_id="learning")

model.save("model.tflearn")
