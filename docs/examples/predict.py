# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np

def predict_new_velocity(x):
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

    # load the trained model
    model = tflearn.DNN(net, tensorboard_verbose=3)
    model.load("model.tflearn")

    predict_y = model.predict(x)
    new_y = np.argmax(predict_y, axis=1)
    return new_y.astype(np.uint8)
