'''
Coding Just for Fun
Created by burness on 16/8/31.
'''
import tflearn
from tflearn.layers.estimator import regression

x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                       placeholder=None)

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

softmax = tflearn.fully_connected(x, 12, activation='softmax', scope='fc8')


regression = regression(softmax, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)

model = tflearn.DNN(regression, checkpoint_path='finetuning-cv_bot',
                    max_checkpoints=3, tensorboard_verbose=2, tensorboard_dir="./logs")

model.load('animal-classifier')

model.predict()
