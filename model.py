from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave

import numpy as np
import os
import random
import tensorflow as tf
import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2

        self.e = 1e-5
         
    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

        return

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        softmax = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        softmax = tf.reduce_mean(softmax)
        return softmax

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels - no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    index = list(range(len(train_inputs)))
    np.random.shuffle(index)
    tf.gather(train_inputs, index)
    tf.gather(train_labels, index)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    b = model.batch_size
    for i in range(0, int((len(train_inputs)/b)-1)):
        xBatch = train_inputs[i*b:(i+1)*b]
        yBatch = train_labels[i*b:(i+1)*b]

        xBatch = tf.image.random_flip_left_right(xBatch)

        with tf.GradientTape() as tape:
            logits = model.call(xBatch, is_testing=False)
            loss = model.loss(logits, yBatch)
            model.loss_list.append(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model.loss_list


def test(model, test_inputs, test_labels):
    acc = 0
    batch_num = 0
    b = model.batch_size
    for i in range(0, int((len(test_inputs)/b)-1)):
        xBatch = test_inputs[i*b:(i+1)*b]
        yBatch = test_labels[i*b:(i+1)*b]

        logits = model.call(xBatch, is_testing=False)
        acc += model.accuracy(logits, yBatch)
        batch_num += 1

    acc /= batch_num
    return acc

def main():
    X, Y = get_data()
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='rmsprop',loss='mse')

    model.fit(x=X, 
    y=Y,
    batch_size=1,
    epochs=1000)
    return


if __name__ == '__main__':
    main()