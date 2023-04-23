from __future__ import absolute_import
from matplotlib import pyplot as plt
from keras.layers import Conv2D, UpSampling2D, InputLayer
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import tensorflow as tf
import numpy as np
import random
import math
import pickle

# ensures that we run only on cpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_data(file):
    with open(f'data/{file}.p', 'rb') as data_file:
        data = pickle.load(data_file)
    return data


def main():
    train1_images = load_data('train1')
    train1_labels = load_data('train1_labels')
    print(train1_images.shape)
    # train2_images = load_data('train2')
    # train2_labels = load_data('train2_labels')
    # train3_images = load_data('train3')
    # train3_labels = load_data('train3_labels')
    print('it works!!')

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

    model.fit(x=train1_images, y=train1_labels, batch_size=1, epochs=1)

    print(model.evaluate(train1_images, train1_labels, batch_size=1))
    output = model.predict(train1_images)
    output *= 128
    # Output colorizations
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = train1_images[0][:,:,0]
    cur[:,:,1:] = output[0]
    imsave("img_result.png", lab2rgb(cur))
    imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))
    return


if __name__ == '__main__':
    main()
