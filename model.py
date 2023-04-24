from __future__ import absolute_import
from matplotlib import pyplot as plt
from keras.layers import Conv2D, UpSampling2D, InputLayer
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import random
import math
import pickle
import os

from skimage import img_as_ubyte

from PIL import Image

def load_data(file):
    with open(f'data/{file}.p', 'rb') as data_file:
        data = pickle.load(data_file)
    return data


def main():
    model = Sequential()
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
    model.compile(optimizer='rmsprop', loss='mse')

    list_ds = tf.data.Dataset.list_files('data/Images/*', shuffle=False)
    list_ds = list_ds.shuffle(8091, reshuffle_each_iteration=False)
    
    val_size = int(8091 * 0.2)
    train_ds = list_ds.take(1000)
    val_ds = list_ds.skip(1000).take(1000)
    test_ds = list_ds.skip(2000).take(100)


    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, fancy_upscaling=False)
        img = tf.cast(img, tf.float64)
        img *= 1.0/255
        # Resize the image to the desired size
        return tf.image.resize(img, (256, 256))
    
    def process_path(file_path):
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)

        img = tfio.experimental.color.rgb_to_lab(img)
        l_image = img[:,:,0]
        l_image = tf.reshape(l_image, (1, 256, 256, 1))
        ab_image = img[:,:,1:]
        ab_image = tf.reshape(ab_image, (1, 256, 256, 2))

        return l_image, ab_image
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)

    model.fit(train_ds,
    validation_data=val_ds,
    epochs=1)

    output = model.predict(test_ds)
    output *= 128

    img = []
    for images, labels in test_ds.take(1):
        img = images

    # Output colorizations
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = img[0][:,:,0]
    cur[:,:,1:] = output[0][:,:,:]
    imsave("img_result.png", img_as_ubyte(tfio.experimental.color.lab_to_rgb(cur)))
    imsave("img_gray_version.png", img_as_ubyte(tfio.experimental.color.rgb_to_grayscale(tfio.experimental.color.lab_to_rgb(cur))))
    # imsave("img_result.png", img_as_ubyte(lab2rgb(cur)))
    # imsave("img_gray_version.png", img_as_ubyte(rgb2gray(lab2rgb(cur))))

    return

    images = load_data('train1')
    labels = load_data('train1_labels')

    model.fit(x=images, y=labels, batch_size=1, epochs=1)

    # images = load_data('train2')
    # labels = load_data('train2_labels')

    # model.fit(x=images, y=labels, batch_size=1, epochs=1)
    
    # train3_images = load_data('train3')
    # train3_labels = load_data('train3_labels')

    # Save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    print(model.evaluate(images, labels, batch_size=1))
    output = model.predict(images)
    output *= 128

    # Output colorizations
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = images[0][:,:,0]
    cur[:,:,1:] = output[0]
    imsave("img_result.png", lab2rgb(cur))
    imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))
    return


if __name__ == '__main__':
    main()
