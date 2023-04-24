from __future__ import absolute_import
from keras.layers import Conv2D, UpSampling2D, InputLayer
from skimage.color import lab2rgb, rgb2gray
from skimage.io import imsave
from keras.models import Sequential

from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import tensorflow as tf
import numpy as np
import pickle


def load_data(file):
    with open(f'data/{file}.p', 'rb') as data_file:
        data = pickle.load(data_file)
    return data

def main():
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
    model.compile(optimizer='adam', loss='mse')

    for i in range(1, 2):
        images = load_data(f'train{i}')
        labels = load_data(f'train{i}_labels')

        model.fit(x=images, y=labels, batch_size=10, epochs=100)

    output = model.predict(images)
    output *= 128

    # Output colorizations
    for i in range(1, 6):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = images[i][:,:,0]
        cur[:,:,1:] = output[i][:,:,:]

        cur_gray = rgb2gray(lab2rgb(cur))
        cur = lab2rgb(cur)

        im = (255*cur).astype(np.uint8)
        imgray = (255*cur_gray).astype(np.uint8)
        imsave(f"img_result{i}.png", im)
        imsave(f"img_gray_version{i}.png", imgray)
    return

if __name__ == '__main__':
    main()
