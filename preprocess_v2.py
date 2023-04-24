from PIL import Image
from tqdm import tqdm
from skimage import color, io
import tensorflow as tf
import numpy as np

import pickle
import random

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def process_images(image_names, data_folder):
    train_images = []
    train_labels = []
    
    # Create a progress bar for iterating over a list of image_names
    pbar = tqdm(image_names)
 
    # Set the description of the current progress bar
    # Example: [1/1] Processing 'data/Images/10815824_2997e03d76.jpg': 100%
    for i, image_name in enumerate(pbar):
        img = f'{data_folder}/Images/{image_name}'
        
        # The Image class in the Pillow library is used to open images as an np array. 3 means RGB.
        with Image.open(img) as img: # print("img shape:", np.shape(img)): (333, 500, 3)
            img_array = np.array(img.resize((256,256))) # print("img_array shape", np.shape(img_array)): (256, 256, 3)
            
        # Convert RGB to LAB
        img_lab_rs = color.rgb2lab(1.0/255*img_array) # Shape: (256, 256, 3)
        
        # Extract Lightness from LAB and convert to tensor 
        img_l_rs = img_lab_rs[:,:,0]
        img_l_rs = img_l_rs.reshape((256, 256, 1))
        # tens_rs_l = tf.convert_to_tensor(img_l_rs) # tens_rs_l tf.Tensor([256 256], shape=(2,), dtype=int32)

        # Extract A and B from LAB and convert to tensor 
        img_ab_rs = img_lab_rs[:,:,1:]
        img_ab_rs /= 128
        # tens_rs_ab = tf.convert_to_tensor(img_ab_rs) # tens_rs_ab tf.Tensor([256 256 2], shape=(3,), dtype=int32)
        
        train_images += [img_l_rs]
        train_labels += [img_ab_rs]

    return train_images, train_labels

def load_data(data_folder, batch):
    train_images = []
    train_labels = []

    # all_pics = io.ImageCollection(f'{data_folder}/Images/*.jpg', load_func=load_imgs)

    text_file_path = f'{data_folder}/captions.txt'
    with open(text_file_path) as file:
        examples = file.read().splitlines()[1:]

    image_names_to_captions = {}

    for example in examples:
        img_name, caption = example.split(',', 1)
        image_names_to_captions[img_name] = image_names_to_captions.get(img_name, []) + [caption]

    # shuffle images
    shuffled_images = list(image_names_to_captions.keys())
    random.seed(0)
    random.shuffle(shuffled_images)

    # adjust splicing to change batch sizes
    if batch == 'train1':
        data = shuffled_images[1000:1100]
    elif batch == 'train2':
        data = shuffled_images[3500:6000]
    elif batch == 'train3':
        data = shuffled_images[6000:]
    else:
        data = shuffled_images[:1000]

    images, labels = process_images(data, data_folder)

    create_pickle(np.array(images), batch)
    create_pickle(np.array(labels), batch + '_labels')

def create_pickle(data, file):
    with open(f'data/{file}.p', 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    print(f'Data has been dumped into {file}.p!')


if __name__ == '__main__':
    ## Download this and put the Images into your ../data directory
    ## Flickr 8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
    
    load_data('data', 'train1')
    # load_data('data', 'train2')
    # load_data('data', 'train3')
    # load_data('data', 'test')
    