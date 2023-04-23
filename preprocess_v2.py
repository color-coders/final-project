from PIL import Image
from tqdm import tqdm
from skimage import color, io
import tensorflow as tf
import numpy as np

import pickle
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# def shuffle(images):
#     # shuffle images
#     random.seed(0)
#     random.shuffle(images)
#     return images

# def split(images):
#     # randomly split examples into training and testing sets
#     train = images[:6400]
#     test = images[6400:]
#     return train, test

# def load_imgs(img):
#     images = io.imread(img)
#     return images

def process_images(image_names, data_folder):
    train_images = []
    train_labels = []
    
    # Create a progress bar for iterating over a list of image_names
    pbar = tqdm(image_names)
 
    # Set the description of the current progress bar
    # Example: [1/1] Processing 'data/Images/10815824_2997e03d76.jpg': 100%
    for i, image_name in enumerate(pbar):
        img = f'{data_folder}/Images/{image_name}'

        # removed pbar description for better performance
        # pbar.set_description(f"[{i+1}/{len(image_names)}] Processing '{img}'")
        
        # The Image class in the Pillow library is used to open images as an np array. 3 means RGB.
        with Image.open(img) as img: # print("img shape:", np.shape(img)): (333, 500, 3)
            img_array = np.asarray(img.resize((256,256))) # print("img_array shape", np.shape(img_array)): (256, 256, 3)
            
        # Convert RGB to LAB
        img_lab_rs = color.rgb2lab(img_array) # Shape: (256, 256, 3) 
        
        # Extract Lightness from LAB and convert to tensor 
        img_l_rs = img_lab_rs[:,:,0]
        tens_rs_l = tf.convert_to_tensor(img_l_rs) # tens_rs_l tf.Tensor([256 256], shape=(2,), dtype=int32)

        # Extract A and B from LAB and convert to tensor 
        img_ab_rs = img_lab_rs[:,:,1:]
        tens_rs_ab = tf.convert_to_tensor(img_ab_rs) # tens_rs_ab tf.Tensor([256 256 2], shape=(3,), dtype=int32)
        
        train_images += [tens_rs_l]
        train_labels += [tens_rs_ab]

    return train_images, train_labels

def load_data(data_folder):
    image_names = []
    train_images = []
    train_labels = []

    # all_pics = io.ImageCollection(f'{data_folder}/Images/*.jpg', load_func=load_imgs)

    # shuffled_pics = shuffle(list(all_pics))
    
    # for img_idx in range(len(all_pics)):
    #     image_name = os.path.basename(all_pics.files[img_idx])
    #     image_names.append(image_name)

    text_file_path = f'{data_folder}/captions.txt'
    with open(text_file_path) as file:
        examples = file.read().splitlines()[1:]

    #map each image name to a list containing all 5 of its captons
    image_names_to_captions = {}

    for example in examples:
        img_name, caption = example.split(',', 1)
        image_names_to_captions[img_name] = image_names_to_captions.get(img_name, []) + [caption]

    shuffled_images = list(image_names_to_captions.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    test_image_names = shuffled_images[:1000]
    train_image_names = shuffled_images[1000:]

    train_images, train_labels = process_images(train_image_names, data_folder)
    test_images, test_labels = process_images(test_image_names, data_folder)

    return dict(
        train_images = np.array(train_images),
        train_labels = np.array(train_labels),
        test_images = np.array(test_images),
        test_labels = np.array(test_labels)
    )

def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')

if __name__ == '__main__':
    ## Download this and put the Images into your ../data directory
    ## Flickr 8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
    
    # img = load_imgs('data_test/667626_18933d713e.jpg')
    # test, train = process_image('data')
    create_pickle('data')
    
