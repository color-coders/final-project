import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from skimage import color

from IPython import embed

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# def load_img(img_path):
# 	out_np = np.asarray(Image.open(img_path))
# 	if(out_np.ndim==2):
# 		out_np = np.tile(out_np[:,:,None],3)
# 	return out_np

# def resize_img(img, HW=(256,256), resample=3):
# 	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

# def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
# 	# return original size L and resized L as torch Tensors
# 	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
# 	img_lab_orig = color.rgb2lab(img_rgb_orig)
# 	img_lab_rs = color.rgb2lab(img_rgb_rs)

# 	img_l_orig = img_lab_orig[:,:,0]
# 	img_l_rs = img_lab_rs[:,:,0]

# 	tens_orig_l = tf.Tensor(img_l_orig)[None,None,:,:]
# 	tens_rs_l = tf.Tensor(img_l_rs)[None,None,:,:]

# 	return (tens_orig_l, tens_rs_l)

# def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
# 	# tens_orig_l 	1 x 1 x H_orig x W_orig
# 	# out_ab 		1 x 2 x H x W

# 	HW_orig = tens_orig_l.shape[2:]
# 	HW = out_ab.shape[2:]

# 	# call resize function if needed
# 	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
# 		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
# 	else:
# 		out_ab_orig = out_ab

# 	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
# 	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

# def load_data(data_folder):
#     '''
#     Method that was used to preprocess the data in the data.p file. You do not need 
#     to use this method, nor is this used anywhere in the assignment. This is the method
#     that the TAs used to pre-process the Flickr 8k dataset and create the data.p file 
#     that is in your assignment folder. 

#     Feel free to ignore this, but please read over this if you want a little more clairity 
#     on how the images and captions were pre-processed 
#     '''
#     text_file_path = f'{data_folder}/captions.txt'

#     with open(text_file_path) as file:
#         examples = file.read().splitlines()[1:]
    
#     #map each image name to a list containing all 5 of its captons
#     image_names_to_captions = {}
#     for example in examples:
#         img_name, caption = example.split(',', 1)
#         image_names_to_captions[img_name] = image_names_to_captions.get(img_name, []) + [caption]



#     def get_all_captions(image_names):
#         to_return = []
#         for image in image_names:
#             captions = image_names_to_captions[image]
#             for caption in captions:
#                 to_return.append(caption)
#         return to_return


#     # get lists of all the captions in the train and testing set
#     train_captions = get_all_captions(train_image_names)
#     test_captions = get_all_captions(test_image_names)

#     #remove special charachters and other nessesary preprocessing
#     window_size = 20

#     # count word frequencies and replace rare words with '<unk>'
#     word_count = collections.Counter()
#     for caption in train_captions:
#         word_count.update(caption)

#     def unk_captions(captions, minimum_frequency):
#         for caption in captions:
#             for index, word in enumerate(caption):
#                 if word_count[word] <= minimum_frequency:
#                     caption[index] = '<unk>'

#     unk_captions(train_captions, 50)
#     unk_captions(test_captions, 50)

#     # pad captions so they all have equal length
#     def pad_captions(captions, window_size):
#         for caption in captions:
#             caption += (window_size + 1 - len(caption)) * ['<pad>'] 
    
#     pad_captions(train_captions, window_size)
#     pad_captions(test_captions,  window_size)

#     # assign unique ids to every work left in the vocabulary
#     word2idx = {}
#     vocab_size = 0
#     for caption in train_captions:
#         for index, word in enumerate(caption):
#             if word in word2idx:
#                 caption[index] = word2idx[word]
#             else:
#                 word2idx[word] = vocab_size
#                 caption[index] = vocab_size
#                 vocab_size += 1
#     for caption in test_captions:
#         for index, word in enumerate(caption):
#             caption[index] = word2idx[word] 
    
#     # use ResNet50 to extract image features
#     print("Getting training embeddings")
#     train_image_features, train_images = get_image_features(train_image_names, data_folder)
#     print("Getting testing embeddings")
#     test_image_features,  test_images  = get_image_features(test_image_names, data_folder)

#     return dict(
#         train_captions          = np.array(train_captions),
#         test_captions           = np.array(test_captions),
#         train_image_features    = np.array(train_image_features),
#         test_image_features     = np.array(test_image_features),
#         train_images            = np.array(train_images),
#         test_images             = np.array(test_images),
#         word2idx                = word2idx,
#         idx2word                = {v:k for k,v in word2idx.items()},
#     )

def shuffle(images):
    # randomly split examples into training and testing sets
    random.seed(0)
    random.shuffle(images)
    train = images[:6400]
    test = images[6400:]
    return train, test

def process_image(image_names):
    images = []
    pbar = tqdm(image_names)
    for i, image_name in enumerate(pbar):
        img_path = f'data/Images/{image_name}'
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{img_path};")
        with Image.open(img_path) as img:
            img_array = np.asarray(img.resize((256,256)))
        
        img_lab_rs = color.rgb2lab(img_array)
        img_l_rs = img_lab_rs[:,:,0]

        tens_rs_l = tf.convert_to_tensor(img_l_rs)[None,None,:,:]
        images.append(tens_rs_l)

    return images

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]
	print(img_l_orig.shape)
	print(img_l_rs.shape)
	img_ab = img_lab_rs[:, :,1:]
	print(img_ab.shape)

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]
	img_ab = torch.Tensor(img_ab)[None,None,:,:]
	img_ab.reshape(1, 2, 256, 256)

	return (tens_orig_l, tens_rs_l, img_ab)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

if __name__ == '__main__':
    ## Download this and put the Images and captions.txt indo your ../data directory
    ## Flickr 8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
    image = process_image(['10815824_2997e03d76.jpg'])
    img = load_img('data/Images/667626_18933d713e.jpg')
    (tens_l_orig, tens_l_rs, img_ab) = preprocess_img(img, HW=(256,256))
    print(tens_l_orig.shape)
    print(img_ab.shape)

    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))

    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.imshow(img_bw)
    plt.axis('off')
    plt.show()
