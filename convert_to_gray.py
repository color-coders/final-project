from skimage import io, transform, color
import numpy as np


def convert_gray(img):
    rgb = io.imread(img).astype(np.float64)
    rs = transform.resize(rgb, (256,256))
    gray = color.rgb2gray(rs).astype(np.uint8)
    return gray

if __name__ == '__main__':
    data_path = 'data_test/*.jpg'
    gray_data_resave_path = 'gray_data_resave'
    all_pic = io.ImageCollection(data_path, load_func=convert_gray)
    for i in range(len(all_pic)):
        io.imsave(f'{gray_data_resave_path}/{i}.jpg', all_pic[i])
