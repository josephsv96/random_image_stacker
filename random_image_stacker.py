import numpy as np
# import cv2
# import imgaug
# import glob
# import random
# import matplotlib.pyplot as plt
# import pathlib

# from imgaug import augmenters as iaa
from PIL import Image


def img_arr_from_path(img_paths, img_scaling):
    """Return a numpy array of the images from the given paths.
    Basic preprocessing can be done; here scaling according to given
    scale.


    !!! Warning: Only png image for now.
    Need to convert original images to png to use alpha channel funtionality

    Arguments:
        img_paths {PosixPath} -- Path of image directory
        img_scaling {list} -- Output scaling of image arr

    Returns:
        numpy.array -- Image array
    """
    img_arr = np.zeros([len(img_paths), img_scaling[0], img_scaling[1], 4])
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        img = img.resize(img_scaling)
        img = np.array(img, dtype=np.float32)
        img_arr[i, :, :, :] = img

    return img_arr


def load_bg_fg(bg_img_dir, fg_img_dir, bg_scaling, fg_scaling):
    """To load bg and fg images as numpy arrays
    !!! Warning: Only png image for now.
    Need to convert original images to png to use alpha channel funtionality

    Arguments:
        bg_img_dir {PosixPath} -- Path of bg image directory
        fg_img_dir {PosixPath} -- Path of fg image directory
        bg_scaling {list} -- Output size of bg
        fg_scaling {list} -- Output size of fg

    Returns:
        bg_arr {numpy.array} -- Image array of bg images
        bg_arr {numpy.array} -- Image array of bg images
    """
    bg_img_paths = list(bg_img_dir.glob('**/*.png'))
    fg_img_paths = list(fg_img_dir.glob('**/*.png'))

    bg_arr = img_arr_from_path(bg_img_paths, bg_scaling)
    print('bg_arr', bg_arr.shape)

    fg_arr = img_arr_from_path(fg_img_paths, fg_scaling)
    print('fg_arr', fg_arr.shape)

    return bg_arr, fg_arr
