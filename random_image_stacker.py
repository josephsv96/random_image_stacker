"""Paths of background images and foreground images should be given.
The images should either be seperate PNG files or NPY files. For BMP files,
alpha channel will be added to make it work.
The main program relies on the alpha channel to make the annotation.
"""

import numpy as np
import cv2
from tqdm import tqdm
# import imgaug
# import glob
# import random
# import matplotlib.pyplot as plt
# import pathlib

# from imgaug import augmenters as iaa
from PIL import Image


# from random import seed
from random import randint
# from random import random
# seed(1)


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
        if img.shape[-1] == 3:
            img_arr[i, :, :, :3] = img
            img_arr[i, :, :, 3].fill(255)
        else:
            img_arr[i, :, :, :] = img

    return img_arr


def load_bg_fg_img(bg_img_dir, fg_img_dir, bg_scaling, fg_scaling):
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
        fg_arr {numpy.array} -- Image array of fg images
    """
    bg_img_paths = list(bg_img_dir.glob('**/*.bmp'))
    fg_img_paths = list(fg_img_dir.glob('**/*.bmp'))

    bg_arr = img_arr_from_path(bg_img_paths, bg_scaling)
    print('bg_arr', bg_arr.shape)

    fg_arr = img_arr_from_path(fg_img_paths, fg_scaling)
    print('fg_arr', fg_arr.shape)

    return bg_arr, fg_arr


def labels_from_file(img_dir, delim='_'):
    """To get labels of images from image file names

    Arguments:
        img_dir {PosixPath} -- Path of image directory

    Keyword Arguments:
        delim {str} -- Delimiter in file name (default: {'_'})

    Returns:
        labels {numpy.array}-- 1-d array of integer class labels
        label_config {dict} -- Config of class labels
    """
    img_paths = list(img_dir.glob('**/*.bmp'))
    cat_labels = []
    # To get the first word in file name
    for i in range(len(img_paths)):
        cat_labels.append(img_paths[i].stem.split(delim)[0])
    cat_labels = np.array(cat_labels)
    labels_unique, labels = np.unique(cat_labels, return_inverse=True)

    # Config
    label_config_keys = list(np.unique(labels) + 1)
    # To convert keys from int64 to int
    label_config_keys = [int(i) for i in label_config_keys]
    label_config = dict(zip(label_config_keys, labels_unique))

    return labels+1, label_config


# REFACTORED UNTIL HERE


def rand_pos_gen(aug_config):
    # rename to random_val_gen

    # Initializing config
    base_x = aug_config['base_x']
    base_y = aug_config['base_y']
    x_var = aug_config['x_var']
    y_var = aug_config['y_var']
    w_init = aug_config['w_init']
    h_init = aug_config['h_init']
    w_var = aug_config['w_var']
    h_var = aug_config['h_var']

    # Generate random positions
    # within the base limits
    x_init = base_x + int(np.random.randn(1) * x_var)
    y_init = base_y + int(np.random.randn(1) * y_var)  # on the base

    # Generate random sizing
    # within the base limits
    w_rand = w_init + int(np.random.randn(1) * w_var)
    h_rand = h_init + int(np.random.randn(1) * h_var)  # on the base

    return [x_init, y_init, w_rand, h_rand]


def random_box(img_arr, img_labels):
    # ADD LABELS ARRAY AS ARGUMENT HERE
    # ANNOTATION DATA DEPEND ON LABELS
    # img_labels = np.arange(0, img_arr.shape[0])
    random_index = randint(0, len(img_arr) - 1)
    return [img_arr[random_index], img_labels[random_index]]


def place_box(bg_img, fg_img, x_lim, y_lim, width, height):
    x1 = x_lim
    x2 = x1 + width
    y1 = y_lim
    y2 = y1 + height

    fg_resized = cv2.resize(fg_img, (x2-x1, y2-y1))
    for i in range(0, y2-y1):
        for j in range(0, x2-x1):
            if (fg_resized[i, j, 3] >= 100):
                bg_img[y1+i, x1+j, :] = fg_resized[i, j, :]

    # Creating annotation of that BOX
    # WARNING: Taking the alpha channel as annotation, is decpreciated
    annotation = np.zeros((bg_img.shape[0], bg_img.shape[1]))
    fg_annot = fg_resized[:, :, 3]
    annotation[y1:y2, x1:x2] = fg_annot/255

    return bg_img, annotation, x2, y2


def generate_stacked_img(bg_arr, fg_arr, fg_labels, aug_config, box_num=4):
    # ADD randomstate funtionality to reproduce results
    # print('Using aug config:', aug_config)
    # init_im = empty_scene
    # WRONG!! bg_arr should also come with labels
    empty_scene, _ = random_box(bg_arr, fg_arr)
    init_im = np.zeros(empty_scene.shape)
    annots = []

    # Generating random initial positions for box 1 => top left
    [x_init, y_init, w_rand, h_rand] = rand_pos_gen(aug_config)

    # BOX_1: x_lim => random, y_lim => on the board (small variation)
    [box, box_class] = random_box(fg_arr, fg_labels)
    init_im, annot_1, b1_x, b1_y = place_box(
        init_im, box, x_init, y_init, w_rand, h_rand)
    annot_1 = annot_1 * box_class
    annots.append(annot_1)

    # Generating random initial positions box 1 => bottom left
    [x_init, y_init, w_rand, h_rand] = rand_pos_gen(aug_config)

    # BOX_2: x_lim => random, y_lim => (y_lim + height) of BOX_1
    [box, box_class] = random_box(fg_arr, fg_labels)
    init_im, annot_2, b2_x, b2_y = place_box(
        init_im, box, x_init, b1_y, w_rand, h_rand)
    annot_2 = annot_2 * box_class
    annots.append(annot_2)

    # Generating random initial positions box 1 => top right
    [x_init, y_init, w_rand, h_rand] = rand_pos_gen(aug_config)

    # BOX_3: x_lim => (x_lim + width) of BOX_1
    # y_lim => on the board (small variation)
    [box, box_class] = random_box(fg_arr, fg_labels)
    init_im, annot_3, b3_x, b3_y = place_box(
        init_im, box, b1_x, y_init, w_rand, h_rand)
    annot_3 = annot_3 * box_class
    annots.append(annot_3)

    if box_num == 4:
        # BOX_4: x_lim => min( (x_lim + width) of BOX_1 and BOX_2 )
        # y_lim => (y_lim + height) of BOX_3
        [box, box_class] = random_box(fg_arr, fg_labels)
        init_im, annot_4, b4_x, b4_y = place_box(
            init_im, box, min(b1_x, b2_x), b3_y, w_rand, h_rand)
        annot_4 = annot_4 * box_class
        annots.append(annot_4)

    # Annotations
    annotations = sum(annots)
    # Limiting values above (only works when above 6, not ideal way)
    annotations[annotations > 6] = 0

    image = init_im

    return image, annotations, empty_scene


def overlay_images(src1, src2):
    """Merge src2 over src1 based on the alpha channel indices of src2
    !!!Warning: src2 is required to have 4 channels

    Arguments:
        src1 {numpy.array} -- Base image
        src2 {numpy.array} -- Overlayed image

    Returns:
        dst {numpy.array} -- Merged image with 3 output channels
    """
    mask = ((src2[:, :, 3] - 255) * -1)/255
    dst = np.zeros([src1.shape[0], src1.shape[1], 3])
    for i in range(3):
        dst[:, :, i] = np.add(src1[:, :, i] * mask, src2[:, :, i])

    return dst


def random_overlay(bg_arr, fg_arr, fg_labels, aug_config, box_num):
    # base_img = bg_arr[0]
    stacked_im, annot_im, base_img = generate_stacked_img(bg_arr, fg_arr,
                                                        fg_labels, aug_config,
                                                        box_num)

    src1 = base_img
    src2 = stacked_im

    output_img = overlay_images(src1, src2)

    return output_img, annot_im


def random_img_gen(bg_arr, fg_arr, fg_labels, aug_config, box_num, gen_num):
    image_set = np.zeros([gen_num, bg_arr.shape[1], bg_arr.shape[2], 3])
    label_set = np.zeros([gen_num, bg_arr.shape[1], bg_arr.shape[2], 1])
    for i in tqdm(range(gen_num)):
        merged_im, annot = random_overlay(bg_arr, fg_arr, fg_labels,
                                            aug_config, box_num)

        image_set[i, :, :, :] = merged_im
        label_set[i, :, :, 0] = annot

    return image_set, label_set
