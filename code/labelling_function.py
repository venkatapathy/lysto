# -*- coding: utf-8 -*-
"""labelling-function.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19s_TjapwKyGTn67pUknNFtijww71eJ7n
"""

import numpy as np

def calculate_brown_score(image):
    image = image.reshape((30, 30, 3))

    # Extract the red channel
    red_channel = image[:, :, 0]

    # Calculate the brown score as the mean intensity of the red channel
    brown_score = np.mean(red_channel)

    return brown_score

def calculate_green_score(image):
    image = image.reshape((30, 30, 3))

    # Extract the red channel
    green_channel = image[:, :, 1]

    # Calculate the brown score as the mean intensity of the red channel
    green_score = np.mean(green_channel)

    return green_score    

def calculate_blue_score(image):
    image = image.reshape((30, 30, 3))

    # Extract the red channel
    blue_channel = image[:, :, 2]

    # Calculate the blue score as the mean intensity of the blue channel
    blue_score = np.mean(blue_channel)

    return blue_score

black_range =  [15, 25, 15, 25, 15, 25] 

def compute_black_score(image_patch):
    image_patch = image_patch.reshape((30, 30, 3))
    black_pixels = np.count_nonzero(
        (image_patch[:, :, 0] >= black_range[0]) &
        (image_patch[:, :, 0] <= black_range[1]) &
        (image_patch[:, :, 1] >= black_range[2]) &
        (image_patch[:, :, 1] <= black_range[3]) &
        (image_patch[:, :, 2] >= black_range[4]) &
        (image_patch[:, :, 2] <= black_range[5])
    )
    black_score = black_pixels
    return black_score




blue_range = [60, 105, 55, 120, 85, 170]  # Example blue color range in RGB channels
blue_range2 = [10, 55, 0, 26, 0, 45]      # Example blue color range in RGB channels


def compute_blue_score(image_patch):
    image_patch = image_patch.reshape((30, 30, 3))

    total_pixels = np.count_nonzero(np.any(image_patch != [0, 0, 0], axis=-1))
    blue_pixels = np.count_nonzero(
        (image_patch[:, :, 0] >= blue_range[0]) &
        (image_patch[:, :, 0] <= blue_range[1]) &
        (image_patch[:, :, 1] >= blue_range[2]) &
        (image_patch[:, :, 1] <= blue_range[3]) &
        (image_patch[:, :, 2] >= blue_range[4]) &
        (image_patch[:, :, 2] <= blue_range[5]) 
    )
    blue_pixels2 = np.count_nonzero(
        (image_patch[:, :, 0] >= blue_range2[0]) &
        (image_patch[:, :, 0] <= blue_range2[1]) &
        (image_patch[:, :, 1] >= blue_range2[2]) &
        (image_patch[:, :, 1] <= blue_range2[3]) &
        (image_patch[:, :, 2] >= blue_range2[4]) &
        (image_patch[:, :, 2] <= blue_range2[5]) 
    )
    blue_score = blue_pixels + blue_pixels2
    return blue_score

brown_range = [10, 55, 0, 28, 0, 45]   # Example brown color range in RGB channels

def compute_brown_score(image_patch):
    image_patch = image_patch.reshape((30, 30, 3))
    brown_pixels = np.count_nonzero(
        (image_patch[:, :, 0] >= brown_range[0]) &
        (image_patch[:, :, 0] <= brown_range[1]) &
        (image_patch[:, :, 1] >= brown_range[2]) &
        (image_patch[:, :, 1] <= brown_range[3]) &
        (image_patch[:, :, 2] >= brown_range[4]) &
        (image_patch[:, :, 2] <= brown_range[5])
    )
    brown_score = brown_pixels
    return brown_score