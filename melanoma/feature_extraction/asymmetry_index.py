import numpy as np
import skimage
from skimage import color, filter

def checkOverlap(shape1, shape2):
    #Find the accuracy of symmetry
    all_pixels = 0.
    correct = 0.
    wrong = 0.

    for i in range(shape1.shape[0]):
        for j in range(shape1.shape[1]):

            curr_pixel1 = (shape1[i][j])
            curr_pixel2 = (shape2[i][j])

            if(curr_pixel1 or curr_pixel2):
                all_pixels += 1
                if(curr_pixel1 and curr_pixel2):
                    correct += 1
                else:
                    wrong += 1

    return correct, wrong, all_pixels

def get_asymmetry_index(img):
    imgcolor = img
    img = skimage.color.rgb2gray(img)
    x = []
    y = []

    #DOING FOR THE FIRST TIME TO GET LEFT AND TOP
    top = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    left = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    #Don't want to take the white parts
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 1:
                x.append(j)
                y.append(i)

    # Trying to find center, x-intercept and y-intercept
    centroid = (sum(x) / len(x), sum(y) / len(y))
    y_axis = centroid[1]
    x_axis = centroid[0]

    #Performing splitting for top/down images
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] < 0.95):
                if(i < y_axis):
                    top[i][j] = True

    #Performing splitting for left/right images
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] < 0.95):
                if (j < x_axis):
                    left[i][j] = True


    #DOING FOR FLIP UP/DOWN TO GET THE DOWN PART
    flipped_ud = np.flipud(img)

    bottom = np.zeros((flipped_ud.shape[0], flipped_ud.shape[1]), dtype=bool)

    #Performing splitting for top/down images
    for i in range(flipped_ud.shape[0]):
        for j in range(flipped_ud.shape[1]):
            if(flipped_ud[i][j] < 0.95):
                if(i < y_axis):
                    bottom[i][j] = True

    #DOING FOR FLIP UP/DOWN TO GET THE DOWN PART
    flipped_lr = np.fliplr(img)

    right = np.zeros((flipped_lr.shape[0], flipped_lr.shape[1]), dtype=bool)

    #Performing splitting for top/down images
    for i in range(flipped_lr.shape[0]):
        for j in range(flipped_lr.shape[1]):
            if(flipped_lr[i][j] < 0.95):
                if(j < x_axis):
                    right[i][j] = True


    correct_TB, wrong_TB, all_TB = checkOverlap(top, bottom)
    correct_LR, wrong_LR, all_LR = checkOverlap(left, right)

    return 1 - sum([correct_TB / all_TB, correct_TB / all_LR])/2.
