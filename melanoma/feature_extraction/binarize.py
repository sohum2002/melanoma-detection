import scipy
from scipy import ndimage

def binarize(img, show = False):

    #Binarize the image
    for idx1 in range(img.shape[0]):
        for idx2 in range(img.shape[1]):
            if img[idx1][idx2] > 0.9:
                img[idx1][idx2] = False
            else:
                img[idx1][idx2] = True

    img = ndimage.binary_fill_holes(img)

    return img
