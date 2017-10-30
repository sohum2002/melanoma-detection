import scipy
from scipy import ndimage
import numpy as np
from binarize import *

def get_centroid(curr_img):
    curr_img = binarize(curr_img, show = True)
    outline_img = np.zeros((curr_img.shape[0], curr_img.shape[1]), dtype=bool)

    for i in range(1, curr_img.shape[0] - 1):
        for j in range(1, curr_img.shape[1] - 1):
            if (curr_img[i][j] == True): #if black...
                if((curr_img[i-1][j-1] == False) or
                   (curr_img[i-1][j] == False) or
                   (curr_img[i+1][j+1] == False) or
                   (curr_img[i][j+1] == False) or
                   (curr_img[i+1][j] == False) or
                   (curr_img[i][j-1] == False) or
                   (curr_img[i-1][j+1] == False) or
                   (curr_img[i+1][j-1] == False)):
                    outline_img[i][j] = True
                else:
                    outline_img[i][j] = False

    #Find centroid
    centroid = scipy.ndimage.measurements.center_of_mass(outline_img)

    #Use euclidean to find distances
    centroid_to_edges = []
    max_rad = -1
    max_x = -1
    max_y = -1

    for i in range(outline_img.shape[0]):
        for j in range(outline_img.shape[1]):
            if outline_img[i][j] == True:
                curr_diam = (np.abs(i - centroid[0])**2) + (np.abs(j - centroid[1])**2)
                curr_diam = np.sqrt(curr_diam)

                if(curr_diam > max_rad):
                    max_rad = curr_diam
                    max_x = i
                    max_y = j

                centroid_to_edges.append(curr_diam)
    return (max_y, max_x), max_rad, outline_img, centroid_to_edges, centroid[::-1]
