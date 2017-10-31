from binarize import *
import skimage
from skimage.measure import perimeter, label, regionprops
from skimage import color, filter
import math
from centroid import *

def get_border_irregularity_ci(img, show = False):
    img = skimage.color.rgb2gray(img)
    img = binarize(img)

    #Find area and perimeter
    label_img = label(img)
    region = regionprops(label_img)

    img_area = max([props.area for props in region]) #Want the max because they could be many spaces
    img_perimeter = max([props.perimeter for props in region])

    #Calculate CI's formula
    return (img_perimeter**2) / (4.*math.pi*img_area)

def get_border_edge_abruptness(img, show = False):
    #Import lesion and convert it to black/white
    img = skimage.color.rgb2gray(img)

    #Get the centroid and distances to edgesx
    _max_rad_pt, _max_rad, outline_img, centroid_to_edges, centroid = get_centroid(img)

    #Get the average distance
    mean_dist = np.mean(centroid_to_edges)

    binarized_img = img.astype(bool)
    binarized_img = ndimage.binary_fill_holes(binarized_img)

    #Get the perimeter of image
    label_img = label(binarized_img)
    region = regionprops(label_img)
    img_perimeter = max([props.perimeter for props in region])

    edge_score = 0.
    for d in centroid_to_edges:
        edge_score += (d - mean_dist)**2

    edge_score /= (img_perimeter * (mean_dist**2))

    return edge_score
