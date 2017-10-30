import skimage
from skimage import color, filter
from skimage.filter import threshold_adaptive, threshold_otsu
from skimage.measure import perimeter, label, regionprops
from binarize import *
from color_variation import *

def get_color_homogeneity(color_img, show = False):

    all_var_r = []
    all_var_g = []
    all_var_b = []

    bw_img = skimage.color.rgb2gray(color_img)
    binarized_img = binarize(bw_img)

    #Find area and perimeter
    label_img = label(binarized_img)
    region = regionprops(label_img)
    area = max([props.area for props in region]) #Want the max because they could be many spaces

    #Call function to compute rgb values
    r_lesion, g_lesion, b_lesion, r_skin, g_skin, b_skin = get_RGB(color_img)

    mean_r_skin = np.mean(r_skin)
    mean_g_skin = np.mean(g_skin)
    mean_b_skin = np.mean(b_skin)

    var_r = 0.
    var_g = 0.
    var_b = 0.

    for i in range(len(r_lesion)):
        var_r += (r_lesion[i] - mean_r_skin)**2
        var_g += (g_lesion[i] - mean_g_skin)**2
        var_b += (b_lesion[i] - mean_b_skin)**2

    var_r /= area
    var_g /= area
    var_b /= area

    all_var_r.append(var_r)
    all_var_g.append(var_g)
    all_var_b.append(var_b)

    if(show):
        ppl.imshow(color_img)
        ppl.show()
    return all_var_r, all_var_g, all_var_b
