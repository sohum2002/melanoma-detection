from centroid import *
from color_variation import *
from color_homogeneity import *
from relative_chromaticity import *
from asymmetry_index import *
from border_irregularity import *
import skimage
from skimage import color, filter
import numpy as np

class FeatureExtractor():

    def normalize_feature(self, feature):
        feature_mean = np.mean(feature)
        feature_stddev = np.std(feature)
        return (feature - feature_mean) / feature_stddev

    def get_features(self, color_img):

        #Convert to black and white
        bw_img = skimage.color.rgb2gray(color_img)

        #Get centroid for later usage
        max_rad_point, max_rad, outline_img, centroid_to_edges, centroid = get_centroid(bw_img)
        print "[FEATURE] Centroid retrieved"

        #Getting the ratio of the radius of the image
        ratio = max_rad
        print "[FEATURE] Radius retrieved"

        #Color Variation
        r_lesion, g_lesion, b_lesion, r_skin, g_skin, b_skin = get_RGB(color_img)
        print "[FEATURE] Color Variation retrieved"

        #Color Homogeneity Varianceo
        all_var_r, all_var_g, all_var_b = get_color_homogeneity(color_img)
        normalized_var_r = all_var_r
        normalized_var_g = all_var_g
        normalized_var_b = all_var_b
        print "[FEATURE] Color Homogeneity retrieved"

        #Relative Chromaticity of RGB
        rel_r, rel_g, rel_b = get_relative_chromaticity(color_img)
        normalized_rel_r = rel_r
        normalized_rel_g = rel_g
        normalized_rel_b = rel_b
        print "[FEATURE] Relative Chromaticity of RGB retrieved"

        #Finding Asymmetry of Image
        asymmetry_index = get_asymmetry_index(color_img)
        print "[FEATURE] Asymmetry Index retrieved"

        #Border Irregularity Compactness Index
        border_irregularity_compactness_index = get_border_irregularity_ci(color_img)
        print "[FEATURE] Border Irregularity Compactness Index"

        #Border Irregularity - Edge Abruptness
        border_irregularity_edge_abruptness = get_border_edge_abruptness(color_img)
        print "[FEATURE] Border Irregularity Edge Abruptness"

        return np.asarray([
            normalized_var_r[0], #ColorHomogeneityVariance (R)
            normalized_var_g[0], #ColorHomogeneityVariance (G)
            normalized_var_b[0], #ColorHomogeneityVariance (B)
            normalized_rel_r, #ColorRelativeChromaticity (R)
            normalized_rel_g, #ColorRelativeChromaticity (G)
            normalized_rel_b,  #ColorRelativeChromaticity (B)
            asymmetry_index, #TODO Getting none
            border_irregularity_compactness_index,
            border_irregularity_edge_abruptness
        ])
