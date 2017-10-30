import scipy
from scipy import ndimage
from scipy.misc import imread, imsave
import skimage
from skimage import color, filter

class Preprocessing():
    def preprocess_img(self, img_color):
        #Store for later usage
        start = img_color

        #Convert to black & white
        img = skimage.color.rgb2gray(img_color)
        print "[PRE-PROCESSING] bw img done"

        #Change shape to fixed width of 340
        basewidth = img.shape[1] / 340.
        img = scipy.misc.imresize(arr = img, size = (int(img.shape[0]/basewidth), int(img.shape[1]/basewidth)))
        start = scipy.misc.imresize(arr = start, size = (int(start.shape[0]/basewidth), int(start.shape[1]/basewidth)))
        print "[PRE-PROCESSING] size normaliztion done"

        #Median filtering
        img = ndimage.median_filter(img, 7)
        print "[PRE-PROCESSING] median filtering done"

        #Sharpening
        filter_blurred_l = ndimage.gaussian_filter(img, 1)
        alpha = 3
        img = img + alpha * (img - filter_blurred_l)
        print "[PRE-PROCESSING] sharpening done"

        return img, start
