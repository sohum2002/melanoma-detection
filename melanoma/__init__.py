from flask import Flask, request
import scipy
from scipy import ndimage
from scipy.misc import imread, imsave
import skimage
from skimage import color, filter

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    #Retrieve image
    img_url = request.data
    img = imread(app.root_path + "/img/" + img_url)/255.

    #Preprocess
    p = Preprocessing()
    preprocessed_img = p.preprocessImage(img)

    #Extract lesion from image
    img_lesion = extract_lesion(preprocessed_img)

    #Run in predict

    #Send back response in JSON
    return 'Melanoma prediction started'
