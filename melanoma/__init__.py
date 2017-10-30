from flask import Flask, request
import scipy
from scipy import ndimage
from scipy.misc import imread, imsave
import skimage
from skimage import color, filter
from preprocessing.preprocess import *
from preprocessing.macwe import *
from feature_extraction.feature_extractor import *
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
    preprocessed_img, start = p.preprocess_img(img)

    #Extract lesion from image
    img_lesion = extract_lesion(preprocessed_img, start)
    imsave(app.root_path + "/img/processed/" + img_url, img_lesion)

    #Feature Extraction
    f = FeatureExtractor()
    features = f.get_features(img_lesion)
    for i in features:
        print i

    #Train Model


    #Send back response in JSON
    return 'Melanoma prediction started'
