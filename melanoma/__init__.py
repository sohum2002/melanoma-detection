from flask import Flask, request
import scipy
from scipy import ndimage
from scipy.misc import imread, imsave
import skimage
from skimage import color, filter
from preprocessing.preprocess import *
from preprocessing.macwe import *
from feature_extraction.feature_extractor import *
from ml.prediction_model import *
from flask import json

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

    #Call trained model
    prediction_prob = predict_input(features)

    #Send back response in JSON
    json_str = {}
    json_str['probability_false'] = prediction_prob[0][0]
    json_str['probability_true'] = prediction_prob[0][1]
    response = app.response_class(
        response = json.dumps(json_str),
        status = 200,
        mimetype='application/json'
    )
    return response
