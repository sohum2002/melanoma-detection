from sklearn.externals import joblib
from flask import Flask, request

app = Flask(__name__)

def predict_input(feature_list):
    clf = retrieve_classifier()
    #prediction_class = clf.predict(feature_list)
    prediction_prob = clf.predict_proba(feature_list)
    #print "[Prediction] Prediction class" + str(prediction_class)
    print "[Prediction] Prediction probability" + str(prediction_prob)
    return prediction_prob

def retrieve_classifier():
    clf = joblib.load(app.root_path + "/linear_model.pkl")
    return clf
