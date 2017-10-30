from trained_model import *

def predict_input(feature_list):
    print "Trying to predict for " + feature_list
    clf = retrieve_classifier()
    prediction = clf.predict(feature_list)
    print "Prediction: " + prediction
    return prediction
