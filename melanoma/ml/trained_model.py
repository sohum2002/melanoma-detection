from sklearn.externals import joblib

def retrieve_classifier():
    clf = joblib.load('linear_model.pkl')
    return clf
