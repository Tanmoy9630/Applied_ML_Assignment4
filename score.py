"""score.ipynb
"""

import joblib
import sklearn
import train
import pandas as pd
from train import data_prep
filename = "best_model.joblib"
best_model = joblib.load(filename) ## Loading the previously saved "best model"

def score(text:str, model, threshold:float):
    testing_X = data_prep(text) ## preparing the input text/msg 
    propensity = ((model.predict_proba(testing_X)).tolist())[0][1]
    if propensity >= threshold: # Determine the prediction based on the propensity and the specified threshold.
        prediction = 1
    else:
        prediction = 0
    return prediction, propensity
