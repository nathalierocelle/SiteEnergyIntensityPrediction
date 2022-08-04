import joblib
import numpy as np
from xgboost import XGBRegressor
from category_encoders import TargetEncoder

target_enc = joblib.load(r'target_encoder.joblib')

def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)

def encode_value(entered_value,options):
    values = list(target_enc.fit_transform(options))
    keys = options
    dictionary = dict(zip(keys, values))
    encoded_value = dictionary[entered_value]
    return encoded_value
        