
import json
import time

import numpy as np
import pandas as pd
import azureml.core
from azureml.core.model import Model
import joblib

columns = ['gre', 'gpa', 'rank_1', 'rank_2', 'rank_3']

def init():
    global model
    
    print("Azure ML SDK version:", azureml.core.VERSION)
    model_name = 'aks-intro'
    print('Looking for model path for model: ', model_name)
    model_path = Model.get_model_path(model_name=model_name)
    print('Looking for model in: ', model_path)
    model = joblib.load(model_path)
    print('Model initialized:', time.strftime('%H:%M:%S'))

def run(input_json):     
    try:
        inputs = json.loads(input_json)
        data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)),
                               columns = columns)
        # Get the predictions...
        prediction = model.predict(data_df)
        prediction = json.dumps(prediction.tolist())
    except Exception as e:
        prediction = str(e)
    return prediction
