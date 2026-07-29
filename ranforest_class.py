import joblib
import numpy as np
import pandas as pd

INFLATION_FACTOR = 1.867

class RandomForestModel:
    def __init__(self):
        self.model = joblib.load("random_forest_pipeline.pkl")
        self.pred_value = None
        self.inputs = None

    def predict(self, inputs):
        self.inputs = np.array(inputs).reshape(1, -1) #get inputs and reshape so that it is not [1, 2, 3] but [[1, 2, 3]]
        self.pred_value = self.model.predict(self.inputs)[0] * INFLATION_FACTOR
        return self.pred_value #I multiply the prediction value by an inflation factor.

    def print_prediction(self):
        print(f"Random Forest Model House Price Prediction: ${self.pred_value}")