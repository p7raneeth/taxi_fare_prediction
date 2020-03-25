import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from datetime import datetime
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def read_data(path):
    data = pd.read_csv(os.path.join(path, 'test.csv'), parse_dates=['pickup_datetime', 'key'])
    return data

def transform_data(test_data):
    test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])
    test_data['year'] = pd.to_datetime(test_data['pickup_datetime']).dt.year
    test_data['Month'] = pd.to_datetime(test_data['pickup_datetime']).dt.month
    test_data['Date'] = pd.to_datetime(test_data['pickup_datetime']).dt.day
    test_data['hour'] = pd.to_datetime(test_data['pickup_datetime']).dt.hour
    test_data['day_of_week'] = pd.to_datetime(test_data['pickup_datetime']).dt.dayofweek
    return test_data

def calculate_distance(X_test):
    print(X_test.columns)
    lat1 = np.radians(X_test.iloc[:, 3])
    lat2 = np.radians(X_test.iloc[:, 5])
    delta_lat = np.radians(X_test.iloc[:, 3] - X_test.iloc[:, 5])
    delta_long = np.radians(X_test.iloc[:, 2] - X_test.iloc[:, 4])
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_long / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (6371 * c)
    X_test['distance'] = d
    X_test.drop(['key', 'pickup_datetime'], axis=1, inplace=True)
    return X_test

def testdata_prediction(X_test):
    rf_model = pickle.load(open('model.pkl','rb'))
    print(X_test.columns)
    y_pred = rf_model.predict(X_test)
    return y_pred


if __name__ == '__main__':
    path = r'C:\Users\Vikesh\PycharmProjects\archive\test_data'
    data = read_data(path)
    data = transform_data(data)
    data = calculate_distance(data)
    pred_data = testdata_prediction(data)