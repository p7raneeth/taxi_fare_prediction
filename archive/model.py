import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from datetime import datetime
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from test import calculate_distance

def read_data(path):
    data = pd.read_csv(os.path.join(path, 'train.csv'), parse_dates=['pickup_datetime', 'key'])    
    y_train = data['fare_amount']
    data.drop(['Unnamed: 0', 'fare_amount'], axis=1, inplace=True)
    return data, y_train

def transform_data(train_data):
    train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])
    train_data['year'] = pd.to_datetime(train_data['pickup_datetime']).dt.year
    train_data['Month'] = pd.to_datetime(train_data['pickup_datetime']).dt.month
    train_data['Date'] = pd.to_datetime(train_data['pickup_datetime']).dt.day
    train_data['hour'] = pd.to_datetime(train_data['pickup_datetime']).dt.hour
    train_data['day_of_week'] = pd.to_datetime(train_data['pickup_datetime']).dt.dayofweek
    return train_data

def calculate_distance(X_train):
    print(X_train.iloc[:5, :7])
    lat1 = np.radians(X_train.iloc[:, 4])
    lat2 = np.radians(X_train.iloc[:, 6])
    delta_lat = np.radians(X_train.iloc[:, 4] - X_train.iloc[:, 6])
    delta_long = np.radians(X_train.iloc[:, 3] - X_train.iloc[:, 5])
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_long / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (6371 * c)
    X_train['distance'] = d
    return X_train


def model_training(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators='warn', criterion='mse', random_state=7)
    X_train.drop(['Unnamed: 0.1','key', 'pickup_datetime'], axis=1, inplace=True)
    print(X_train.head())
    rf_model.fit(X_train, y_train)
    print(X_train.columns)
    print(f'no of features while training!! is {len(X_train.columns)}')
    return rf_model


if __name__ == "__main__":
    path = r'C:\Users\Vikesh\PycharmProjects\archive\training_data'
    data, y_train = read_data(path)
    data = transform_data(data)
    data = calculate_distance(data)
    model_output = model_training(data,y_train)
    pickle.dump(model_output, open('model.pkl','wb'))
    print('execution completed!!')