import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime
from geopy.geocoders import Nominatim
import zipfile


app = Flask(__name__)
with zipfile.ZipFile('model.zip', 'r') as my_zip:
    my_zip.extractall('./')

model = pickle.load(open(r'.\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = []
    Nominatim(domain='localhost:8080', scheme='http')
    locator = Nominatim(user_agent='p7raneeth')
    pickup_point = request.form.get('pickup point')
    pickup_point_crd = locator.geocode(pickup_point)
    drop_point = request.form.get('drop point')
    drop_point_crd = locator.geocode(drop_point)
    year = datetime.strftime(datetime.now(), format = '%Y')
    month = datetime.strftime(datetime.now(), format = '%m')
    hour = datetime.strftime(datetime.now(), format = '%H')
    date = datetime.strftime(datetime.now(), format = '%d')
    psng_cnt = request.form.get('psng count')
    d = calculate_distance(pickup_point_crd, drop_point_crd)
    dow = datetime.today().weekday()
    int_features.append([pickup_point_crd[1][1], pickup_point_crd[1][0], drop_point_crd[1][1], drop_point_crd[1][0],
                        psng_cnt, year, month, hour, date, dow, d])
    prediction = model.predict(int_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Taxi fare should be $ {} @ 90 % Accuracy'.format(output))


def calculate_distance(pickup_point, drop_point):
    distance = []
    lat1 = np.radians(pickup_point[1][0])
    lat2 = np.radians(drop_point[1][0])
    long1 = np.radians(pickup_point[1][1])
    long2 = np.radians(drop_point[1][1])
    delta_lat = np.radians(lat1 - lat2)
    delta_long = np.radians(long1 - long2)
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_long / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (6371 * c)
    distance = d
    return distance


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
