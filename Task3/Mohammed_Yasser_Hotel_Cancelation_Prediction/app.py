from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def get_booking_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def prepare_input(data):
    car_parking_space = int(data.get('car_parking_space', 0))
    car_parking_space = np.log1p(car_parking_space) 

    lead_time = int(data.get('lead', 0))
    lead_time = np.log1p(lead_time)

    market_segment_mapping = {
        'Aviation': 0,
        'Complementary': 1,
        'Corporate': 2,
        'Offline': 3,
        'Online': 4
    }
    market_segment_type = data.get('market', 0)
    market_segment_encoded = market_segment_mapping.get(market_segment_type, 0)

    repeated = int(data.get('repeated', 0))
    repeated = np.log1p(repeated)

    average_price = float(data.get('average price', 0))
    print(average_price)

    special_requests = int(data.get('request', 0))
    special_requests = np.log1p(special_requests)

    total_nights = int(data.get('weekend', 0)) + int(data.get('week', 0))
    total_nights = np.log1p(total_nights)
    
    date_str = data.get('date', '')
    booking_month = datetime.strptime(date_str, '%Y-%m-%d').month if date_str else 0

    booking_season_mapping = {
        'Fall': 0,
        'Spring': 1,
        'Summer': 2,
        'Winter': 3
    }
    booking_season = get_booking_season(booking_month)
    booking_season_encoded = booking_season_mapping.get(booking_season, 0)

    booking_day_of_week = datetime.strptime(date_str, '%Y-%m-%d').weekday() if date_str else 0
    only_children = 1 if int(data.get('children', 0)) != 0 and int(data.get('adult', 0)) == 0 else 0
    is_family = 1 if int(data.get('adult', 0)) > 1 and int(data.get('children', 0)) > 0 else 0

    # One-hot encoding for meal
    meal = data.get('meal', 'Not Selected')
    meal_options = ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']
    meal_encoded = [1 if meal == option else 0 for option in meal_options]

    # One-hot encoding for room
    room = data.get('room', 'Room_Type_1')
    room_options = ['Room_Type_1', 'Room_Type_2', 'Room_Type_3', 'Room_Type_4', 'Room_Type_5', 'Room_Type_6', 'Room_Type_7']
    room_encoded = [1 if room == option else 0 for option in room_options]

    

    # Combine all features into a single array
    features = np.array([
        car_parking_space, lead_time, market_segment_encoded, repeated, average_price, special_requests,
        total_nights, booking_month, booking_season_encoded, booking_day_of_week, only_children, is_family
    ] + meal_encoded + room_encoded)

    features_to_scale = np.array([average_price, total_nights, lead_time]).reshape(1, -1)
    scaled_features = scaler.transform(features_to_scale)

    features[1] = scaled_features[0,2]
    features[6] = scaled_features[0,1]
    features[4] = scaled_features[0,0]

    print(features)
    return features


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Preprocess Features
    features = prepare_input(request.form)

    # Make prediction
    prediction = model.predict([features])[0]

    prediction_label = 'canceled' if prediction == 0 else 'not_canceled'

    # Return the result as JSON
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)