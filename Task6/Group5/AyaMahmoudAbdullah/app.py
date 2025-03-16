from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd  
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

FLASK_API_URL = "http://127.0.0.1:3000/predict" 

with open("random_forest_model.pkl", 'rb') as file:
    lr = pickle.load(file)

Car_condition_encoder = joblib.load('Car_condition_encoder.joblib')
pickup_period_encoder = joblib.load('pickup_period_encoder.joblib')
pickup_season_encoder = joblib.load('pickup_season_encoder.joblib')
Traffic_Condition_encoder = joblib.load('Traffic_Condition_encoder.joblib')
Weather_encoder = joblib.load('Weather_encoder.joblib')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_season(arg): 
    if arg in [12, 1, 2]:
        return "Winter"
    elif arg in [3, 4, 5]:
        return "Spring"
    elif arg in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"
    
def get_dayperiod(arg): 
    if arg > 5 and arg < 12:
        return "Morning"
    elif arg >= 12 and arg < 17:
        return "Afternoon"
    elif arg >= 17 and arg < 22:
        return "Evenning"
    else: # 22 -> 5
        return "Night"
    

@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("i am here")
        data = request.get_json()
        print("Received Data:", data)

        print("🔍 Car Condition Encoder Type:", type(Car_condition_encoder))
        print("🔍 Pickup Period Encoder Type:", type(pickup_period_encoder))
        print("🔍 Pickup Season Encoder Type:", type(pickup_season_encoder))
        print("🔍 Traffic Condition Encoder Type:", type(Traffic_Condition_encoder))
        print("🔍 Weather Encoder Type:", type(Weather_encoder))

        input_data = data.get('features')

        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        feature_names = ['User_ID', 'User_Name', 'Driver_Name', 'Car_Condition', 'Weather',
       'Traffic_Condition', 'key', 'pickup_datetime',
       'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'passenger_count', 'hour', 'day', 'month',
       'weekday', 'year', 'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist',
       'nyc_dist', 'distance', 'bearing']

        input_df = pd.DataFrame([input_data], columns=feature_names)
        print("DataFrame Created:", input_df)
        input_df.drop('User_ID',axis=1, inplace=True)
        input_df.drop('User_Name',axis=1, inplace=True)
        input_df.drop('Driver_Name',axis=1, inplace=True)
        input_df.drop('key',axis=1, inplace=True)
        input_df.drop('pickup_datetime',axis=1, inplace=True)

        input_df['pickup_season'] = input_df['month'].apply(get_season)
        input_df['pickup_period'] = input_df['hour'].apply(get_dayperiod)

        try:
            input_df['Car_Condition'] = Car_condition_encoder.transform(input_df['Car_Condition'])
            input_df['pickup_period'] = pickup_period_encoder.transform(input_df['pickup_period'])
            input_df['pickup_season'] = pickup_season_encoder.transform(input_df['pickup_season'])
            input_df['Traffic_Condition'] = Traffic_Condition_encoder.transform(input_df['Traffic_Condition'])
            input_df['Weather'] = Weather_encoder.transform(input_df['Weather'])
        except Exception as e:
            print("Encoding Error:", e)
            return jsonify({'error': f"Encoding error: {str(e)}"}), 400

        input_df = scaler.transform(input_df)
        print("DataFrame afterupdates:", input_df)
        prediction = lr.predict(input_df)
    
        return jsonify({'predict': prediction.tolist()})

    except KeyError as e:
        return jsonify({'error': f"Missing key in request data: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({'error': f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)