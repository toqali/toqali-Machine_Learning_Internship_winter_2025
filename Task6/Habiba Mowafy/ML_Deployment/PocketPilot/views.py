from django.shortcuts import render
import pandas as pd
import pickle
import os
import numpy as np

# Load the trained model
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'PocketPilot', 'ml_models', 'random_forest_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Provide alternate path if the first one doesn't work
        alt_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'random_forest_model.pkl')
        with open(alt_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
# Load the preprocessor
def load_preprocessor():
    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'PocketPilot', 'ml_models', 'preprocessor.pkl')
    try:
        with open(preprocessor_path, 'rb') as file:
            preprocessor = pickle.load(file)
        return preprocessor
    except FileNotFoundError:
        # Provide alternate path if the first one doesn't work
        alt_path = os.path.join(os.path.dirname(__file__), 'ml_models', 'preprocessor.pkl')
        with open(alt_path, 'rb') as file:
            preprocessor = pickle.load(file)
        return preprocessor

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        # Extract all form data for rendering purposes
        form_data = {
            'user_id': request.POST.get('user_id', ''),
            'user_name': request.POST.get('user_name', ''),
            'driver_name': request.POST.get('driver_name', ''),
            'pickup_location': request.POST.get('pickup_location', ''),
            'destination': request.POST.get('destination', ''),
            'passenger_count': request.POST.get('passenger_count', '1'),
        }
        
        # Extract only the features needed for prediction
        try:
            # Process categorical features with the SAME encoding used during training
            car_condition_raw = request.POST.get('car_condition', '')
            traffic_condition_raw = request.POST.get('traffic_condition', '')
            
            # Apply the encoding
            car_condition_map = {'bad': 0, 'good': 1, 'very good': 2, 'excellent': 3}
            traffic_condition_map = {'flow traffic': 0, 'dense traffic': 1, 'congested traffic': 2}
            
            # Convert to lowercase for case insensitivity and map to numerical values
            car_condition = car_condition_map.get(car_condition_raw.lower(), 0)  # Default to 0 if not found
            traffic_condition = traffic_condition_map.get(traffic_condition_raw.lower(), 0)  # Default to 0 if not found
            
            # Process numerical features - FIXED: Corrected syntax error
            request_hour = int(request.POST.get('request_hour', '0'))
            request_day_of_week = int(request.POST.get('request_day_of_week', '0'))
            ride_distance = float(request.POST.get('ride_distance', '0'))
            
            # Process boolean features
            is_rush_hour = 1 if request.POST.get('is_rush_hour', '') == 'on' else 0
            is_weekend = 1 if request.POST.get('is_weekend', '') == 'on' else 0
            is_night = 1 if request.POST.get('is_night', '') == 'on' else 0
            is_holiday = 1 if request.POST.get('is_holiday', '') == 'on' else 0
            
            # Process weather features (one-hot encoded)
            weather_condition = request.POST.get('weather_condition', '')
            weather_cloudy = 1 if weather_condition == 'cloudy' else 0
            weather_rainy = 1 if weather_condition == 'rainy' else 0
            weather_stormy = 1 if weather_condition == 'stormy' else 0
            weather_sunny = 1 if weather_condition == 'sunny' else 0
            weather_windy = 1 if weather_condition == 'windy' else 0
            
            # Prepare features for model in the correct format
            features = {
                'car_condition': [car_condition],
                'traffic_condition': [traffic_condition],
                'request_hour': [request_hour],
                'request_day_of_week': [request_day_of_week],
                'ride_distance': [ride_distance],
                'is_rush_hour': [is_rush_hour],
                'is_weekend': [is_weekend],
                'is_night': [is_night],
                'is_holiday': [is_holiday],
                'weather_cloudy': [weather_cloudy],
                'weather_rainy': [weather_rainy],
                'weather_stormy': [weather_stormy],
                'weather_sunny': [weather_sunny],
                'weather_windy': [weather_windy]
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame(features)
            
            # Load model and preprocessor
            model = load_model()
            preprocessor = load_preprocessor()
            
            # Apply preprocessing transformation
            input_scaled = preprocessor.transform(input_df)
            
            # Make prediction using the transformed data
            prediction = model.predict(input_scaled)
            
            # Round to two decimal places for fare amount
            predicted_fare = round(float(prediction[0]), 2)
            
            # Add prediction to form data for rendering
            form_data['predicted_fare'] = predicted_fare
            
            # Debug line to see what's happening
            print(f"Prediction successful: ${predicted_fare}")
            
            return render(request, 'result.html', {'result': form_data})
            
        except Exception as e:
            import traceback
            error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(f"Error in prediction: {error_message}")  # Debug line
            return render(request, 'predict.html', {'error': error_message, 'form_data': form_data})
    
    return render(request, 'predict.html')