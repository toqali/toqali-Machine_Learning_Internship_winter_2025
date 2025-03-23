import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
import joblib

# Construct paths to model files in the 'model_files' directory
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model_files', 'Uber_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'model_files', 'X_scaler.pkl')
YSCALER_PATH = os.path.join(settings.BASE_DIR, 'model_files', 'y_scaler.pkl')

# Load the model and scalers
model = joblib.load(MODEL_PATH)
X_scaler = joblib.load(SCALER_PATH)
y_scaler = joblib.load(YSCALER_PATH)

# Define mapping dictionaries for friendly feature values
car_condition_mapping = {
    'Very Good': 3,
    'Excellent': 1,
    'Bad': 0,
    'Good': 2
}

traffic_condition_mapping = {
    'Congested Traffic': 0,
    'Flow Traffic': 2,
    'Dense Traffic': 1
}

weather_mapping = {
    'windy': 4,
    'cloudy': 0,
    'stormy': 2,
    'sunny': 3,
    'rainy': 1
}

def home(request):
    return render(request, 'home.html')

def predict_fare(request):
    if request.method == 'POST':
        # 1. Retrieve and convert form data
        
        # Map friendly strings to numeric values using dictionaries:
        car_condition = float(car_condition_mapping[request.POST['car_condition']])
        traffic_condition = float(traffic_condition_mapping[request.POST['traffic_condition']])
        weather_encoded = float(weather_mapping[request.POST['weather_encoded']])
        
        # Other features (assumed numeric)
        passenger_count = float(request.POST['passenger_count'])
        hour = float(request.POST['hour'])
        day = float(request.POST['day'])
        month = float(request.POST['month'])
        year = float(request.POST['year'])
        distance = float(request.POST['distance'])
        bearing = float(request.POST['bearing'])
        
        # Coordinates from map (these fields are read-only and set by JavaScript)
        pickup_lat = float(request.POST['pickup_latitude'])
        pickup_lon = float(request.POST['pickup_longitude'])
        dropoff_lat = float(request.POST['dropoff_latitude'])
        dropoff_lon = float(request.POST['dropoff_longitude'])
        
        # 2. Construct the input array in the exact order as during training:
        # Order: ['Car Condition', 'Traffic Condition', 'pickup_longitude', 'pickup_latitude',
        #         'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour', 'day',
        #         'month', 'year', 'distance', 'bearing', 'Weather_encoded']
        X_raw = np.array([[
            car_condition,
            traffic_condition,
            pickup_lon,
            pickup_lat,
            dropoff_lon,
            dropoff_lat,
            passenger_count,
            hour,
            day,
            month,
            year,
            distance,
            bearing,
            weather_encoded
        ]])
        
        # 3. Scale the input features
        X_scaled = X_scaler.transform(X_raw)
        
        # 4. Predict the fare (the model expects the scaled input)
        y_pred_scaled = model.predict(X_scaled)
        
        # 5. Inverse transform the predicted value (if the target was scaled)
        fare_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        fare_pred = fare_pred[0][0]  # Extract the single float value
        
        # 6. Render the result page, rounding the fare prediction for display
        return render(request, 'result.html', {'fare': round(fare_pred, 2)})
    
    # If GET, simply render the prediction form page
    return render(request, 'predict_fare.html')
