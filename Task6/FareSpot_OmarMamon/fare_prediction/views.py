from django.shortcuts import render
import pandas as pd
import pickle
import os
import numpy as np
import math
from datetime import datetime
import uuid

def home(request):  
  return render(request, 'home.html')
  
# Load your trained model and scaler
def load_model():
  # Adjust this path to match your project structure
  model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fare_prediction', 'models', 'lgb_model.pkl')
  scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fare_prediction', 'models', 'scaler.pkl')

  try:
      with open(model_path, 'rb') as file:
          model = pickle.load(file)
  except FileNotFoundError:
      # Provide alternate path if the first one doesn't work
      alt_model_path = os.path.join(os.path.dirname(__file__), 'models', 'lgb_model.pkl')
      with open(alt_model_path, 'rb') as file:
          model = pickle.load(file)

  try:
      with open(scaler_path, 'rb') as file:
          scaler = pickle.load(file)
  except FileNotFoundError:
      # Provide alternate path if the first one doesn't work
      alt_scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
      with open(alt_scaler_path, 'rb') as file:
          scaler = pickle.load(file)

  return model, scaler

# Helper function to safely convert to numeric values
def safe_float(value, default=0.0):
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    try:
        if value is None or value == '':
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

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
          'ride_id': request.POST.get('ride_id', str(uuid.uuid4())[:8].upper()),
          'pickup_lat': request.POST.get('pickup_lat', '0'),
          'pickup_lon': request.POST.get('pickup_lon', '0'),
          'dropoff_lat': request.POST.get('dropoff_lat', '0'),
          'dropoff_lon': request.POST.get('dropoff_lon', '0'),
          'ride_distance': request.POST.get('ride_distance', '0'),
          'ride_bearing': request.POST.get('ride_bearing', '0'),
          'weather': request.POST.get('weather', 'sunny'),
          'traffic': request.POST.get('traffic', 'moderate'),
          'car_condition': request.POST.get('car_condition', 'good'),
          'pickup_time': request.POST.get('pickup_time', ''),
          'distance_jfk': request.POST.get('distance_jfk', '0'),
          'distance_ewr': request.POST.get('distance_ewr', '0'),
          'distance_lga': request.POST.get('distance_lga', '0'),
          'distance_sol': request.POST.get('distance_sol', '0'),
          'distance_nyc': request.POST.get('distance_nyc', '0'),
      }
      
      try:
          # Get basic form data with safe conversion
          # 1. Process passenger count
          passengers = safe_int(request.POST.get('passenger_count'), 1)
          
          # 2. Process ride distance
          ride_distance = safe_float(request.POST.get('ride_distance'), 0.0)
          
          # 3. Process ride bearing
          ride_bearing = safe_float(request.POST.get('ride_bearing'), 0.0)
          
          # 4. Process date and time information - Handle empty values with defaults
          current_date = datetime.now()
          ride_year = safe_int(request.POST.get('request_year'), current_date.year)
          ride_month = safe_int(request.POST.get('request_month'), current_date.month)
          ride_day = safe_int(request.POST.get('request_day'), current_date.day)
          request_hour = safe_int(request.POST.get('request_hour'), current_date.hour)
          
          # Generate a datetime object
          ride_datetime = datetime(
              year=ride_year, 
              month=ride_month, 
              day=ride_day, 
              hour=request_hour
          )
          
          # Extract minute from pickup time
          pickup_time = request.POST.get('pickup_time', '00:00')
          if ':' in pickup_time:
              pickup_hour = safe_int(pickup_time.split(':')[0], request_hour)
              pickup_minute = safe_int(pickup_time.split(':')[1], 0)
          else:
              pickup_hour = request_hour
              pickup_minute = 0
          
          # 5. Process weekday and create cyclical encoding
          request_weekday = safe_int(request.POST.get('request_weekday'), current_date.weekday())
          weekday_sin = np.sin(2 * np.pi * request_weekday / 7)
          weekday_cos = np.cos(2 * np.pi * request_weekday / 7)
          
          # 6. Determine time of day and create cyclical encoding
          hour_of_day = request_hour
          # Map 24 hours to an angle (0 to 2π)
          time_angle = 2 * np.pi * hour_of_day / 24
          # Use the exact feature names that were used during training
          time_sin = np.sin(time_angle)
          time_cos = np.cos(time_angle)
          
          # 7. Process is_rush_hour
          # Determine if it's rush hour based on time
          morning_rush = (hour_of_day >= 7 and hour_of_day <= 10)
          evening_rush = (hour_of_day >= 16 and hour_of_day <= 19)
          is_rush_hour = 1 if (morning_rush or evening_rush) else 0
          
          # 8. Process car condition
          car_condition_raw = request.POST.get('car_condition', '').lower()
          car_condition_map = {'poor': 0, 'fair': 1, 'good': 2, 'excellent': 3}
          car_condition_encoded = car_condition_map.get(car_condition_raw, 2)  # Default to 2 (good) if not found
          
          # 9. Process weather conditions (one-hot encoded)
          weather_condition = request.POST.get('weather', 'sunny').lower()
          weather_cloudy = 1 if weather_condition == 'cloudy' else 0
          weather_rainy = 1 if weather_condition == 'rainy' else 0
          weather_stormy = 1 if weather_condition == 'stormy' else 0
          weather_sunny = 1 if weather_condition == 'sunny' else 0
          weather_windy = 1 if weather_condition == 'windy' else 0
          
          # 10. Process traffic conditions
          traffic_raw = request.POST.get('traffic', '').lower()
          traffic_map = {'heavy': 0, 'moderate': 1, 'light': 2}
          traffic_encoded = traffic_map.get(traffic_raw, 1)  # Default to 1 (moderate) if not found
          
          # 11. Create distance_bin_encoded
          # Define the distance bins
          distance_bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, float('inf'))]
          distance_bin_encoded = 0  # Default to 0
          
          for i, (low, high) in enumerate(distance_bins):
              if low <= ride_distance < high:
                  distance_bin_encoded = i
                  break
          
          # 12. Calculate direction encoding from ride_bearing
          # Convert bearing to compass direction and then to cyclical features
          direction_sin = np.sin(ride_bearing * np.pi / 180)
          direction_cos = np.cos(ride_bearing * np.pi / 180)
          
          # 13. Process distance measurements with safe conversion
          distance_jfk = safe_float(request.POST.get('distance_jfk'), 0.0)
          distance_ewr = safe_float(request.POST.get('distance_ewr'), 0.0)
          distance_lga = safe_float(request.POST.get('distance_lga'), 0.0)
          distance_sol = safe_float(request.POST.get('distance_sol'), 0.0)
          distance_nyc = safe_float(request.POST.get('distance_nyc'), 0.0)
          
          # 14. Prepare features in the expected format for the model
          features = {
              'passengers': [passengers],
              'ride_distance': [ride_distance],
              'ride_bearing': [ride_bearing],
              'pickup_minute': [pickup_minute],
              'is_rush_hour': [is_rush_hour],
              'car_condition_encoded': [car_condition_encoded],
              'weather_rainy': [weather_rainy],
              'weather_stormy': [weather_stormy],
              'weather_sunny': [weather_sunny],
              'weather_windy': [weather_windy],
              'traffic_encoded': [traffic_encoded],
              'distance_bin_encoded': [distance_bin_encoded],
              'time_sin': [time_sin],
              'time_cos': [time_cos],
              'direction_sin': [direction_sin],
              'direction_cos': [direction_cos],
              'weekday_sin': [weekday_sin],
              'weekday_cos': [weekday_cos]
          }
          
          # Convert to DataFrame
          input_df = pd.DataFrame(features)
          
          # Load model and scaler
          model, scaler = load_model()
          
          # Get the feature names from the scaler
          try:
              # Try to get feature names from scaler
              scaler_feature_names = scaler.feature_names_in_
              print(f"Scaler feature names: {scaler_feature_names}")
              
              # Ensure all required columns exist with the correct names
              for col in scaler_feature_names:
                  if col not in input_df.columns:
                      # If a column is missing, try to find a similar one
                      if col == 'time_of_day_sin' and 'time_sin' in input_df.columns:
                          input_df['time_of_day_sin'] = input_df['time_sin']
                      elif col == 'time_of_day_cos' and 'time_cos' in input_df.columns:
                          input_df['time_of_day_cos'] = input_df['time_cos']
                      else:
                          input_df[col] = 0  # Default value
              
              # Select only the expected columns in the correct order
              input_df = input_df[scaler_feature_names]
              
          except AttributeError:
              # If scaler doesn't have feature_names_in_, use the expected columns
              expected_columns = [
                  'passengers', 'ride_distance', 'ride_bearing', 'pickup_minute', 
                  'is_rush_hour', 'car_condition_encoded', 'weather_rainy', 
                  'weather_stormy', 'weather_sunny', 'weather_windy', 
                  'traffic_encoded', 'distance_bin_encoded', 'time_of_day_sin',
                  'time_of_day_cos', 'direction_sin', 'direction_cos', 'weekday_sin', 
                  'weekday_cos'
              ]
              
              # Ensure all required columns exist with the correct names
              for col in expected_columns:
                  if col not in input_df.columns:
                      # If a column is missing, try to find a similar one
                      if col == 'time_of_day_sin' and 'time_sin' in input_df.columns:
                          input_df['time_of_day_sin'] = input_df['time_sin']
                      elif col == 'time_of_day_cos' and 'time_cos' in input_df.columns:
                          input_df['time_of_day_cos'] = input_df['time_cos']
                      else:
                          input_df[col] = 0  # Default value
              
              # Select only the expected columns in the correct order
              input_df = input_df[expected_columns]
          
          # Scale the features
          input_scaled = scaler.transform(input_df)
          
          # Make prediction
          prediction = model.predict(input_scaled)
          prediction = np.exp(prediction)  # Assuming the model predicts log fare
          
          # Round to two decimal places for fare amount
          predicted_fare = round(float(prediction[0]), 2)
          
          # Add prediction to form data for rendering
          form_data['predicted_fare'] = predicted_fare
          
          # Add additional information for display
          form_data['is_rush_hour'] = "Yes" if is_rush_hour else "No"
          
          # Get month name
          month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
          month_name = month_names[ride_month - 1]
          
          form_data['date'] = f"{month_name} {ride_day}, {ride_year}"
          form_data['time'] = f"{request_hour}:{pickup_minute:02d}"
          
          # Debug line to see what's happening
          print(f"Prediction successful: ${predicted_fare}")
          print(f"Input features: {input_df.to_dict()}")
          
          return render(request, 'result.html', {'result': form_data})
          
      except Exception as e:
          import traceback
          error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
          print(f"Error in prediction: {error_message}")  # Debug line
          return render(request, 'predict.html', {'error': error_message, 'form_data': form_data})
  
  return render(request, 'predict.html')