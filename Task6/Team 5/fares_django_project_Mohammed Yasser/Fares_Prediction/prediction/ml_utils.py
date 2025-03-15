# Fares_Prediction/ml_utils.py

import pickle
import os
import numpy as np
import pandas as pd
from django.conf import settings

from sklearn.preprocessing import LabelEncoder


class TaxiFarePredictor:
    def __init__(self):
        # Paths to model files
        model_path = os.path.join(settings.BASE_DIR, 'Fares_Prediction', 'ml_models', 'model.pkl')
        encoder_path = os.path.join(settings.BASE_DIR, 'Fares_Prediction', 'ml_models', 'encoder.pkl')
        scaler_path = os.path.join(settings.BASE_DIR, 'Fares_Prediction', 'ml_models', 'scaler.pkl')
        
        # Load model, encoder, and scaler
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
            
        with open(encoder_path, 'rb') as encoder_file:
            self.label_encoder = pickle.load(encoder_file)
            
        with open(scaler_path, 'rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)

    
    def preprocess_input(self, data):
        """
        Preprocess the input data for prediction
        """
        # Create a DataFrame from the input data
        df = pd.DataFrame([data])
        # print(df.iloc[0])

        non_numeric_cols = ['user_id', 'user_name', 'driver_name', 'key', 'pickup_datetime']
        df = df.drop(columns=[col for col in non_numeric_cols if col in df.columns])

        def bucket_passenger_count(count):
            if count == 1:
                return 'Single'
            elif count == 2:
                return 'Pair'
            elif count > 2:
                return 'Group'
            else:
                print (count)
                return 'Unknown'

        # Apply the function to create a new column
        df['passenger_bucket'] = df['passenger_count'].apply(bucket_passenger_count)
                
        self.car_condition_columns = [
            'Car Condition_Excellent', 
            'Car Condition_Very Good', 
            'Car Condition_Good', 
            'Car Condition_Bad'
        ]
        if 'car_condition' in df.columns:
            # Initialize columns with zeros
            for col in self.car_condition_columns:
                df[col] = 0
                
            # Set the appropriate column to 1 based on the car_condition value
            for i, condition in enumerate(df['car_condition']):
                col_name = f'Car Condition_{condition}'
                if col_name in self.car_condition_columns:
                    df.at[i, col_name] = 1
            
            # Drop the original column
            df = df.drop(columns=['car_condition'])
        
            


        df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                3: "Spring", 4: "Spring", 5: "Spring",
                                6: "Summer", 7: "Summer", 8: "Summer",
                                9: "Fall", 10: "Fall", 11: "Fall"})
        

        categorical_features = ['hour', 'day', 'month', 'weekday', 'year', 'season', 'passenger_bucket']
        print(df['month'])
        for feature in categorical_features:
            # print(df[feature])
            df[feature] = self.label_encoder[feature].transform(df[feature])
        # print('month', df['month'])

        df = df.drop(columns=[
            'csrfmiddlewaretoken', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'weather', 'traffic_condition', 'passenger_count', 'bearing',
        ])
        # print(df.columns)

        # skewed features
        skewed_features = ['jfk_dist', 'lga_dist', 'ewr_dist', 'nyc_dist', 'sol_dist', 'distance']
        for feature in skewed_features:
            df[feature] = np.log1p(df[feature])

                # Scale numerical features
        numerical_features = ['distance', 'jfk_dist', 'lga_dist', 'ewr_dist', 'nyc_dist', 'sol_dist']
        # for feature in numerical_features:
        #     if feature in df.columns:
        df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        # Reorder columns to match the order expected by the model
        expected_columns = ['hour', 'day', 'month', 'weekday', 'year', 'jfk_dist', 'ewr_dist', 'lga_dist',
                            'sol_dist', 'nyc_dist', 'distance', 'passenger_bucket', 'season',
                            'Car Condition_Bad', 'Car Condition_Excellent', 'Car Condition_Good',
                            'Car Condition_Very Good']
        df = df[expected_columns]
        # print(df.iloc[0])
        return df
    
    def predict(self, data):
        """
        Make a prediction using the trained model
        """
        # Preprocess the data
        preprocessed_data = self.preprocess_input(data)
        print(preprocessed_data.iloc[0])
        # Make prediction
        prediction = self.model.predict(preprocessed_data)[0]
        
        #inverse log transformation of the prediction
        prediction = np.expm1(prediction)
        
        return prediction