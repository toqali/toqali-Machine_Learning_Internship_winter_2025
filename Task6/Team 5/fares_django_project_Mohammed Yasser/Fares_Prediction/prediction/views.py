# Fares_Prediction/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
from datetime import datetime
from .forms import FareForm
from .ml_utils import TaxiFarePredictor

# Initialize the model
fare_predictor = TaxiFarePredictor()

def index(request):
    """Render the main page with the fare prediction form"""
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    """Handle the fare prediction request"""
    if request.method == 'POST':
        # try:
            # Parse the form data
            data = {}
            # print(request.POST)
            for key, value in request.POST.items():
                # Convert values to appropriate types
                if key in ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
                           'distance', 'bearing', 'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist']:
                    data[key] = float(value) if value else 0.0
                elif key in ['passenger_count', 'hour', 'day', 'month', 'weekday', 'year']:
                    data[key] = int(value) if value else 0
                else:
                    data[key] = value
            
            # Validate pickup_datetime
            if 'pickup_datetime' in data:
                data['pickup_datetime'] = datetime.fromisoformat(data['pickup_datetime'].replace('Z', '+00:00'))
            # Make prediction
            predicted_fare = fare_predictor.predict(data)
            
            # Return the prediction as JSON
            return JsonResponse({
                'status': 'success',
                'fare': round(float(predicted_fare), 2)
            })
        # except Exception as e:
        #     return JsonResponse({
        #         'status': 'error',
        #         'message': str(e)
        #     }, status=400)
    else:
        return JsonResponse({
            'status': 'error',
            'message': 'Method not allowed'
        }, status=405)