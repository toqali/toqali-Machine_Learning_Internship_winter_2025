# Fares_Prediction/forms.py

from django import forms

class FareForm(forms.Form):
    user_id = forms.CharField(required=False)
    user_name = forms.CharField(required=False)
    driver_name = forms.CharField(required=False)
    key = forms.CharField(required=False)
    
    pickup_datetime = forms.DateTimeField(required=True)
    pickup_longitude = forms.FloatField(required=True)
    pickup_latitude = forms.FloatField(required=True)
    dropoff_longitude = forms.FloatField(required=True)
    dropoff_latitude = forms.FloatField(required=True)
    
    passenger_count = forms.IntegerField(required=True, min_value=1, max_value=6)
    CAR_CONDITION_CHOICES = [
        ('Excellent', 'Excellent'),
        ('Very Good', 'Very Good'),
        ('Good', 'Good'),
        ('Fair', 'Fair'),
        ('Bad', 'Bad'),
    ]
    
    car_condition = forms.ChoiceField(choices=CAR_CONDITION_CHOICES, required=True)
    weather = forms.CharField(required=True)
    traffic_condition = forms.CharField(required=True)
    
    # Hidden fields calculated by JavaScript
    distance = forms.FloatField(required=False)
    bearing = forms.FloatField(required=False)
    jfk_dist = forms.FloatField(required=False)
    ewr_dist = forms.FloatField(required=False)
    lga_dist = forms.FloatField(required=False)
    sol_dist = forms.FloatField(required=False)
    nyc_dist = forms.FloatField(required=False)
    
    hour = forms.IntegerField(required=False)
    day = forms.IntegerField(required=False)
    month = forms.IntegerField(required=False)
    weekday = forms.IntegerField(required=False)
    year = forms.IntegerField(required=False)