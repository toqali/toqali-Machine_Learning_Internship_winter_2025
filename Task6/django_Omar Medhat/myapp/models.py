from django.db import models


class RideData(models.Model):
    user_id = models.CharField(max_length=50)
    user_name = models.CharField(max_length=100)
    driver_name = models.CharField(max_length=100)
    car_condition = models.CharField(max_length=50)
    weather = models.CharField(max_length=50)
    traffic_condition = models.CharField(max_length=100)
    fare_amount = models.FloatField()
    pickup_datetime = models.DateTimeField()
    pickup_longitude = models.FloatField()
    pickup_latitude = models.FloatField()
    dropoff_longitude = models.FloatField()
    dropoff_latitude = models.FloatField()
    distance = models.FloatField()
    bearing = models.FloatField()

    def __str__(self):
        return f"{self.user_name} - {self.fare_amount}"
