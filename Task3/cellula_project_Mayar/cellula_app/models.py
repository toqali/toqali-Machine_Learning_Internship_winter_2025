from django.db import models

class Booking(models.Model):
    booking_id = models.CharField(max_length=100, unique=True)
    number_of_adults = models.IntegerField()
    number_of_children = models.IntegerField()
    number_of_weekend_nights = models.IntegerField()
    number_of_week_nights = models.IntegerField()
    type_of_meal = models.CharField(max_length=100)
    car_parking_space = models.BooleanField()
    room_type = models.CharField(max_length=100)
    lead_time = models.IntegerField()
    market_segment_type = models.CharField(max_length=100)
    repeated = models.BooleanField()
    p_c = models.IntegerField()  # P-C
    p_not_c = models.IntegerField()  # P-not-C
    average_price = models.FloatField()
    special_requests = models.IntegerField()
    date_of_reservation = models.DateField(null=True, blank=True)
    booking_status = models.CharField(max_length=50)

    def __str__(self):
        return f'Booking {self.booking_id} - {self.booking_status}'
