from django.urls import path
from .views import predict_fare

urlpatterns = [
    path('', predict_fare, name='predict_fare'),
]