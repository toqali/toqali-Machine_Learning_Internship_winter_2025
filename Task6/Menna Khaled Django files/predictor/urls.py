from django.urls import path
from .views import home, predict_fare

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict_fare, name='predict_fare'),
]
