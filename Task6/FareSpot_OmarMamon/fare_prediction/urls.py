from django.urls import path
from . import views
from .views import predict


urlpatterns = [
    path('', views.home, name='home'),
    path("predict/", predict, name="predict"),  # Ensure this exists,
]