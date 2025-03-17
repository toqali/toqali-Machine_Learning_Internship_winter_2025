from django.shortcuts import render
from .models import RideData


def home(request):
    data = RideData.objects.all()

    # Get filter parameters from request
    driver_name = request.GET.get("driver_name")
    weather = request.GET.get("weather")
    traffic_condition = request.GET.get("traffic")
    min_fare = request.GET.get("min_fare")
    max_fare = request.GET.get("max_fare")

    # Apply filters
    if driver_name:
        data = data.filter(driver_name__icontains=driver_name)
    if weather:
        data = data.filter(weather__icontains=weather)
    if traffic_condition:
        data = data.filter(traffic_condition__icontains=traffic_condition)
    if min_fare:
        data = data.filter(fare_amount__gte=min_fare)
    if max_fare:
        data = data.filter(fare_amount__lte=max_fare)

    return render(request, "home.html", {"data": data})
