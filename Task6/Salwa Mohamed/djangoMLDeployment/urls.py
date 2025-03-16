from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # path('', include('basicConcepts.urls')),
    path('', include('predicted_fare.urls')),
    path('admin/', admin.site.urls),
]
