from django.contrib import admin
from django.urls import path, include  # Use `include` to link to `myapp.urls`

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("myapp.urls")),  # This should point to `myapp.urls`
]
