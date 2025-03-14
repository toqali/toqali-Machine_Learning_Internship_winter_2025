from django.urls import path
from . import views

urlpatterns = [
    # User Authentication URLs
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('bookings/', views.booking_list, name='booking_list'),
    path('booking/<str:booking_id>/', views.booking_detail, name='booking_detail'),
]
