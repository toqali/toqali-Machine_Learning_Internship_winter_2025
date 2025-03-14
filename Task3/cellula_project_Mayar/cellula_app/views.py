from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Booking


# ===========================
# User Authentication Views
# ===========================

def home(request):
    """Home Page"""
    return render(request, 'cellula_app/home.html')


def register(request):
    """User Registration"""
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
        else:
            User.objects.create_user(username=username, password=password)
            messages.success(request, 'Account created successfully! Please log in.')
            return redirect('login')  # Redirect to login after registration
    return render(request, 'cellula_app/register.html')


def user_login(request):
    """User Login"""
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('admin_dashboard')  # Redirect to the dashboard
        else:
            messages.error(request, 'Invalid credentials')
    return render(request, 'cellula_app/login.html')


def user_logout(request):
    """User Logout"""
    logout(request)
    return redirect('home')


# ===========================
# Booking Management Views
# ===========================

@login_required
def booking_list(request):
    """List of Bookings (With Search)"""
    query = request.GET.get('q')
    if query:
        bookings = Booking.objects.filter(booking_id__icontains=query)
    else:
        bookings = Booking.objects.all()[:100]  # Show only 100 for faster loading

    return render(request, 'cellula_app/booking_list.html', {'bookings': bookings, 'query': query})


# ===========================
# Dashboard Views (For All Users)
# ===========================

@login_required
def admin_dashboard(request):
    """Dashboard — View all bookings with search and filters (For All Users)"""
    query = request.GET.get('q')
    filter_by = request.GET.get('filter_by')

    # Apply search and filters
    bookings = Booking.objects.all()
    if query:
        if filter_by == 'booking_id':
            bookings = bookings.filter(booking_id__icontains=query)
        elif filter_by == 'room_type':
            bookings = bookings.filter(room_type__icontains=query)
        elif filter_by == 'booking_status':
            bookings = bookings.filter(booking_status__icontains=query)

    bookings = bookings[:100]  # Limit results for performance

    return render(request, 'cellula_app/admin_dashboard.html', {'bookings': bookings})


@login_required
def booking_detail(request, booking_id):
    """Booking Detail View (For All Users)"""
    booking = get_object_or_404(Booking, booking_id=booking_id)
    return render(request, 'cellula_app/booking_detail.html', {'booking': booking})
