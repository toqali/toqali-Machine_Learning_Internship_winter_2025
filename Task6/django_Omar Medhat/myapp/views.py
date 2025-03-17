from django.shortcuts import render


def home(request):
    return render(request, "home.html")  # Ensure `home.html` exists in `templates/`


def about(request):
    return render(request, "about.html")  # Make sure this file exists!
