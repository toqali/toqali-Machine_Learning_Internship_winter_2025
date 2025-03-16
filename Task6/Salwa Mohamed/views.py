from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import pickle
import os
import logging
import numpy as np

# إعداد التسجيل
logger = logging.getLogger(__name__)

# تحميل النموذج مرة واحدة
model = None
try:
    model_path = os.path.join(settings.BASE_DIR, 'savedModels', 'best_regression_model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")

# قيم min و max لكل ميزة (حسب مجموعة التدريب)
feature_scaling = {
    'car_condition': (0, 5),
    'weather': (0, 3),
    'traffic_condition': (0, 4),
    'passenger_count': (1, 6),
    'hour': (0, 23),
    'day': (1, 31),
    'month': (1, 12),
    'weekday': (0, 6),
    'year': (2020, 2023),
    'jfk_dist': (0, 50),
    'ewr_dist': (0, 50),
    'lga_dist': (0, 50),
    'sol_dist': (0, 20),
    'nyc_dist': (0, 50),
    'distance': (0, 100),
    'bearing': (0, 360)
}

# تطبيع المدخلات باستخدام Min-Max
def normalize_input(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

def predict_fare(request):
    predicted_fare = None

    if request.method == 'POST':
        if model is None:
            return HttpResponse("Model not loaded. Please check the logs.", status=500)

        try:
            # استخراج الميزات وتطبيعها
            input_features = []
            for feature, (min_val, max_val) in feature_scaling.items():
                value = request.POST.get(feature)
                try:
                    value = float(value)
                    normalized_value = normalize_input(value, min_val, max_val)
                    input_features.append(normalized_value)
                except (ValueError, TypeError):
                    return HttpResponse(f"Invalid value for {feature}: {value}", status=400)

            logger.info(f"📊 Normalized Input: {input_features}")

            # تحقق من تطابق الميزات
            if len(input_features) != model.n_features_in_:
                return HttpResponse(f"Error: Expected {model.n_features_in_} features, got {len(input_features)}", status=400)

            # تنفيذ التنبؤ
            predicted_fare = model.predict([input_features])[0]
            logger.info(f"🚀 Predicted fare: {predicted_fare}")

            return render(request, 'main.html', {'fare': round(predicted_fare, 2)})

        except Exception as e:
            logger.exception("❌ Prediction error")
            return HttpResponse(f"Prediction error: {e}", status=500)

    return render(request, 'main.html', {'fare': predicted_fare})
