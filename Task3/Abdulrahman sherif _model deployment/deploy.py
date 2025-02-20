from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained models
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
log_reg_model = pickle.load(open('logistic_regression.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Retrieve form inputs
        number_of_adults = int(request.form['number_of_adults'])
        number_of_children = int(request.form['number_of_children'])
        number_of_weekend_nights = int(request.form['number_of_weekend_nights'])
        number_of_week_nights = int(request.form['number_of_week_nights'])
        car_parking_space = int(request.form['car_parking_space'])
        lead_time = int(request.form['lead_time'])
        repeated = int(request.form['repeated'])
        P_C = int(request.form['P_C'])
        P_not_C = int(request.form['P_not_C'])
        average_price = float(request.form['average_price'])
        special_requests = int(request.form['special_requests'])
        reservation_year = int(request.form['reservation_year'])
        reservation_month = int(request.form['reservation_month'])
        reservation_day = int(request.form['reservation_day'])
        reservation_dayofweek = int(request.form['reservation_dayofweek'])

        # Categorical inputs
        meal_plan = request.form['meal_plan']
        room_type = request.form['room_type']
        market_segment = request.form['market_segment']

        # Convert categorical inputs to binary features
        meal_plan_features = [1 if meal_plan == f"Meal Plan {i}" else 0 for i in range(2, 4)] + [1 if meal_plan == "Not Selected" else 0]
        room_type_features = [1 if room_type == f"Room Type {i}" else 0 for i in range(2, 8)]
        market_segment_features = [1 if market_segment == seg else 0 for seg in ["Complementary", "Corporate", "Offline", "Online"]]

        # Create input array
        input_features = np.array([
            number_of_adults, number_of_children, number_of_weekend_nights,
            number_of_week_nights, car_parking_space, lead_time, repeated, P_C, P_not_C,
            average_price, special_requests, reservation_year, reservation_month, reservation_day,
            reservation_dayofweek
        ] + meal_plan_features + room_type_features + market_segment_features).reshape(1, -1)

        # Model selection
        model_choice = request.form['model_choice']

        if model_choice == "KNN":
            prediction = knn_model.predict(input_features)[0]
        else:
            prediction = log_reg_model.predict(input_features)[0]

        prediction = "CANCELED" if prediction == 1 else "NOT CANCELED"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)