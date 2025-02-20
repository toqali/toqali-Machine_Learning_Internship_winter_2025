from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle  # To load the model
import numpy as np

app = Flask(__name__)

# Load the machine learning models
with open("random_forest.pkl", "rb") as f:
    model1 = pickle.load(f)
with open("logistic_regression.pkl", "rb") as f:
    model2 = pickle.load(f)
with open("xgboost.pkl", "rb") as f:
    model3 = pickle.load(f)

# Load the CSV data (for dashboard)
df = pd.read_csv("D:\\Downloads\\Hotel Project\\first_inten_project.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    # Compute summary statistics for the dashboard
    total_bookings = len(df)
    cancelled = df[df['booking status'] == "Canceled"].shape[0]
    not_cancelled = df[df['booking status'] == "Not_Canceled"].shape[0]
    
    # Count room types for visualization
    room_counts = df['room type'].value_counts().to_dict()

    prediction_result = None  # Default: No prediction
    
    if request.method == "POST":
        try:
            # Get form data
            number_of_adults = int(request.form["number_of_adults"])
            number_of_children = int(request.form["number_of_children"])
            number_of_weekend_nights = int(request.form["number_of_weekend_nights"])
            number_of_week_nights = int(request.form["number_of_week_nights"])
            lead_time = float(request.form["lead_time"])
            total_price = float(request.form["total_price"])

            # Convert to NumPy array (reshape for sklearn models)
            input_data = np.array([
                number_of_adults,
                number_of_children,
                number_of_weekend_nights,
                number_of_week_nights,
                lead_time,
                total_price
            ]).reshape(1, -1)

            # Make predictions using each model
            pred1 = model1.predict(input_data)[0]
            pred2 = model2.predict(input_data)[0]
            pred3 = model3.predict(input_data)[0]

            # Majority vote (if at least 2 models agree)
            final_prediction = "Will Come" if (pred1 + pred2 + pred3) >= 2 else "Will Not Come"

            prediction_result = final_prediction

        except Exception as e:
            prediction_result = f"Error: {str(e)}"

    return render_template("index.html", 
                           total_bookings=total_bookings,
                           cancelled=cancelled,
                           not_cancelled=not_cancelled,
                           room_counts=room_counts,
                           prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
