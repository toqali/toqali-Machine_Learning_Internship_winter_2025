import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the ML model, scaler, PCA and feature names
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received form data:", request.form)  
        form_data = request.form
        user_input = {
            "number of adults": int(form_data["number_of_adults"]),
            "number of children": int(form_data["number_of_children"]),
            "number of weekend nights": int(form_data["number_of_weekend_nights"]),
            "number of week nights": int(form_data["number_of_week_nights"]),
            "type of meal": form_data["type_of_meal"],
            "room type":form_data["room_type"],
            "market segment type":form_data["market_segment_type"],
            "lead time": int(form_data["lead_time"]),
            "average price": int(form_data["average_price"]),
            "special requests": int(form_data["special_request"]),
            "reservation_day": int(form_data["reservation_day"]),
            "reservation_month": int(form_data["reservation_month"]),
            "reservation_year": int(form_data["reservation_year"]),
        }

        # Encode categorical features
        categorical_cols = ["type of meal", "room type", "market segment type"]
        label_encoders = {}
        for col in categorical_cols:
            if col in user_input:
                le = LabelEncoder()
                user_input[col] = le.fit_transform(pd.Series(user_input[col]))
                label_encoders[col] = le

        # Ensure feature order matches training
        correct_order = [col for col in scaler.feature_names_in_ if col != "Booking_ID"]
        ordered_input = {key: user_input[key] for key in correct_order if key in user_input}

        # Ensure test_input_df is properly defined before scaling
        test_input_df = pd.DataFrame([ordered_input])[correct_order]

        # Use the preloaded scaler
        test_input_scaled = scaler.transform(test_input_df)  # Remove unnecessary StandardScaler() definition
        test_input_scaled_df = pd.DataFrame(test_input_scaled, columns=correct_order)

        # PCA for Dimensionality Reduction
        X_pca = pca.transform(test_input_scaled_df)  
        
        # Transform the input data
        test_input_pca = pca.transform(X_pca)[:, :model.n_features_in_]


        # Predict
        prediction = model.predict(test_input_pca)
        result = "Cancelled" if prediction[0] == 1 else "Not Cancelled"

        return render_template('index.html', prediction_text=f'Booking Status: {result}')
    
    except Exception as e:
        print("Error:", str(e))  # Debugging
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)