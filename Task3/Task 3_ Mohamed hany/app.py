import joblib
from flask import Flask, request, render_template, jsonify
import pandas as pd

# Load the trained model, scaler, label encoders, and feature names
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')  # Load feature names

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    print("Home route is called")
    return render_template('Index.html')  # Render HTML interface

@app.route('/predict', methods=['POST'])
def predict():
    print("Received POST request at /predict")
    try:
        # Retrieve form data from the user
        form_data = request.form  # Use request.form for form data
        print("Form Data:", form_data)  # Debugging line to print received data

        # Create a dictionary with all features (initialize missing features to 0)
        user_input = {feature: 0 for feature in feature_names}  # Initialize all features to 0
        user_input.update({
            "type of meal": form_data["type_of_meal"],
            "room type": form_data["room_type"],
            "market segment type": form_data["market_segment_type"],
            "average price": float(form_data["average_price"]),  # Convert to float
            "lead time": float(form_data["lead_time"])  # Convert to float
        })

        # Encode categorical features using the pre-loaded label encoders
        for col in ["type of meal", "room type", "market segment type"]:
            if user_input[col] not in label_encoders[col].classes_:
                return jsonify({"error": f"Invalid value for {col}: {user_input[col]}"}), 400
            user_input[col] = label_encoders[col].transform([user_input[col]])[0]

        # Convert user input to a DataFrame for scaling
        input_df = pd.DataFrame([user_input])

        # Ensure the columns are in the correct order
        input_df = input_df[feature_names]

        # Scale the data using the pre-loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make the prediction using the pre-loaded model
        prediction = model.predict(scaled_input)

        # Return the prediction result
        result = "Cancelled" if prediction[0] == 1 else "Not Cancelled"
        return jsonify({"prediction": result})
    
    except Exception as e:
        # If there’s an error, display the error message
        print("Error:", str(e))  # Debugging line
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5002)  # Disable reloader to avoid restarting issues