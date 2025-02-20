from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)  # Init. Flask app
model = pickle.load(open("hotel_booking_model.pkl", "rb"))  # Load model

def preprocess_input(form_data):
    """Preprocess input data to match the trained model."""
    try:
        
        date_of_reservation = pd.to_datetime(form_data.get("date of reservation"))  # date of reservation conversion
        date_ordinal = date_of_reservation.toordinal()
        
        # Extract needed features for the model
        input_data = {
            "lead time": int(form_data.get("lead time")),
            "repeated": int(form_data.get("repeated")),
            "special requests": int(form_data.get("special requests")),
            "average price ": float(form_data.get("average price ")),
            "P-not-C": int(form_data.get("P-not-C")),
            "date of reservation": date_ordinal,
            "market segment type": form_data.get("market segment type")
        }
        # Encode categorical features
        market_segment_map = {"Offline": 0, "Online": 1, "Corporate": 2, "Aviation": 3, "Complementary": 4}
        input_data["market segment type"] = market_segment_map.get(input_data["market segment type"], -1)  # Default to -1 if unknown
        
        return np.array([list(input_data.values())])
    
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        final_features = preprocess_input(request.form)
        prediction = model.predict(final_features)
        prediction_label = "Canceled" if prediction[0] == 1 else "Not_Canceled"   #Encode target
        return render_template('index.html', prediction_text=f"Booking Status might be {prediction_label}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':  # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
