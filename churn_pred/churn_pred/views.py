import joblib
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render

# Load the trained model
# model = joblib.load("../models/random_forest_model.pkl")
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Racine du projet
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')

model = joblib.load(MODEL_PATH)

# Load the frequency encoding mapping for 'State'
# state_freq_mapping = joblib.load("../models/state_freq_mapping.pkl")  # Load the precomputed dictionary
state_freq_mapping = joblib.load(os.path.join(BASE_DIR, 'models', 'state_freq_mapping.pkl'))


# Function to convert 'Yes'/'No' to 1/0
def convert_yes_no(value):
    if value.lower() == 'yes':
        return 1
    elif value.lower() == 'no':
        return 0
    else:
        raise ValueError("Value must be 'Yes' or 'No'")

# Home and Prediction page
def home(request):
    if request.method == "POST":
        try:
            # Extract form data
            data = request.POST
            print("Form Data:", data)  # Debugging

            if 'state' not in data:
                raise ValueError("The 'state' field is missing.")

            state_value = data["state"]
            print(f"State Value: {state_value}")  # Debugging

            # Convert 'state' using frequency encoding
            state_index = state_freq_mapping.get(state_value, 0)  # Default to 0 if state not found

            # Convert form data to numerical features
            features = np.array([
                state_index,  # Fixed: Now uses frequency encoding
                convert_yes_no(data['international_plan']),
                convert_yes_no(data['voice_mail_plan']),
                float(data['account_length']),
                float(data['area_code']),
                float(data['number_vmail_messages']),
                float(data['total_day_minutes']),
                float(data['total_day_calls']),
                float(data['total_day_charge']),
                float(data['total_eve_minutes']),
                float(data['total_eve_calls']),
                float(data['total_eve_charge']),
                float(data['total_night_minutes']),
                float(data['total_night_calls']),
                float(data['total_night_charge']),
                float(data['total_intl_minutes']),
                float(data['total_intl_calls']),
                float(data['total_intl_charge']),
                float(data['customer_service_calls']),
            ]).reshape(1, -1)

            # Make prediction
            prediction = model.predict(features)
            result = "Churn" if prediction[0] == 1 else "No Churn"

            return render(request, "index1.html", {"result": result})

        except Exception as e:
            return render(request, "index1.html", {"error": str(e)})

    return render(request, "index1.html")
