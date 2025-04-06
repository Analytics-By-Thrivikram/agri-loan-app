import numpy as np
from flask import Flask, request, render_template
import joblib

flask_app = Flask(__name__)

# Load the saved model with error handling
try:
    model = joblib.load("random_forest_model.pkl")
except FileNotFoundError:
    model = None


@flask_app.route('/')
def home():
    return render_template('index.html', prediction_text='')


@flask_app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Error: Model file not found!")

    try:
        form_data = request.form.to_dict()

        # Define crop and loan purpose mappings
        crop_mapping = {"Wheat": 0, "Groundnut": 1, "Maize": 2, "Rice": 3, "Soybean": 4, "Sugarcane": 5}
        loan_purpose_mapping = {"Debt Consolidation": 0, "Farm Equipment": 1, "Irrigation": 2, "Seeds & Fertilizers": 3}

        # Initialize an array for feature values
        float_features = []

        # Crop encoding - Set other crops to 0 except for the selected one
        crop_type = form_data.get("Type of Crop(s)")
        crop_features = [1 if crop_mapping.get(crop) == int(crop_type) else 0 for crop in crop_mapping.keys()]
        float_features.extend(crop_features)

        # Loan purpose encoding - Set other loan purposes to 0 except for the selected one
        loan_purpose = form_data.get("Loan Purpose")
        loan_features = [1 if loan_purpose_mapping.get(loan) == int(loan_purpose) else 0 for loan in loan_purpose_mapping.keys()]
        float_features.extend(loan_features)

        # Add numerical input features
        for key, value in form_data.items():
            if key not in ["Type of Crop(s)", "Loan Purpose"]:
                try:
                    float_features.append(float(value))
                except ValueError:
                    return render_template("index.html", prediction_text=f"Error: Invalid input for {key}")

        features = np.array(float_features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        # Map prediction values
        prediction_labels = {
            1: "Not Eligible",
            2: "Eligible"
        }

        # Get prediction label
        prediction_result = prediction_labels.get(prediction[0], "Unknown Prediction")

        return render_template("index.html", prediction_text=f"The loan eligibility prediction is: {prediction_result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    flask_app.run(debug=True)
