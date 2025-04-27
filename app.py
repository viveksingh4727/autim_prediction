from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS for all domains (for now, can restrict later)
CORS(app)

# Load the saved model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)



# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure the request is JSON
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()

        # Extract features properly
        features = np.array([[
            data.get('A1_Score', 0),
            data.get('A2_Score', 0),
            data.get('A3_Score', 0),
            data.get('A4_Score', 0),
            data.get('A5_Score', 0),
            data.get('A6_Score', 0),
            data.get('A7_Score', 0),
            data.get('A8_Score', 0),
            data.get('A9_Score', 0),
            data.get('A10_Score', 0),
            data.get('age', 0),
            data.get('gender', 0),
            data.get('ethnicity', 0),
            data.get('jaundice', 0),
            data.get('austim', 0),
            data.get('contry_of_res', 0),
            data.get('used_app_before', 0),
            data.get('result', 0),
            data.get('relation', 0)
        ]])

        # Predict using the model
        prediction = model.predict(features)

        result = "Autism Detected" if prediction[0] == 1 else "No Autism"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
