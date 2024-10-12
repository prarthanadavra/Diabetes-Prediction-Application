from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, world!"

@app.route("/predict", methods=['POST'])
def predict():
    # Get data from the form and convert it to float
    try:
        Age = int(request.form.get("Age"))
        Glucose = int(request.form.get("Glucose"))
        BloodPressure = int(request.form.get("BloodPressure"))
        Insulin = int(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Ensure all fields are filled with numeric values."})

    # Create input array for the model
    input_query = np.array([[Age, Glucose, BloodPressure, Insulin, BMI]])

    # Make prediction
    result = model.predict(input_query)[0]

    # Return the result as JSON
    return jsonify({'Diabetes': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
