import pandas as pd
import joblib
from flask import Flask, request, jsonify

model = joblib.load("app/model.joblib")

app = Flask(__name__)

def predict_one(is_student, balance, income):
    """Returns a prediction of wether a person will default."""
    
    predict_df = pd.DataFrame({
        "student": [is_student],
        "balance": [balance],
        "income": [income],
    })
    
    prediction = model.predict(predict_df)[0]
    
    return prediction

@app.route("/")
def home():
    return "Hello World"

@app.route("/predict", methods=["POST"])
def predict_route():

    request_dict = request.get_json()

    is_student = request_dict["is_student"]
    balance = float(request_dict["balance"])
    income = float(request_dict["income"])

    default = predict_one(is_student, balance, income)

    response = {
        "default": bool(default)
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)