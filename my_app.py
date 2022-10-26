from crypt import methods
from flask import Flask, request, url_for, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = load_model("model/lr_model")
cols = ["age", "sex", "bmi", "children", "smoker", "region"]

@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = [x for x in request.form.values()]
    input_data_array = np.array(input_data)
    input_data_df = pd.DataFrame([input_data_array], columns=cols)
    prediction = predict_model(model, data=input_data_df, round=0)
    prediction = int(prediction.Label[0])
    return render_template("predict.html", pred=f"Expected Bill will be {prediction}")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])
    prediction = predict_model(model, data=input_data)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)