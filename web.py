from flask import Flask, request, render_template
from pycaret.regression import *
import pandas as pd
import numpy as np

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


if __name__ == "__main__":
    app.run()