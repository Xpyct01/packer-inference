from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
import joblib
import json


MODEL_PATH = "model.gz"
app = Flask(__name__)
metrics = PrometheusMetrics(app)


def get_predictions(input_data):
    model = joblib.load(MODEL_PATH)
    result = model.predict([input_data])[0]
    return {"result": result}


@app.route("/predict", methods=['POST'])
def predict():
    request_data = request.get_json()
    input_data = request_data['input_data']
    return get_predictions(input_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
