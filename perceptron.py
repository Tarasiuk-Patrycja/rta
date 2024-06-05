from flask import Flask, request, jsonify
from sklearn.linear_model import Perceptron
import numpy as np

app = Flask(__name__)

model = Perceptron()
model.fit([[0, 0], [1, 1]], [0, 1])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
