from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from database import get_connection

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('comeback_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    score_diff = float(data['score_diff'])
    period = int(data['period'])
    time_seconds = float(data['time_seconds'])
    momentum = float(data['momentum'])
    
    total_time_remaining = ((4 - min(period, 4)) * 720) + time_seconds
    
    features = np.array([[score_diff, total_time_remaining, period, momentum]])
    
    prob = float(model.predict_proba(features)[0][1])

    
    return jsonify({
        'comeback_probability': round(prob * 100, 1),
        'score_diff': score_diff,
        'period': period,
        'time_remaining': time_seconds,
        'momentum': momentum
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
