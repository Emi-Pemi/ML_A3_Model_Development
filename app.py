from flask import Flask, request, jsonify, render_template
from inference import HousingPredictor
import os

app = Flask(__name__)
predictor = HousingPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                         'Population', 'AveOccup', 'Latitude', 'Longitude']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'status': 'error'}), 400
        
        # Convert and validate
        features = [
            float(data.get('MedInc', 0)),
            float(data.get('HouseAge', 0)),
            float(data.get('AveRooms', 0)),
            float(data.get('AveBedrms', 0)),
            float(data.get('Population', 0)),
            float(data.get('AveOccup', 0)),
            float(data.get('Latitude', 0)),
            float(data.get('Longitude', 0))
        ]
        
        prediction = predictor.predict(features)
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': f"Prediction failed: {str(e)}",
            'status': 'error'
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)