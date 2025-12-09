from flask import Flask, request, jsonify
from flask_cors import CORS
from services.modelService import ModelService

app = Flask(__name__)
CORS(app)
model_service = ModelService()

@app.route('/api/models/<model_name>/predict', methods=['POST'])
def predict(model_name):
    try:
        import numpy as np
        data = request.json
        input_data = np.array(data['input'])
        result = model_service.predict(model_name, input_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
