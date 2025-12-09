from flask import Flask, request, jsonify
from services.modelService import ModelService

app = Flask(__name__)
model_service = ModelService()

@app.route('/api/models/<model_name>/predict', methods=['POST'])
def predict(model_name):
    try:
        data = request.json
        input_data = np.array(data['input'])
        result = model_service.predict(model_name, input_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/models/<model_name>/info', methods=['GET'])
def model_info(model_name):
    info = model_service.get_model_info(model_name)
    return jsonify(info)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "ML Model Server"})
