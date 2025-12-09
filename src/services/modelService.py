import tensorflow as tf
import numpy as np
from typing import Dict, Any

class ModelService:
    def __init__(self):
        self.models = {}
        self.model_versions = {}
    
    def load_model(self, model_name: str, version: str, model_path: str):
        """Load a machine learning model"""
        try:
            model = tf.keras.models.load_model(model_path)
            key = f"{model_name}:{version}"
            self.models[key] = model
            self.model_versions[model_name] = version
            return {"status": "success", "message": f"Model {model_name} v{version} loaded"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict(self, model_name: str, data: np.ndarray) -> Dict[str, Any]:
        """Make predictions using a loaded model"""
        version = self.model_versions.get(model_name)
        if not version:
            return {"error": f"Model {model_name} not loaded"}
        
        key = f"{model_name}:{version}"
        model = self.models.get(key)
        if not model:
            return {"error": f"Model {model_name} not found"}
        
        try:
            predictions = model.predict(data)
            return {
                "success": True,
                "predictions": predictions.tolist(),
                "model": model_name,
                "version": version
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model"""
        version = self.model_versions.get(model_name)
        if not version:
            return {"error": f"Model {model_name} not loaded"}
        
        key = f"{model_name}:{version}"
        model = self.models.get(key)
        
        return {
            "name": model_name,
            "version": version,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "parameters": model.count_params()
        }
