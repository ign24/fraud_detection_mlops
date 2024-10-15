from flask import Flask, request, jsonify
import tensorflow as tf
import logging

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDeploymentPipeline")

# Initialize the Flask application
app = Flask(__name__)

# Load the model
MODEL_PATH = 'models/fraud_detection_model.keras'
logger.info(f"Loading model from {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()
        features = data.get('features')
        if not isinstance(features, list) or len(features) != 30:
            return jsonify({'error': 'Input data is not valid'}), 400
        
        # Make the prediction
        prediction = model.predict([features])
        response = {'prediction': prediction[0][0]}
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

def deploy_model():
    logger.info("Starting the model deployment using Flask")
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    deploy_model()
