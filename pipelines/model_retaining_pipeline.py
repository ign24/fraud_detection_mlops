import tensorflow as tf
import tensorflow_privacy
import mlflow
import mlflow.tensorflow
import logging
import os
from sklearn.model_selection import train_test_split
import pandas as pd

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelRetrainingPipeline")

# Configuration parameters
DATA_PATH = 'data/new_transactions.csv'
MODEL_OUTPUT_PATH = 'models/fraud_detection_model_retrained.keras'
EXPERIMENT_NAME = "Fraud_Detection_Retraining"
MODEL_NAME = "fraud_detection_model"
EPOCHS = 5
BATCH_SIZE = 16

# DP-SGD parameters
LEARNING_RATE = 0.01
NOISE_MULTIPLIER = 1.1
L2_NORM_CLIP = 1.0

# Load the data
def load_data(path):
    logger.info(f"Loading data from {path}")
    data = pd.read_csv(path)
    X = data.drop(columns=['Class'])
    y = data['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
def build_model():
    logger.info("Building the fraud detection model with Differential Privacy")
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(30,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Retraining function
def retrain_model():
    try:
        # Start MLflow tracking
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.tensorflow.autolog(log_models=False)

        with mlflow.start_run() as run:
            # Load the data
            X_train, X_test, y_train, y_test = load_data(DATA_PATH)

            # Build the model
            model = build_model()

            # Differentially private optimizer
            dp_optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=L2_NORM_CLIP,
                noise_multiplier=NOISE_MULTIPLIER,
                learning_rate=LEARNING_RATE
            )

            model.compile(optimizer=dp_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Retrain the model
            logger.info("Starting model retraining with Differential Privacy")
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

            # Evaluate the retrained model
            loss, accuracy = model.evaluate(X_test, y_test)
            logger.info(f"Evaluation loss: {loss}, Accuracy: {accuracy}")

            # Save the retrained model
            logger.info(f"Saving retrained model to {MODEL_OUTPUT_PATH}")
            model.save(MODEL_OUTPUT_PATH)

            # Register the retrained model in MLflow Model Registry
            mlflow.tensorflow.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME, signature=mlflow.models.infer_signature(X_train[:5].to_numpy(), model.predict(X_train[:5].to_numpy())))

            logger.info("Retraining and registration completed successfully.")
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")

if __name__ == "__main__":
    retrain_model()
