import tensorflow as tf
import tensorflow_privacy
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingPipeline")

# Configuration parameters
DATA_PATH = 'data/prepared_data.csv'
MODEL_OUTPUT_PATH = 'models/fraud_detection_model.keras'
EPOCHS = 20
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

# Training function
def train_model():
    try:
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

        # Train the model
        logger.info("Starting model training with Differential Privacy")
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

        # Save the trained model
        logger.info(f"Saving trained model to {MODEL_OUTPUT_PATH}")
        model.save(MODEL_OUTPUT_PATH)

        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error in training: {e}")

if __name__ == "__main__":
    train_model()
