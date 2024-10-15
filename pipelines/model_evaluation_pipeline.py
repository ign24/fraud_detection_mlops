import tensorflow as tf
import tensorflow_model_analysis as tfma
import pandas as pd
import logging
import os

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelEvaluationPipeline")

# Configuration parameters
MODEL_PATH = 'models/fraud_detection_model.keras'
DATA_PATH = 'data/prepared_data.csv'
EVAL_OUTPUT_DIR = 'evaluation/'

# Load the model
def load_model(path):
    logger.info(f"Loading model from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at path: {path}")
    return tf.keras.models.load_model(path)

# Configure and run the evaluation
def run_evaluation(model, data_path, output_dir):
    logger.info("Configuring evaluation with TFMA")

    # Load the data
    data = pd.read_csv(data_path)
    if data.isnull().sum().sum() > 0:
        raise ValueError("The dataset contains null values. Please clean the data before proceeding.")

    # Convert pandas DataFrame to TFMA compatible format
    tfma_data = data.to_dict(orient='records')

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Class')],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.metrics.MetricConfig(class_name='AUC'),
                tfma.metrics.MetricConfig(class_name='Precision'),
                tfma.metrics.MetricConfig(class_name='Recall')
            ])
        ],
        slicing_specs=[tfma.SlicingSpec()]
    )

    eval_shared_model = tfma.default_eval_shared_model(eval_saved_model_path=MODEL_PATH)

    logger.info("Running model analysis with TFMA")
    tfma.run_model_analysis(
        eval_shared_model=eval_shared_model,
        data=tfma_data,
        eval_config=eval_config,
        output_path=output_dir
    )

# Full model evaluation function for the DAG
def evaluate_model():
    try:
        # Load the trained model
        model = load_model(MODEL_PATH)
        
        # Run the model evaluation
        run_evaluation(model, DATA_PATH, EVAL_OUTPUT_DIR)
        
        logger.info("Model evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")

if __name__ == "__main__":
    evaluate_model()