import pandas as pd
import tensorflow_data_validation as tfdv
import logging
import os

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataValidationPipeline")

# Configuration parameters
DATA_PATH = 'data/new_data.csv'  # Path to the new dataset
REFERENCE_STATS_PATH = 'data/reference_stats'  # Path to the reference statistics for TFDV validation

# Load the data
def load_data(path):
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)

# Validate data using TFDV
def validate_data(df):
    logger.info("Validating data with TFDV")
    # Generate statistics for the new data
    new_stats = tfdv.generate_statistics_from_dataframe(df)
    
    # Load reference statistics
    if not os.path.exists(REFERENCE_STATS_PATH):
        logger.warning("Reference statistics not found. Saving new statistics as reference.")
        tfdv.write_statistics_text(new_stats, REFERENCE_STATS_PATH)
        return
    
    reference_stats = tfdv.load_statistics(input_path=REFERENCE_STATS_PATH)
    
    # Compare new statistics with reference
    anomalies = tfdv.validate_statistics(new_stats, reference_stats)
    if anomalies.anomaly_info:
        logger.warning(f"Data drift detected: {anomalies}")
        raise ValueError("Data drift detected in the new data.")
    else:
        logger.info("No data drift detected.")

# Full data validation function for the DAG
def validate_new_data():
    try:
        # Load data
        data = load_data(DATA_PATH)
        
        # Validate the data using TFDV
        validate_data(data)
        
        logger.info("Data validation completed successfully.")
    except Exception as e:
        logger.error(f"Error in data validation: {e}")

if __name__ == "__main__":
    validate_new_data()
