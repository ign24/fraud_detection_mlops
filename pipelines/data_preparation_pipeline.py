import pandas as pd
from imblearn.over_sampling import SMOTE
import logging
from sklearn.model_selection import train_test_split
from google.cloud import storage

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataPreparationPipeline")

# Configuration parameters
DATA_PATH = 'data/raw/creditcard.csv'  # Path to the initial dataset
OUTPUT_PATH = 'data/prepared_data.csv'

# Load the data from local storage or GCS
def load_data(path):
    logger.info(f"Loading data from {path}")
    
    # Example for reading from GCS
    if path.startswith("gs://"):
        storage_client = storage.Client()
        bucket_name, file_name = path[5:].split('/', 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_text()
        return pd.read_csv(pd.compat.StringIO(data))
    
    # Reading locally if not using GCS
    return pd.read_csv(path)

# Apply SMOTE to balance the classes
def balance_data(df):
    # Data validation
    if df.isnull().sum().sum() > 0:
        raise ValueError("The dataset contains null values. Please clean the data before proceeding.")
    
    logger.info("Applying SMOTE to balance the classes")
    X = df.drop(columns=['Class'])
    y = df['Class']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)
    
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(balanced_df, test_size=0.2, random_state=42)
    return train_data, test_data

# Save the prepared data locally or to GCS
def save_data(df, path):
    logger.info(f"Saving prepared data to {path}")
    
    if path.startswith("gs://"):
        storage_client = storage.Client()
        bucket_name, file_name = path[5:].split('/', 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(df.to_csv(index=False), 'text/csv')
    else:
        df.to_csv(path, index=False)

# Full data preparation function for the training process
def prepare_data():
    try:
        # Load data
        data = load_data(DATA_PATH)
        
        # Balance the data using SMOTE (only for initial training)
        train_data, test_data = balance_data(data)
        
        # Save the prepared data
        save_data(train_data, OUTPUT_PATH)
        
        logger.info("Data preparation completed successfully.")
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")

if __name__ == "__main__":
    prepare_data()