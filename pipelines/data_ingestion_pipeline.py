from kafka import KafkaConsumer
import json
import logging
import time
from google.cloud import storage
import os

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataIngestionPipeline")

# Configuration parameters
KAFKA_BROKER_URL = 'localhost:9092'  # Change according to environment settings
KAFKA_TOPIC = 'credit_card_transactions'
GCS_BUCKET_NAME = 'your-gcs-bucket-name'  # Replace with your GCS bucket name
GCS_FILE_PATH = 'data/new_transactions.json'

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def process_message(message):
    """
    Processes each message received from Kafka.
    Args:
        message (dict): Deserialized Kafka message.
    """
    logger.info(f"Processing transaction: {message}")
    
    # Save message to GCS
    try:
        blob = bucket.blob(GCS_FILE_PATH)
        blob.upload_from_string(json.dumps(message) + '\n', content_type='application/json', if_generation_match=0)
        logger.info(f"Message saved to GCS: {GCS_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error saving message to GCS: {e}")

def ingest_data():
    """
    Function for ingesting data from Kafka topic.
    """
    # Create Kafka consumer
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER_URL],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='fraud_detection_group'
    )

    logger.info("Connected to Kafka, waiting for messages...")

    try:
        for message in consumer:
            process_message(message.value)
    except KeyboardInterrupt:
        logger.info("Data ingestion manually stopped.")
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        logger.info("Attempting to reconnect to Kafka...")
        time.sleep(5)

if __name__ == "__main__":
    ingest_data()