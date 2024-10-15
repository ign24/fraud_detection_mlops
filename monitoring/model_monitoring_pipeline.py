from kafka import KafkaConsumer
import json
import logging
import time
import requests
from prometheus_client import Gauge, start_http_server
import os
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable
from sklearn.model_selection import train_test_split

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelMonitoringPipeline")

# Configuration parameters
KAFKA_BROKER_URL = os.getenv('KAFKA_BROKER_URL', 'localhost:9092')  # Configurable for different environments
KAFKA_TOPIC = 'credit_card_transactions'
CSV_FILE_PATH = 'data/new_transactions.csv'
BASELINE_CSV_FILE_PATH = 'data/baseline_data.csv'  # The initial baseline data file for drift comparison
PROMETHEUS_PORT = 8000
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')  # Slack webhook URL for notifications

# Prometheus metrics
transaction_count_gauge = Gauge('transaction_count', 'Number of transactions processed')
data_drift_gauge = Gauge('data_drift', 'Indicator of potential data drift')


def process_message(message):
    """
    Processes each message received from Kafka.
    Args:
        message (dict): Deserialized Kafka message.
    """
    logger.info(f"Processing transaction: {message}")

    # Write the message to a CSV file
    try:
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Writing headers if file is new
            if file.tell() == 0:
                writer.writerow(message.keys())
            # Writing message content
            writer.writerow(message.values())
        transaction_count_gauge.inc()  # Increment the transaction count metric
    except Exception as e:
        logger.error(f"Error writing message to CSV: {e}")


def detect_data_drift():
    """
    Function to detect data drift using Evidently and update Prometheus gauge.
    """
    try:
        # Load baseline and current data
        baseline_data = pd.read_csv(BASELINE_CSV_FILE_PATH)
        current_data = pd.read_csv(CSV_FILE_PATH)

        # Define the column mapping for Evidently
        column_mapping = ColumnMapping(target='Class')

        # Create the data drift report
        drift_report = Report(metrics=[DataDriftTable(), DataDriftPreset()])
        drift_report.run(reference_data=baseline_data, current_data=current_data, column_mapping=column_mapping)

        # Extract drift information
        drift_summary = drift_report.as_dict()
        drift_detected = drift_summary["metrics"][0]["result"]["drift_share"] > 0.5  # Adjust threshold as needed

        # Update the Prometheus gauge
        data_drift_gauge.set(1 if drift_detected else 0)

        # Send notification if drift is detected
        if drift_detected:
            send_drift_notification()
    except Exception as e:
        logger.error(f"Error in detecting data drift: {e}")


def send_drift_notification():
    """
    Sends a notification if data drift is detected.
    """
    if SLACK_WEBHOOK_URL:
        message = {
            "text": "Data drift detected in the fraud detection model pipeline. Immediate attention is required."
        }
        try:
            response = requests.post(SLACK_WEBHOOK_URL, json=message)
            if response.status_code == 200:
                logger.info("Data drift notification sent successfully.")
            else:
                logger.error(f"Failed to send data drift notification: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error sending data drift notification: {e}")
    else:
        logger.warning("Slack Webhook URL not configured. Unable to send data drift notification.")


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
            detect_data_drift()  # Check for data drift after processing each message
    except KeyboardInterrupt:
        logger.info("Data ingestion manually stopped.")
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        logger.info("Attempting to reconnect to Kafka...")
        time.sleep(5)


if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")

    # Start data ingestion
    ingest_data()