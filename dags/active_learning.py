from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Configuración de los argumentos por defecto para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
with DAG(
    'fraud_detection_active_learning',
    default_args=default_args,
    description='DAG for active learning lifecycle of fraud detection model',
    schedule_interval=timedelta(days=7),
    catchup=False,
) as dag:

    # Tarea 1: Ejecutar el contenedor de ingestión de datos
    ingest_data_task = BashOperator(
        task_id='ingest_data',
        bash_command='docker build -f Dockerfile.data_ingestion -t fraud_detection_data_ingestion . && docker run --rm fraud_detection_data_ingestion',
        retries=3
    )

    # Tarea 2: Ejecutar el contenedor de preparación de datos (TFDV)
    validate_data_task = BashOperator(
        task_id='validate_data',
        bash_command='docker build -f Dockerfile.data_preparation -t fraud_detection_data_preparation . && docker run --rm fraud_detection_data_preparation',
        retries=3
    )

    # Tarea 3: Ejecutar el contenedor de reentrenamiento del modelo
    retrain_model_task = BashOperator(
        task_id='retrain_model',
        bash_command='docker build -f Dockerfile.model_retraining_pipeline -t fraud_detection_model_retraining . && docker run --rm fraud_detection_model_retraining',
        retries=3
    )

    # Tarea 4: Ejecutar el contenedor de monitoreo del modelo
    monitor_model_task = BashOperator(
        task_id='monitor_model',
        bash_command='docker build -f Dockerfile.monitoring -t fraud_detection_model_monitoring . && docker run --rm fraud_detection_model_monitoring',
        retries=3
    )

    # Definir la secuencia de tareas
    ingest_data_task >> validate_data_task >> retrain_model_task >> monitor_model_task