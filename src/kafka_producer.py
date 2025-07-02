import pandas as pd
from kafka import KafkaProducer
import json
import time
import logging

# Setup basic logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'dse_index_data'
DATA_SOURCE_FILE = 'data/dse_streaming.csv'
# Time to wait between sending messages
SIMULATION_SPEED_SECONDS = 5

def create_kafka_producer(broker_url):
    """Creates a Kafka producer instance."""
    logging.info(f"Attempting to connect to Kafka broker at {broker_url}...")
    try:
        # api_version is often needed for compatibility
        producer = KafkaProducer(
            bootstrap_servers=[broker_url],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=5,
            request_timeout_ms=30000,
            api_version=(0, 10, 1)
        )
        logging.info("✅ Successfully connected to Kafka broker.")
        return producer
    except Exception as e:
        logging.error(f"❌ Failed to connect to Kafka broker: {e}")
        return None

def stream_data(producer, topic, file_path):
    """
    Reads data from a CSV file and streams it to a Kafka topic indefinitely.
    """
    logging.info(
        f"Starting to stream from '{file_path}' to topic '{topic}'."
    )

    try:
        data_df = pd.read_csv(file_path)
        logging.info(f"✅ Loaded {len(data_df)} records for streaming.")
    except FileNotFoundError:
        logging.error(f"❌ Error: The file '{file_path}' was not found.")
        return

    # Loop indefinitely to simulate a continuous stream
    while True:
        for _, row in data_df.iterrows():
            message = row.to_dict()

            logging.info(f"📨 Sending message: {message}")
            producer.send(topic, value=message)

            time.sleep(SIMULATION_SPEED_SECONDS)

        logging.info(
            "✅ Finished one pass over the data. "
            "Restarting stream for continuous simulation."
        )

if __name__ == "__main__":
    producer = create_kafka_producer(KAFKA_BROKER)

    if producer:
        try:
            stream_data(producer, KAFKA_TOPIC, DATA_SOURCE_FILE)
        except KeyboardInterrupt:
            logging.info("🔌 Stream manually interrupted by user.")
        finally:
            producer.flush()
            producer.close()
            logging.info("🛑 Kafka producer closed.") 