import json
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (StructType, StructField,
                               StringType, DoubleType, TimestampType)
from pyspark.sql.window import Window
import pyspark.sql.functions as F


# --- Configuration ---
KAFKA_BROKER = "kafka:29092"
KAFKA_TOPIC = "dse_index_data"
SPARK_MASTER_URL = "spark://spark-master:7077"
APP_NAME = "DSE_Streaming_Prediction"

MODEL_PATH = '/app/data/models/dsex_model_v1.pkl'
MODEL_COLUMNS_PATH = '/app/data/models/model_columns.json'
TRAINING_DATA_PATH = '/app/data/training_data/dse_training.csv'


# Define the schema for the incoming Kafka data
DATA_SCHEMA = StructType([
    StructField("Date", TimestampType(), True),
    StructField("Price", DoubleType(), True),
    StructField("Open", DoubleType(), True),
    StructField("High", DoubleType(), True),
    StructField("Low", DoubleType(), True),
    StructField("Vol", StringType(), True),  # Keep as string for now
    StructField("Change", DoubleType(), True)
])


def train_model_fallback():
    """Train a new model using the current environment if loading fails."""
    print("🔄 Training new model in current environment...")
    
    # Load training data
    df = pd.read_csv(TRAINING_DATA_PATH)
    
    # Feature engineering (same as original training script)
    df['SMA_5'] = df['Price'].rolling(window=5).mean()
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    df['Momentum_5'] = (df['Price'] / df['Price'].shift(5)) * 100
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # Create target variable
    df['Target'] = (df['Price'].shift(-1) > df['Price']).astype(int)
    
    # Drop NaN rows
    df.dropna(inplace=True)
    
    # Features and target
    feature_columns = ['SMA_5', 'SMA_20', 'Momentum_5', 'Volume_SMA_20']
    X = df[feature_columns]
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    print(f"✅ New model trained with accuracy: "
          f"{model.score(X_test, y_test):.3f}")
    
    return model, feature_columns


def create_spark_session():
    """Creates and configures a Spark Session."""
    return (SparkSession.builder
            .appName(APP_NAME)
            .master(SPARK_MASTER_URL)
            .config("spark.jars.packages",
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0")
            .getOrCreate())


def clean_volume(volume_str):
    """Converts volume string (e.g., '100M', '50K') to a numeric value."""
    if volume_str is None:
        return 0.0
    if isinstance(volume_str, str):
        volume_str = volume_str.strip().upper()
        if 'M' in volume_str:
            return float(volume_str.replace('M', '')) * 1_000_000
        elif 'K' in volume_str:
            return float(volume_str.replace('K', '')) * 1_000
    return float(volume_str)


# Register the UDF
clean_volume_udf = F.udf(clean_volume, DoubleType())


def engineer_features(df):
    """Applies feature engineering to the streaming DataFrame."""
    df = df.withColumn("Volume_Clean", clean_volume_udf(col("Vol")))

    # Define window specifications
    window_5 = Window.orderBy("Date").rowsBetween(-4, 0)
    window_20 = Window.orderBy("Date").rowsBetween(-19, 0)

    # Create features
    df = df.withColumn("SMA_5", F.avg("Price").over(window_5))
    df = df.withColumn("SMA_20", F.avg("Price").over(window_20))
    df = df.withColumn(
        "Momentum_5",
        (col("Price") / F.lag("Price", 5).over(Window.orderBy("Date"))) * 100
    )
    df = df.withColumn("Volume_SMA_20", F.avg("Volume_Clean").over(window_20))

    return df


def make_prediction(df, model, model_columns):
    """Makes a prediction on a Pandas DataFrame."""
    # This requires collecting data to the driver.
    # Ok for small batches, but not for large-scale production.
    
    # Convert Date to string to avoid pandas datetime casting issues
    df = df.withColumn("Date", col("Date").cast("string"))
    df_pandas = df.toPandas()

    # Ensure all required columns are present
    for col_name in model_columns:
        if col_name not in df_pandas.columns:
            df_pandas[col_name] = 0  # Or some other default

    # Handle NaN values in features - fill with reasonable defaults
    # For moving averages, use the current price as fallback
    if 'SMA_5' in df_pandas.columns:
        df_pandas['SMA_5'].fillna(df_pandas['Price'], inplace=True)
    if 'SMA_20' in df_pandas.columns:
        df_pandas['SMA_20'].fillna(df_pandas['Price'], inplace=True)
    if 'Momentum_5' in df_pandas.columns:
        df_pandas['Momentum_5'].fillna(100.0, inplace=True)  # No change
    if 'Volume_SMA_20' in df_pandas.columns:
        df_pandas['Volume_SMA_20'].fillna(
            df_pandas.get('Volume_Clean', 0), inplace=True)

    # Keep only model columns and make prediction
    X_test = df_pandas[model_columns]
    prediction = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # Add predictions to the DataFrame
    df_pandas['prediction'] = prediction[0]
    df_pandas['probability_0'] = proba[0][0]
    df_pandas['probability_1'] = proba[0][1]

    return df_pandas


def process_batch(batch_df, epoch_id, model, model_columns):
    """Processing logic for each micro-batch."""
    if batch_df.count() == 0:
        return

    print(f"--- Processing Batch {epoch_id} ---")

    # 1. Feature Engineering
    features_df = engineer_features(batch_df)

    # 2. Make Predictions
    # For simplicity, we collect the batch and predict.
    # WARNING: Not suitable for large-scale production.
    if not features_df.rdd.isEmpty():
        try:
            # We predict on the last row of the batch for this example
            latest_row_df = features_df.orderBy(col("Date").desc()).limit(1)

            # Predict
            prediction_df = make_prediction(
                latest_row_df, model, model_columns)

            print("--- Prediction Result ---")
            print(prediction_df[['Date', 'Price', 'prediction',
                                 'probability_0', 'probability_1']])

        except Exception as e:
            print(f"Error making prediction: {e}")


if __name__ == "__main__":
    # Load the trained model and columns
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(MODEL_COLUMNS_PATH, 'r') as f:
            model_columns = json.load(f)
        print("✅ Model and column list loaded successfully.")
    except (FileNotFoundError, ModuleNotFoundError) as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to training new model...")
        model, model_columns = train_model_fallback()

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Read from Kafka
    kafka_df = (spark.readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", KAFKA_BROKER)
                .option("subscribe", KAFKA_TOPIC)
                .option("startingOffsets", "latest")
                .load())

    # Deserialize the JSON data
    processed_df = (kafka_df.select(
        from_json(col("value").cast("string"), DATA_SCHEMA).alias("data"))
        .select("data.*"))

    # Apply processing logic to each micro-batch
    query = (processed_df.writeStream
             .foreachBatch(lambda df, epoch_id: process_batch(
                 df, epoch_id, model, model_columns))
             .start())

    print("🚀 Spark Streaming consumer started. Waiting for data from Kafka...")
    query.awaitTermination() 