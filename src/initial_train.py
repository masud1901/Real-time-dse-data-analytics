import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define file paths
TRAINING_DATA_PATH = "data/training_data/dse_training.csv"
MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "dsex_model_v1.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "model_columns.json")


def train_initial_model():
    """
    Loads the prepared training data, engineers features, trains a
    logistic regression model, and saves it to a file.
    """
    print("🚀 Starting initial model training...")

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
        print(f"✅ Successfully loaded training data with {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Training data not found at '{TRAINING_DATA_PATH}'")
        return

    # --- 2. Feature Engineering ---
    print("🔧 Engineering features...")
    df.set_index("Date", inplace=True)

    # Simple Moving Averages
    df["SMA_5"] = df["Price"].rolling(window=5).mean()
    df["SMA_20"] = df["Price"].rolling(window=20).mean()

    # Price Momentum (change over 5 days)
    df["Momentum_5"] = df["Price"].diff(5)

    # Volume Moving Average
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()

    print("✅ Created features: SMA_5, SMA_20, Momentum_5, Volume_SMA_20")

    # --- 3. Define Target Variable ---
    # We want to predict if the price will go UP (1) or DOWN (0) tomorrow.
    df["Target"] = (df["Price"].shift(-1) > df["Price"]).astype(int)
    print("✅ Created target variable: 'Target' (1 for up, 0 for down).")

    # --- 4. Prepare Data for Model ---
    # Drop rows with NaN values created by rolling windows and shifting
    df.dropna(inplace=True)
    print(f"✅ Dropped NaN rows. Final dataset size: {len(df)} rows.")

    # Define feature columns
    feature_columns = ["SMA_5", "SMA_20", "Momentum_5", "Volume_SMA_20"]
    X = df[feature_columns]
    y = df["Target"]

    # Save the column list for the inference pipeline
    import json

    with open(COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f)
    print(f"✅ Saved feature column names to '{COLUMNS_PATH}'")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"✅ Split data: training ({len(X_train)}), testing ({len(X_test)}).")

    # --- 5. Train the Model ---
    print("🤖 Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # --- 6. Evaluate the Model ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("\n--- Model Evaluation ---")
    print(f"📊 Accuracy on test set: {accuracy:.2f}")
    # Note: Accuracy is often not the best metric for imbalanced classes.
    # 50% is the baseline for a random guess.
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # --- 7. Save the Trained Model ---
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Successfully saved trained model to '{MODEL_PATH}'")

    print("\n🎉 Initial model training complete!")
    print("This model is now ready for the real-time inference pipeline.")


if __name__ == "__main__":
    train_initial_model()
