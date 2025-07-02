import pandas as pd
import os

# Define file paths
RAW_DATA_PATH = "data/raw/DSEX 2013-25.csv"
TRAINING_DATA_PATH = "data/training_data/dse_training.csv"
SIMULATION_DATA_PATH = "data/simulation_data/dse_streaming.csv"


def clean_numeric_value(value):
    """
    Removes commas from a string and converts to float.
    Handles non-string inputs gracefully.
    """
    if isinstance(value, str):
        return float(value.replace(",", ""))
    return value


def prepare_data():
    """
    Loads, cleans, and splits the raw DSEX dataset into historical
    training data and data for the real-time simulation.
    """
    print("🚀 Starting data preparation for DSEX dataset...")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SIMULATION_DATA_PATH), exist_ok=True)

    # --- 1. Load the Raw Data ---
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"✅ Successfully loaded raw data with {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Raw data file not found at '{RAW_DATA_PATH}'")
        return

    # --- 2. Data Cleaning & Preprocessing ---
    print("🧹 Cleaning and preprocessing data...")

    # Convert 'Date' column to datetime objects
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    print("✅ Converted 'Date' column to datetime objects.")

    # Clean numeric columns
    numeric_cols = ["Price", "Open", "High", "Low"]
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric_value)
    print(f"✅ Cleaned numeric columns: {numeric_cols}")

    # Handle 'Vol.' column (e.g., '10.5M' -> 10500000)
    def convert_volume(vol):
        if isinstance(vol, str):
            vol = vol.strip().upper()
            if "M" in vol:
                return float(vol.replace("M", "")) * 1_000_000
            if "K" in vol:
                return float(vol.replace("K", "")) * 1_000
        return 0  # Return 0 if format is unexpected or NaN

    df["Volume"] = df["Vol."].apply(convert_volume)
    df.drop("Vol.", axis=1, inplace=True)
    print("✅ Processed 'Vol.' column into numeric 'Volume'.")

    # Clean 'Change %' column
    df["Change_Pct"] = df["Change %"].str.replace("%", "").astype(float)
    df.drop("Change %", axis=1, inplace=True)
    print("✅ Processed 'Change %' into numeric 'Change_Pct'.")

    # Sort by date to ensure correct order for streaming
    df.sort_values(by="Date", ascending=True, inplace=True)
    print("✅ Sorted data by date ascending.")

    # --- 3. Split the Data ---
    # We'll use data up to the end of 2023 for training
    # and 2024 onwards for simulation.
    split_date = "2024-01-01"

    training_df = df[df["Date"] < split_date]
    simulation_df = df[df["Date"] >= split_date]

    if training_df.empty or simulation_df.empty:
        print("❌ ERROR: Split resulted in one or both dataframes being empty.")
        print(
            "Please check your dataset's date range and the split_date "
            f"('{split_date}')."
        )
        return

    print(f"🔪 Splitting data at {split_date}:")
    print(f"   - {len(training_df)} rows for Training (before {split_date})")
    print(f"   - {len(simulation_df)} rows for Simulation " f"({split_date} and after)")

    # --- 4. Save the Split Datasets ---
    training_df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"✅ Saved training data to '{TRAINING_DATA_PATH}'")

    simulation_df.to_csv(SIMULATION_DATA_PATH, index=False)
    print(f"✅ Saved simulation data to '{SIMULATION_DATA_PATH}'")

    print("\n🎉 Data preparation complete!")
    print("You can now proceed to initial model training.")


if __name__ == "__main__":
    prepare_data()
