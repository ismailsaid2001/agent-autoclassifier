from src.data_loader import load_data
from src.ml_pipeline import train_auto_ml

if __name__ == "__main__":
    df = load_data("data/train.csv")
    train_auto_ml(df)
