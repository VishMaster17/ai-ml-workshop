import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Map all 7 emotions to integers
    label_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    df['emotion'] = df['emotion'].str.lower().map(label_map)

    if df['emotion'].isnull().any():
        unknowns = df[df['emotion'].isnull()]
        raise ValueError(f"Unknown labels found in CSV: \n{unknowns['emotion'].unique()}")

    # Convert pixels to normalized numpy arrays
    X = np.array([np.fromstring(row, sep=' ') for row in df['pixels']]) / 255.0
    X = X.reshape(-1, 48, 48)
    y = to_categorical(df['emotion'].astype(int), num_classes=7)

    return train_test_split(X, y, test_size=0.2, random_state=42)
