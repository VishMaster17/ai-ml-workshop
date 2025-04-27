from src.training.model_builder import build_model
from src.training.data_loader import load_data
from src.config import MODEL_PATH, INPUT_SHAPE, NUM_CLASSES
from src.training.visualizer import plot_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(csv_path):
    X_train, X_test, y_train, y_test = load_data(csv_path)
    model = build_model(INPUT_SHAPE, NUM_CLASSES)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=15, batch_size=64,
              callbacks=[ModelCheckpoint(MODEL_PATH, save_best_only=True)])

def train_model_with_accuracy_plot(csv_path):
    X_train, X_test, y_train, y_test = load_data(csv_path)
    model = build_model(INPUT_SHAPE, NUM_CLASSES)

    # Save best model checkpoint
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True)

    # ⚡ Capture history
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=15,
                        batch_size=64,
                        callbacks=[checkpoint])
    
    # ⚡ Plot training vs validation accuracy
    plot_accuracy(history)

