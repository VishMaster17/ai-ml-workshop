import argparse
from src.training.trainer import train_model, train_model_with_accuracy_plot
from src.predictor import predict_emotion_from_real_photo, show_prediction_on_image

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["train", "predict", "predict_visual"])
parser.add_argument("--csv_path", help="Path to training data")
parser.add_argument("--img_path", help="Path to image for prediction")

args = parser.parse_args()

if args.mode == "train":
    train_model_with_accuracy_plot(args.csv_path)
elif args.mode == "predict":
    emotion = predict_emotion_from_real_photo(args.img_path)
    print(f"Predicted Emotion: {emotion}")
elif args.mode == "predict_visual":
    show_prediction_on_image(args.img_path)