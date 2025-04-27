import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.config import INPUT_SHAPE, MODEL_PATH, CLASS_NAMES

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_preprocess_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        raise Exception("No face detected!")

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, INPUT_SHAPE)
    face = face / 255.0
    return face.reshape(1, 48, 48)

def predict_emotion_from_real_photo(img_path):
    model = load_model(MODEL_PATH)
    face = detect_and_preprocess_face(img_path)
    prediction = model.predict(face)
    return CLASS_NAMES[np.argmax(prediction)]

def show_prediction_on_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    model = load_model(MODEL_PATH)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, INPUT_SHAPE)
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=-1)  # ⚡ Add channel dimension (48,48,1)
        face_resized = np.expand_dims(face_resized, axis=0)   # ⚡ Add batch dimension (1,48,48,1)

        pred = model.predict(face_resized)
        emotion = CLASS_NAMES[np.argmax(pred)]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4)

    # Resize for easy viewing
    max_width = 800
    scale = max_width / img.shape[1]
    new_dim = (max_width, int(img.shape[0] * scale))
    resized_img = cv2.resize(img, new_dim)

    cv2.imshow("Prediction", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



