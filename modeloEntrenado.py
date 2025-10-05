# Requisitos: pip install mediapipe opencv-python numpy tensorflow
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Config
SEQ_LEN = 32
CLASS_NAMES = ["hola", "gracias", "si", "no", "tequiero"]  # mismas clases que tu dataset
MODEL_PATH = "modelo_signos.h5"  # tu modelo entrenado
model = load_model(MODEL_PATH)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    hand_list = []
    for hand_landmarks in results.multi_hand_landmarks:
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y])
        hand_list.append(coords)
    if len(hand_list) == 1:
        hand_list.append([[0,0]]*21)
    return np.array(hand_list)

def preprocess(hand_array):
    processed = []
    for h in hand_array:
        h = np.array(h)
        wrist = h[0].copy()
        h -= wrist
        max_val = np.max(np.abs(h))
        if max_val > 1e-6:
            h /= max_val
        processed.append(h.flatten())
    return np.concatenate(processed)  # (84,)

# Buffer para secuencias
buffer = []

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
        hand_array = extract_landmarks(results)
        if hand_array is not None:
            feat = preprocess(hand_array)
            buffer.append(feat)
            if len(buffer) > SEQ_LEN:
                buffer.pop(0)
    else:
        buffer.append(np.zeros(84))  # sin manos → vector de ceros
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)

    # Cuando el buffer está lleno → predecir
    if len(buffer) == SEQ_LEN:
        X = np.array(buffer)  # (32,84)
        X = X[np.newaxis, ...]  # (1,32,84)
        preds = model.predict(X, verbose=0)  # shape (1,5)
        cls = np.argmax(preds[0])
        conf = preds[0, cls]
        label = CLASS_NAMES[cls]

        cv2.putText(frame, f"{label} ({conf:.2f})", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
