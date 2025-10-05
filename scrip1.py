import cv2
import mediapipe as mp
import numpy as np
import os

# Config
DATA_DIR = "dataset"
SEQ_LEN = 32
NUM_CLASSES = 5  # cambia según las señas que vayas a entrenar
CLASS_NAMES = ["hola", "gracias", "si", "no", "tequiero"]  # cambia a tus señas

os.makedirs(DATA_DIR, exist_ok=True)
for cname in CLASS_NAMES:
    os.makedirs(os.path.join(DATA_DIR, cname), exist_ok=True)

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

cap = cv2.VideoCapture(0)

for cname in CLASS_NAMES:
    print(f"Grabando clase: {cname}. Presiona 's' para empezar.")
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f"Preparado para {cname} (s para start)", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Recolector", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    sequences = []
    buffer = []
    while len(sequences) < 30:  # 30 secuencias por clase (ajusta)
        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_array = extract_landmarks(results)
            feat = preprocess(hand_array)
            buffer.append(feat)
            if len(buffer) == SEQ_LEN:
                sequences.append(np.array(buffer))
                buffer = []
                print(f"Secuencia {len(sequences)} capturada para {cname}")
                seq = np.array(sequences[-1], dtype=np.float32)  # (32,84)
                np.save(os.path.join(DATA_DIR, cname, f"{len(sequences)}.npy"), seq)



        cv2.imshow("Recolector", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para parar
            break

cap.release()
cv2.destroyAllWindows()
