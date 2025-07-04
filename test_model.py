import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow/MediaPipe info and warning logs
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

# --- CONFIG ---
CLASS_NAMES = ['call_sign', 'fist', 'one_finger_down', 'one_finger_up', 'open_palm', 'three_fingers', 'thumb up', 'thumb_down', 'two_fingers']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = 'gesture_cnn_best.pth'
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD MODEL ---
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features,NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_transformed = transform(img_pil).unsqueeze(0).to(DEVICE)
    return img_transformed

# --- MEDIAPIPE HANDS ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
print("\U0001F4F9 Starting webcam... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    label = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            margin = 20
            x1 = max(xmin - margin, 0)
            y1 = max(ymin - margin, 0)
            x2 = min(xmax + margin, w)
            y2 = min(ymax + margin, h)
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                label = None
                break
            cv2.imshow('ROI', roi)
            try:
                input_tensor = preprocess_frame(roi)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    label = CLASS_NAMES[predicted.item()]
            except Exception as e:
                print('Error during prediction:', e)
                label = None
            break  # Only use the first detected hand

    if label:
        cv2.putText(display_frame, f'Gesture: {label}', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(display_frame, 'Gesture: ...', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow('Gesture Recognition', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 