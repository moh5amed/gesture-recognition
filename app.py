import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import mediapipe as mp
import cv2

# --- CONFIG ---
CLASS_NAMES = ['call_sign', 'fist', 'one_finger_down', 'one_finger_up', 'open_palm', 'three_fingers', 'thumb up', 'thumb_down', 'two_fingers']
MODEL_PATH = 'gesture_cnn_best.pth'
IMG_SIZE = 224

@st.cache_resource
def load_model():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

model = load_model()

st.title("Live Hand Gesture Recognition (with Hand Tracking)")
st.write("Show your hand gesture to the webcam. The model will predict the gesture in real time using MediaPipe hand tracking.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_prediction = None
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Normalize color
        results = self.hands.process(img_rgb)
        label = None
        bbox = None
        h, w, _ = img.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                margin = 20
                x1 = max(xmin - margin, 0)
                y1 = max(ymin - margin, 0)
                x2 = min(xmax + margin, w)
                y2 = min(ymax + margin, h)
                bbox = (x1, y1, x2, y2)
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                break  # Only use the first detected hand
        if bbox:
            x1, y1, x2, y2 = bbox
            roi = img_rgb[y1:y2, x1:x2]  # Use RGB for model
            if roi.size != 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                roi_pil = Image.fromarray(roi)
                input_tensor = transform(roi_pil).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    label = CLASS_NAMES[predicted.item()]
                self.last_prediction = label
                cv2.putText(img, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                self.last_prediction = None
                cv2.putText(img, "Hand too small", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            self.last_prediction = None
            cv2.putText(img, "No hand detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return av.VideoFrame.from_ndarray(img, format='bgr24')

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="gesture-recognition",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    st.markdown("**Prediction:** " + str(webrtc_ctx.video_processor.last_prediction)) 