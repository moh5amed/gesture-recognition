# GestureTorch: Hand Gesture Recognition System

GestureTorch is a deep learning-based hand gesture recognition system using PyTorch, MediaPipe, and Streamlit. It supports static gesture classification and a live webcam web app for real-time gesture prediction, making it ideal for contactless computer control and research.

## Features
- **PyTorch CNN (ResNet18) for gesture classification**
- **MediaPipe hand tracking** for robust ROI extraction
- **Streamlit web app** with live webcam support (using `streamlit-webrtc`)
- **Data augmentation** for robust training
- **Easy dataset organization and scripts for splitting/augmentation**
- **Supports 9 gestures**: call_sign, fist, one_finger_down, one_finger_up, open_palm, three_fingers, thumb up, thumb_down, two_fingers

## Directory Structure
```
.
├── add_to_train.py           # Script for collecting and adding new training images
├── augment_synthetic.py      # Script for synthetic data augmentation
├── gesture_control.py        # (Optional) Local gesture control script
├── split_and_augment.py      # Script for splitting and augmenting dataset
├── streamlit_app.py          # Streamlit web app (live webcam)
├── test_model.py             # Script for testing model on images/webcam
├── train.py                  # Model training script
├── gesture_cnn.pth           # Trained model (PyTorch)
├── gesture_cnn_best.pth      # Best model checkpoint
├── data/
│   ├── train/                # Raw training images (by class)
│   ├── train_split/          # Training split (by class)
│   ├── val_split/            # Validation split (by class)
│   ├── train_split_aug/      # Augmented training images (by class)
│   └── ...                   # Other data folders
└── README.md
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd gesturetorch
   ```
2. **Install dependencies:**
   ```bash
   pip install torch torchvision streamlit streamlit-webrtc mediapipe opencv-python pillow numpy tqdm
   ```
3. **(Optional) Download or prepare your dataset:**
   - Place your images in `data/train/<class_name>/`.
   - Use `add_to_train.py` to collect new images with your webcam.

## Usage
### 1. **Data Preparation & Augmentation**
- **Split and augment your dataset:**
  ```bash
  python split_and_augment.py
  ```
- **(Optional) Further augment:**
  ```bash
  python augment_synthetic.py
  ```

### 2. **Model Training**
- Train the model:
  ```bash
  python train.py
  ```
- The best model will be saved as `gesture_cnn_best.pth`.

### 3. **Testing**
- Test the model on images or webcam:
  ```bash
  python test_model.py
  ```

### 4. **Web App (Live Webcam)**
- Run the Streamlit app:
  ```bash
  streamlit run streamlit_app.py
  ```
- Open the local URL in your browser, allow webcam access, and see live predictions.
- Deploy to [Streamlit Cloud](https://streamlit.io/cloud) for public access.

## Data Organization
- **Training/validation/test images** are organized by class in their respective folders.
- **Augmented images** are saved in `data/train_split_aug/<class_name>/` with filenames like `123_aug0.jpg`.
- **Supported gestures:**
  - call_sign
  - fist
  - one_finger_down
  - one_finger_up
  - open_palm
  - three_fingers
  - thumb up
  - thumb_down
  - two_fingers

## Model Details
- **Architecture:** ResNet18 (PyTorch)
- **Input size:** 224x224 RGB
- **Hand ROI:** Detected and cropped using MediaPipe
- **Augmentation:** Random crop, flip, rotation, color jitter, affine, perspective, blur, noise

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [MediaPipe](https://mediapipe.dev/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/) 