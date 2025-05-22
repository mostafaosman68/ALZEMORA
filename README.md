# ALZEMORA


A wearable smart hat that assists Alzheimer's patients in recognizing **family members**, **medications**, and remembering **when to take them** using facial recognition, object detection, and voice recognition.

---

## 📌 Features

- 🔍 **Face Recognition**  
  Detects and identifies known faces using `face_recognition` and `YOLOv8`.

- 💊 **Object Detection (Medication)**  
  Detects medication using `YOLOv8` and gives reminders based on time and object recognition.

- 🎙️ **Voice Recognition**  
  Identifies people by their voice using a trained voice recognition model.

- 📷 **ESP32-CAM Integration**  
  Uses ESP32-CAM to capture live images, connected with the main processing code.

- 🗣️ **Audio Feedback**  
  Provides real-time spoken responses for recognized people and objects.

---

## 🛠️ Technologies Used

- `face_recognition`
- `ultralytics/yolov8`
- `speech_recognition`
- `ESP32-CAM`
- `OpenCV`, `NumPy`, `Pyttsx3`
- Python 3.x
- Arduino IDE (for ESP32 firmware)

---

## 🎯 Use Case

When a person appears:
- ESP32-CAM captures the image.
- YOLOv8 locates the face.
- Face is matched using the face recognition model.
- The hat plays audio: _"This is your daughter, Sarah."_

When a medication bottle is detected:
- YOLOv8 detects the medication.
- Time is checked.
- The hat plays audio: _"It's time to take your blood pressure medicine."_

When someone speaks:
- The microphone records the voice.
- The system identifies the person.
- Audio feedback: _"You are talking to John."_

---

## 🔌 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/alzheimers-smart-hat.git
   cd alzheimers-smart-hat
