# 🧠 Smart Memory Aid Hat for Alzheimer's Patients

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

- [`face_recognition`](https://github.com/ageitgey/face_recognition)
- [`ultralytics/yolov8`](https://github.com/ultralytics/ultralytics)
- [`speech_recognition`](https://pypi.org/project/SpeechRecognition/)
- `ESP32-CAM`
- `OpenCV`, `NumPy`, `pyttsx3` for speech synthesis
- Python 3.x
- Arduino IDE (for ESP32 firmware)

---

## 🎯 Use Case

### 🧍 Face Detection & Recognition
1. ESP32-CAM captures the image.
2. YOLOv8 detects the face.
3. `face_recognition` matches the face.
4. Hat plays audio:  
   _"This is your daughter, Sarah."_

### 💊 Medication Detection
1. YOLOv8 detects the medication bottle.
2. System checks current time.
3. Audio reminder plays:  
   _"It's time to take your blood pressure medicine."_

### 🎙️ Voice Recognition
1. Microphone records incoming voice.
2. Voice is matched with stored profiles.
3. Audio feedback:  
   _"You are talking to John."_

---
###Future Enhancements
⏰ Medication scheduler with cloud synchronization

🧭 GPS tracker for user safety

📱 Companion mobile app

🧠 Improved NLP-powered voice interaction

## 🔌 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/mostafaosman68/ALZEMORA
cd ALZEMORA


