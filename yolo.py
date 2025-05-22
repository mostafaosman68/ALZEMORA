import face_recognition
import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import pyttsx3  # For text-to-speech
import threading

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speaking rate
engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

# Create a lock for thread-safe speaking
speak_lock = threading.Lock()
last_spoken = {}  # Track when objects/faces were last announced
speak_cooldown = 5.0  # Only announce the same object again after this many seconds

# Function to speak text in a separate thread
def speak_text(text):
    # Check if this exact text was spoken recently
    current_time = time.time()
    if text in last_spoken and current_time - last_spoken[text] < speak_cooldown:
        return  # Skip if this was announced too recently
    
    # Update the last spoken time
    last_spoken[text] = current_time
    
    # Speak in a separate thread to avoid blocking the main program
    def speak_thread():
        with speak_lock:  # Ensure only one speech at a time
            engine.say(text)
            engine.runAndWait()
    
    threading.Thread(target=speak_thread).start()

# Paths
known_faces_dir = r"C:/Users/ITD/Desktop/NewYOLOO/known_faces"

# Initialize arrays to store face encodings and names
known_face_encodings = []
known_face_names = []

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("best (1).pt")  # Using nano version for speed
class_names = model.names
print("Available classes:")
for idx, name in class_names.items():
    print(f"{idx}: {name}")
    
classes_to_keep = [2, 3]
# Object and face detection settings
object_confidence_threshold = 0.5
tracking_confidence_threshold = 0.35
face_recognition_threshold = 0.6  # Adjusted for better unknown detection

# Create object trackers dictionary
class TrackedObject:
    def __init__(self, class_name, box, confidence, max_history=8):
        self.class_name = class_name
        self.boxes = deque(maxlen=max_history)
        self.confidences = deque(maxlen=max_history)
        self.last_seen = 0
        self.announced = False  # Track if this object has been announced
        self.update(box, confidence)
        
    def update(self, box, confidence):
        self.boxes.append(box)
        self.confidences.append(confidence)
        self.last_seen = 0
        
    def get_smoothed_box(self):
        if not self.boxes:
            return None
        recent_boxes = list(self.boxes)[-3:]
        avg_box = np.mean(recent_boxes, axis=0).astype(int)
        return avg_box.tolist()
        
    def get_confidence(self):
        if not self.confidences:
            return 0
        return np.mean(list(self.confidences))
        
    def increment_last_seen(self):
        self.last_seen += 1

# Dictionary to track objects by their IDs
tracked_objects = {}

# Load known faces
print("Loading known faces...")

# Go through each person's directory in known_faces
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    
    # Skip if not a directory
    if not os.path.isdir(person_dir):
        continue
    # Process each image file of the person
    for img_name in os.listdir(person_dir):
        # Get file path
        img_path = os.path.join(person_dir, img_name)
        
        # Skip non-image files
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Load image and get face encoding
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image, model="hog")
        
        # Skip image if no face is detected
        if not face_locations:
            print(f"No face detected in {img_path}")
            continue
        
        # Get encoding of first face found
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        # Add encoding and name to arrays
        known_face_encodings.append(face_encoding)
        known_face_names.append(person_name)
        
        print(f"Loaded and encoded face from {img_path}")

print(f"Loaded {len(known_face_encodings)} face(s)")

# Initialize webcamq
video_capture = cv2.VideoCapture("http://172.20.10.2:81/stream", cv2.CAP_FFMPEG)
# video_capture = cv2.VideoCapture(0)

# For FPS calculation
prev_time = time.time()
frame_count = 0
fps = 0

# Frame counters
process_face_frame = 0
process_object_frame = 0
frame_number = 0

# Max frames to keep an object without new detection
max_frames_to_keep = 10

# Create a simple face tracker to stabilize unknown faces
class TrackedFace:
    def __init__(self, location, name, confidence=0, max_history=5):
        self.locations = deque(maxlen=max_history)
        self.name = name
        self.confidence = confidence
        self.last_seen = 0
        self.announced = False  # Track if this face has been announced
        self.update(location)
        
    def update(self, location):
        self.locations.append(location)
        self.last_seen = 0
        
    def get_smoothed_location(self):
        if not self.locations:
            return None
        recent_locs = list(self.locations)[-3:]
        return [
            int(np.mean([loc[0] for loc in recent_locs])),
            int(np.mean([loc[1] for loc in recent_locs])),
            int(np.mean([loc[2] for loc in recent_locs])),
            int(np.mean([loc[3] for loc in recent_locs]))
        ]
    
    def increment_last_seen(self):
        self.last_seen += 1

# Dictionary to track faces
tracked_faces = {}

print("Starting recognition. Press 'q' to quit.")
speak_text("Goodmorning Grandma")

# Create named window
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

while True:
    # Increment frame number
    frame_number += 1
    
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    if not ret or frame is None:
        print("Failed to grab frame, retrying...")
        time.sleep(0.1)
        continue
    
    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    if (current_time - prev_time) > 1:
        fps = frame_count / (current_time - prev_time)
        frame_count = 0
        prev_time = current_time
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Variables to hold detection results
    face_locations = []
    face_names = []
    
    # Increment last_seen counter for all tracked faces
    for face_id in tracked_faces:
        tracked_faces[face_id].increment_last_seen()
    
    # FACE RECOGNITION SECTION
    current_faces = set()  # Track faces detected in current frame
    
    if process_face_frame % 2 == 0:  # Process faces more frequently (every 2 frames)
        # Resize frame for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                # Create a face ID based on approximate position
                face_id = f"face_{center_x//10}_{center_y//10}"
                current_faces.add(face_id)
                
                # Check if this face matches any known faces
                if len(known_face_encodings) > 0:
                    # Use a higher tolerance (0.6) for better detection of unknown faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=face_recognition_threshold)
                    name = "Unknown"  # Default to Unknown
                    confidence = 0
                    
                    # If there are matches, find the closest one
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        
                        # Only use the match if confidence is high enough
                        if confidence >= 0.55 and matches[best_match_index]:
                            # For display, include confidence
                            name = f"{known_face_names[best_match_index]} ({confidence:.2f})"
                            # For speech, use only the name
                            speech_name = known_face_names[best_match_index]
                        else:
                            name = f"Unknown ({confidence:.2f})"
                            speech_name = "Unknown person Detected"
                            speak_text(speech_name)
                    else:
                        # If no matches, explicitly mark as Unknown with confidence score
                        name = f"Unknown ({1-min(face_recognition.face_distance(known_face_encodings, face_encoding)):.2f})"
                        speech_name = "Unknown person Detected"
                        speak_text(speech_name)
                else:
                    # No known faces to compare with
                    name = "Unknown"
                    speech_name = "Unknown person Detected"
                    speak_text(speech_name)
                
                # Update the face tracker
                is_new_face = False
                if face_id in tracked_faces:
                    tracked_faces[face_id].update(face_location)
                    # Check if the recognized name has changed
                    if tracked_faces[face_id].name != name:
                        tracked_faces[face_id].name = name
                        is_new_face = True  # Treat as new if identity changed
                else:
                    tracked_faces[face_id] = TrackedFace(face_location, name, confidence)
                    is_new_face = True
                
                # Announce the face if it's new or newly identified
                if is_new_face and confidence >= 0.55:
                    # Extract just the name without confidence score for speech
                    if speech_name == "Mostafa":
                        speak_text(f"Mostafa your son")
                    elif speech_name == "Ahmed":
                        speak_text(f"Ahmed you grandson")
                    elif speech_name == "Jannat":
                        speak_text(f"Jannat")
    
    # Remove faces that haven't been seen for too long
    faces_to_remove = []
    for face_id, tracked_face in tracked_faces.items():
        if tracked_face.last_seen > max_frames_to_keep:
            faces_to_remove.append(face_id)
        elif face_id not in current_faces:
            tracked_face.increment_last_seen()
    
    for face_id in faces_to_remove:
        del tracked_faces[face_id]
    
    # OBJECT DETECTION SECTION WITH TRACKING
    current_objects = set()
    
    if process_object_frame % 3 == 0:
        try:
            # Run YOLOv8 inference
            results = model(frame, conf=object_confidence_threshold, classes=classes_to_keep)
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    center_x, center_y = x1 + w//2, y1 + h//2
                    
                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    
                    # Create object ID based on class and position
                    object_id = f"{class_name}_{center_x//50}_{center_y//50}"
                    current_objects.add(object_id)
                    
                    # Update existing tracker or create new one
                    is_new_object = False
                    if object_id in tracked_objects:
                        tracked_objects[object_id].update([x1, y1, w, h], confidence)
                    else:
                        tracked_objects[object_id] = TrackedObject(class_name, [x1, y1, w, h], confidence)
                        is_new_object = True
                    
                    # Announce the object if it's newly detected and confidence is high
                    if is_new_object and confidence >= 0.6:
                        if class_name == "Panadol":
                            speak_text(f"This is panadol take it at 7pm")
                        elif class_name == "Stopadol":
                            speak_text(f"This is Stopadol take it at 12am")
        except Exception as e:
            print(f"Error in object detection: {e}")
    
    # Update all tracked objects and remove stale ones
    objects_to_remove = []
    for object_id, tracked_obj in tracked_objects.items():
        if object_id not in current_objects:
            tracked_obj.increment_last_seen()
            if tracked_obj.last_seen > max_frames_to_keep:
                objects_to_remove.append(object_id)
    
    for object_id in objects_to_remove:
        del tracked_objects[object_id]
    
    process_face_frame += 1
    process_object_frame += 1
    
    # Display face detection results with stabilization
    for face_id, tracked_face in tracked_faces.items():
        # Get smoothed face location
        smoothed_location = tracked_face.get_smoothed_location()
        if not smoothed_location:
            continue
        
        # Scale back up face locations
        top, right, bottom, left = smoothed_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a box around the face
        color = (0, 255, 0)  # Green for known faces
        
        # Use red for unknown faces to make them more visible
        if "Unknown" in tracked_face.name:
            color = (0, 0, 255)  # Red for unknown faces
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, tracked_face.name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
    
    # Display object detection results
    for object_id, tracked_obj in tracked_objects.items():
        box = tracked_obj.get_smoothed_box()
        if not box:
            continue
            
        x, y, w, h = box
        confidence = tracked_obj.get_confidence()
        
        if confidence >= tracking_confidence_threshold:
            label = f"{tracked_obj.class_name} ({confidence:.2f})"
            color = (255, 0, 0)  # Blue for objects
            
            if tracked_obj.last_seen > 0:
                alpha = max(0.4, 1.0 - (tracked_obj.last_seen / max_frames_to_keep))
                color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display FPS
    # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak_text("Goodnight Grandma")
        time.sleep(1)  # Give time for the shutdown message to be spoken
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()