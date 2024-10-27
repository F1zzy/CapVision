import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import queue
import time

# Constant for buffer time in seconds
BUFFER_TIME = 1.5  # Change this value to set the desired buffer time
DISTANCE_THRESHOLD = 0.3  # Minimum distance change to trigger a new announcement in meters

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 275)  # Set TTS speed

tts_lock = threading.Lock()  # Lock for the TTS engine to avoid concurrent access

# Queue for announcements
announcement_queue = queue.Queue()

# Define the class names we want to detect
class_names = ["person", "chair", "table", "dining table"]
class_ids_to_detect = [0, 62, 60, 86]  # Class IDs for person, chair, table

# Load the TFLite model and allocate tensors
model_path = '1.tflite'  # Update this to your model path
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Webcam
cap = cv2.VideoCapture(0)  # 0 on Laptop , 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Connected To Camera")

# Known width of the object in meters (e.g., average person)
KNOWN_WIDTH = 0.5  # width of a person in meters
FOCAL_LENGTH = 700  # adjust this value based on your camera setup
THRESHOLD_DISTANCE = 10  # Announcement threshold in meters

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (320, 320))
    frame_uint8 = np.clip(frame_resized, 0, 255).astype(np.uint8)
    return np.expand_dims(frame_uint8, axis=0)

# Function to handle speaking the announcements from the queue
def handle_announcements():
    while True:
        announcement = announcement_queue.get()
        with tts_lock:
            if engine.isBusy():
                engine.stop()  # Interrupt the current speech
            engine.say(announcement)  # Say the new announcement
            engine.runAndWait()
        announcement_queue.task_done()

# Start the thread to handle announcements
announcement_thread = threading.Thread(target=handle_announcements, daemon=True)
announcement_thread.start()

# Initialize variable to track the last announcement in the main loop
last_announcement = None
last_announcement_time = 0
last_announcement_distance = None

print("Running")
detected_objects = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    frame_height, frame_width, _ = frame.shape
    frame_center = (frame_width // 2, frame_height // 2)

    current_detections = {}

    for i in range(len(scores)):
        if scores[i] > 0.6:  # Confidence threshold
            class_id = int(class_ids[i])
            if class_id in class_ids_to_detect:
                ymin, xmin, ymax, xmax = boxes[i]
                start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
                end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

                class_name = class_names[class_ids_to_detect.index(class_id)]
                object_width_pixels = end_point[0] - start_point[0]
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / object_width_pixels if object_width_pixels > 0 else float('inf')

                if distance <= THRESHOLD_DISTANCE:
                    current_detections[class_id] = {
                        "class_name": class_name,
                        "distance": distance,
                        "bounding_box": (start_point, end_point),
                        "centroid": ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                    }

    closest_object = None
    if current_detections:
        closest_object = min(current_detections.values(), key=lambda x: x['distance'])

    if closest_object:
        centroid = closest_object['centroid']
        direction = ""
        middle_threshold = 30
        direction += "Front " if abs(centroid[1] - frame_center[1]) < middle_threshold else " "
        direction += "Middle" if abs(centroid[0] - frame_center[0]) < middle_threshold else \
                     "Left " if centroid[0] < frame_center[0] - middle_threshold else "Right"

        announcement = f"{closest_object['class_name']} {closest_object['distance']:.1f} meters {direction.strip()}."

        current_time = time.time()
        distance_change = abs(closest_object['distance'] - last_announcement_distance) if last_announcement_distance else float('inf')
        
        # Check if the announcement criteria are met
        if (last_announcement != announcement) and \
           (current_time - last_announcement_time >= BUFFER_TIME) and \
           (distance_change >= DISTANCE_THRESHOLD):
            print(announcement)
            announcement_queue.put(announcement)
            last_announcement = announcement
            last_announcement_time = current_time
            last_announcement_distance = closest_object['distance']


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
