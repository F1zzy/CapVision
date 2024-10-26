import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# All 80 Class Names for COCO dataset
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Load the TFLite model and allocate tensors
model_path = '1.tflite'  # Update this to your model path
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Webcam 
cap = cv2.VideoCapture(1) #1 on Laptop, 0 on Raspberry Pi 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Known width of the object in meters (e.g., average person)
KNOWN_WIDTH = 0.5  # width of a person in meters

# Focal length (in pixels) can be estimated or calibrated
FOCAL_LENGTH = 700  # adjust this value based on your camera setup

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    # Resize to model's expected input size (320x320)
    frame_resized = cv2.resize(frame, (320, 320))
    
    # Normalize to [0, 255] and convert to UINT8
    frame_uint8 = np.clip(frame_resized, 0, 255).astype(np.uint8)  # Ensure values are between 0 and 255
    return np.expand_dims(frame_uint8, axis=0)  # Add batch dimension

# Function to speack dection details 
def announce_detection(class_name, distance, direction):
    # Prepare the announcement string
    announcement = f"{class_name} {distance:.1f} meters {direction.strip()}."
    print(announcement)  # Print to console for debugging

    # Create a thread for speaking the announcement
    threading.Thread(target=speak, args=(announcement,)).start()

def speak(announcement):
    engine.say(announcement)  # Convert text to speech
    engine.runAndWait()  # Wait for the speech to finish

# Dictionary to track last known distance of detected objects
last_known_distances = {}

# Main loop to read camera and display processed frames for testing purposes. TURN OFF for RASP version 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess_frame(frame)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    # Get the output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    # Calculate frame center
    frame_height, frame_width, _ = frame.shape
    frame_center = (frame_width // 2, frame_height // 2)

    # Draw bounding boxes and labels on the frame
    for i in range(len(scores)):
        if scores[i] > 0.7:  # Confidence threshold set to 80%
            ymin, xmin, ymax, xmax = boxes[i]
            start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))

            # Draw bounding box
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

            # Get the class name from the class ID
            class_id = int(class_ids[i])
            class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

            # Calculate object width in pixels
            object_width_pixels = end_point[0] - start_point[0]

            # Calculate distance using the known width and focal length
            if object_width_pixels > 0:  # Prevent division by zero
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / object_width_pixels
            else:
                distance = float('inf')  # Set to infinity if calculation fails

            # Calculate the centroid of the bounding box
            centroid_x = (start_point[0] + end_point[0]) // 2
            centroid_y = (start_point[1] + end_point[1]) // 2

            # Determine direction
            direction = ""
            middle_threshold = 30  

            if abs(centroid_y - frame_center[1]) < middle_threshold:
                direction += "Middle "
            else:
                if centroid_y < frame_center[1] - middle_threshold:
                    direction += "Above "
                elif centroid_y > frame_center[1] + middle_threshold:
                    direction += "Below "

            if abs(centroid_x - frame_center[0]) < middle_threshold:
                direction += "Middle"
            else:
                if centroid_x < frame_center[0] - middle_threshold:
                    direction += "Left "
                elif centroid_x > frame_center[0] + middle_threshold:
                    direction += "Right"

            # Labeling for Boxes 
            distance_label = f"Distance: {distance:.2f} m"
            label = f"{class_name}: {scores[i]:.2f} | {distance_label} | Direction: {direction.strip()}"
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check if the object is new or within 0.3m
            if class_id not in last_known_distances or distance < 0.5:
                announce_detection(class_name, distance, direction)
                last_known_distances[class_id] = distance  # Update the last known distance

    # Remove objects that are no longer detected
    current_ids = set(int(class_ids[i]) for i in range(len(scores)) if scores[i] > 0.8)
    for obj_id in list(last_known_distances.keys()):
        if obj_id not in current_ids:
            del last_known_distances[obj_id]  # Remove object from tracking if not detected

    # Display the frame with detections
    cv2.imshow('Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
