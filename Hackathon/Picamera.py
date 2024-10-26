import cv2
import numpy as np
import pyttsx3
import tflite_runtime.interpreter as tflite

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='1.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Configure your camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the image to the expected input shape
    input_shape = input_details[0]['shape']
    image_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))

    # Preprocess the image
    input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Scores

    for i in range(len(scores)):
        if scores[i] > 0.8:  # Threshold for detection
            # Get box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                           ymin * frame.shape[0], ymax * frame.shape[0])

            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Get class name (you should have a mapping for class IDs to names)
            class_id = int(class_ids[i])
            distance = "0.5m"  # You would compute this based on your depth sensing logic
            direction = "center"  # You would compute this based on box position

            # Prepare text for TTS
            text_to_speak = f"Detected {class_id} at {distance} to the {direction}."
            speak(text_to_speak)

            # Show label on the frame
            cv2.putText(frame, f"{class_id}: {scores[i]:.2f}", (int(left), int(top - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
