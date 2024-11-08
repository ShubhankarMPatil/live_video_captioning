import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Open a connection to the camera (0 for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables for frame capture timing
capture_interval = 2  # Capture a frame every 2 seconds
last_capture_time = time.time()
last_frame = None
last_caption = ""

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the current time
    current_time = time.time()

    # Check if it's time to capture a new frame
    if current_time - last_capture_time >= capture_interval:
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare the image for the model
        inputs = processor(images=rgb_frame, return_tensors="pt")

        # Generate captions
        with torch.no_grad():
            outputs = model.generate(**inputs)

        # Decode the generated caption
        last_caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Update the last capture time
        last_capture_time = current_time

        # Update the last frame
        last_frame = frame

    # If a new frame was captured, display it with the caption
    if last_frame is not None:
        # Display the caption on the last captured frame
        #cv2.putText(last_frame, last_caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the resulting frame
        cv2.imshow('Camera Feed', last_frame)
    
    # If no frame has been captured yet, just display the current frame
    else:
        cv2.imshow('Camera Feed', frame)
    
    print(last_caption)


    # Wait for a short duration to maintain 60 fps (approx. 16.67 ms per frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
