import cv2
import time
import numpy as np
from caption_generator import CaptionGenerator
from config import device, CAPTURE_INTERVAL

def main():
    # Initialize the caption generator
    caption_generator = CaptionGenerator()

    #Print the devide to make sure CUDA is being used
    print(device)

    # Open a connection to the camera (0 for the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize timing and caption variables
    last_capture_time = time.time()
    last_caption = ""

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get the current time
        current_time = time.time()

        # Check if it's time to capture a new frame for captioning
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate caption
            new_caption = caption_generator.generate_caption(rgb_frame)

            # Only update the caption if it has changed
            if new_caption != last_caption:
                last_caption = new_caption
                print(last_caption)  # Print only the new caption

            # Update last capture time
            last_capture_time = current_time

        # Display the frame (without caption overlay for performance)
        cv2.imshow('Camera Feed', frame)

        # Wait for 16 ms per frame to approximate 60 FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()