# File: live_captioning_gui.py

import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk
from caption_generator import CaptionGenerator
from config import CAPTURE_INTERVAL, device

def main():
    # Initialize the caption generator
    caption_generator = CaptionGenerator()

    # Open a connection to the camera (0 for the default camera)
    cap = cv2.VideoCapture("sample_videos/videoplayback.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize timing and caption variables
    last_capture_time = time.time()
    last_caption = ""

    # Set up Tkinter window
    root = tk.Tk()
    root.title("Live Captioning with BLIP")

    # Video label for displaying camera feed
    video_label = tk.Label(root)
    video_label.pack()

    # Caption label for displaying the caption
    caption_label = tk.Label(root, text="", font=("Helvetica", 16), wraplength=400)
    caption_label.pack()

    def update_frame():
        nonlocal last_capture_time, last_caption

        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            root.after(10, update_frame)
            return

        # Get the current time
        current_time = time.time()

        # Check if it's time to capture a new frame for captioning
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            # Convert frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate caption
            new_caption = caption_generator.generate_caption(rgb_frame)

            # Only update the caption if it has changed
            if new_caption != last_caption:
                last_caption = new_caption
                caption_label.config(text=last_caption)

            # Update the last capture time
            last_capture_time = current_time

        # Display the frame in Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Schedule the next frame update
        root.after(16, update_frame)  # Approx. 60 FPS

    # Start the Tkinter frame update loop
    update_frame()

    # Run the Tkinter main loop
    root.mainloop()

    # Release the camera and close all OpenCV windows when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
