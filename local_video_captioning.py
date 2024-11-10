import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import numpy as np
from summarizer import generate_paragraph # importing the summarizer module

# Initialize BLIP processor and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def process_video(video_path, frame_change_threshold=30.0):
    captions = []
    cap = cv2.VideoCapture(video_path)

    # Calculate the frame limit based on max_duration (in seconds)
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Video")

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the video.")
        return []

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for comparison
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
        diff_score = np.mean(frame_diff)

        # Process frame if there is a significant change
        if diff_score > frame_change_threshold:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            caption = generate_caption(rgb_frame)
            captions.append(caption)

            # Update previous frame to the current one
            prev_frame_gray = frame_gray

        pbar.update(1)  # Update progress bar for each frame processed

    cap.release()
    pbar.close()
    return captions

# Main function
video_path = "sample_videos/videoplayback.mp4"
captions = process_video(video_path, frame_change_threshold=30.0)
summary = generate_paragraph(captions)
print("Video Summary:")
print(summary)
