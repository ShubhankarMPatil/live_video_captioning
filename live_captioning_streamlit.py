import cv2
import time
import streamlit as st
import numpy as np
import pyttsx3
import tempfile
from PIL import Image
from caption_generator import CaptionGenerator
from config import CAPTURE_INTERVAL

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    st.set_page_config(page_title="Live Captioning", layout="wide")
    
    st.title("🎥 Live Captioning with BLIP 📝")
    pause = st.checkbox("⏸ Pause Video")
    
    # Sidebar options
    option = st.radio("Choose Input Mode:", ("📁 Upload Video", "📷 Live Camera"))
    tts_enabled = st.checkbox("🔊 Enable Text-to-Speech")
    
    caption_generator = CaptionGenerator()
    
    # Video capture setup
    cap = None
    uploaded_video_path = None
    
    if option == "📷 Live Camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            return
    else:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # Save uploaded file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_video.read())
            temp_file.close()
            uploaded_video_path = temp_file.name
            cap = cv2.VideoCapture(uploaded_video_path)
    
    last_capture_time = time.time()
    last_caption = ""
    
    stframe = st.empty()
    caption_placeholder = st.empty()
    
    current_time = time.time()
    while cap and cap.isOpened():
        if pause:
            time.sleep(0.1)  # Prevents high CPU usage while paused
            continue
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
        time.sleep(0.01)  # Allow Streamlit to refresh UI smoothly
        
        # Process caption in a separate thread to avoid frame delay
        if current_time - last_capture_time >= CAPTURE_INTERVAL and not pause:
            last_capture_time = current_time
            new_caption = caption_generator.generate_caption(rgb_frame)
            if new_caption != last_caption:
                last_caption = new_caption
                caption_placeholder.markdown(f"### 📝 {last_caption}")
                if tts_enabled:
                    speak_text(last_caption)
        
        current_time = time.time()
        if pause:
            continue
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate caption at intervals
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            new_caption = caption_generator.generate_caption(rgb_frame)
            if new_caption != last_caption:
                last_caption = new_caption
                caption_placeholder.markdown(f"### 📝 {last_caption}")
                if tts_enabled:
                    speak_text(last_caption)
            last_capture_time = current_time
        
        # Display video frame
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
