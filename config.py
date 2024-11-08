import torch

# Set device for running the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frame capture interval in seconds
CAPTURE_INTERVAL = 2  # Capture a frame every 2 seconds
