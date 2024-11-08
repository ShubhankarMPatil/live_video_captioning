import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import device

class CaptionGenerator:
    def __init__(self):
        # Load the BLIP model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    def generate_caption(self, image):
        # Prepare the image for the model
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode the generated caption
        return self.processor.decode(outputs[0], skip_special_tokens=True)
