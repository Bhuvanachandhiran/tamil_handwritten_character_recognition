import torch
import cv2
import numpy as np
from model import TamilNet

class TamilOCR:
    def __init__(self, model_path, char_map_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TamilNet(num_classes=156).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.char_map = self.load_char_map(char_map_path)

    def load_char_map(self, char_map_path):
        char_map = {}
        with open(char_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                index, char = line.strip().split(':')
                char_map[int(index)] = char
        return char_map

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        image = cv2.resize(image, (32, 32))  # Updated to 32x32
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return torch.tensor(image, dtype=torch.float32).to(self.device)

    def predict(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()
        return self.char_map.get(predicted_idx, "Unknown")