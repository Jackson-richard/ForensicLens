import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import time
from backend.utils.logging import logger

class DeepfakeDetector(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
        
        # Freeze backbone initially and train only classification head
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features), features

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

_detector_instance = None

def get_detector() -> DeepfakeDetector:
    global _detector_instance
    if _detector_instance is None:
        logger.info("Initializing CPU-optimized ViT detector")
        _detector_instance = DeepfakeDetector()
        _detector_instance.eval()
        
        _detector_instance.to(torch.device('cpu'))
    return _detector_instance

def predict_image(image: Image.Image):
    detector = get_detector()
    transform = get_inference_transform()
    input_tensor = transform(image).unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        logits, features = detector(input_tensor)
        
        # Apply softmax during inference
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    inference_time = time.time() - start_time
    
    classes = ["Real", "GAN-based", "Diffusion-based", "Face-swap"]
    predicted_class = classes[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100.0, 2)
    
    if predicted_class == "Real":
        response = {
            "authenticity": "Authentic",
            "synthetic_type": None,
            "confidence": confidence_pct
        }
    else:
        response = {
            "authenticity": "Synthetic",
            "synthetic_type": predicted_class,
            "confidence": confidence_pct
        }
    
    return response, features, inference_time
