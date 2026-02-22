import torch
import torch.nn as nn
import time
from backend.utils.logging import logger

GENERATOR_FAMILIES = ["Diffusion", "GAN", "Face-swap", "Unknown"]

class GeneratorAttributor(nn.Module):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, features):
        return self.head(features)

_attributor_instance = None

def get_attributor(in_features: int = 192) -> GeneratorAttributor: 
    global _attributor_instance
    if _attributor_instance is None:
        logger.info("Initializing Generator Attributor head")
        _attributor_instance = GeneratorAttributor(in_features)
        _attributor_instance.eval()
        _attributor_instance.to(torch.device('cpu'))
    return _attributor_instance

def attribute_generator(features: torch.Tensor):
    attributor = get_attributor()
    start_time = time.time()
    with torch.no_grad():
        logits = attributor(features)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
    inference_time = time.time() - start_time
    
    family = GENERATOR_FAMILIES[pred_idx]
    confidence = probs[0][pred_idx].item()
    return family, confidence, inference_time
