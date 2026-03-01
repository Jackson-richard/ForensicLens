import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from backend.ml.detector import get_detector
from backend.utils.logging import logger
import argparse

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_model(dataset_dir: str, epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting training on {device}...")

    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    train_dataset = datasets.ImageFolder(root=dataset_dir, transform=get_train_transform())
    
    # Calculate balanced class weights
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
        
    logger.info(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")
    
    weights = [1.0 / count for count in class_counts]
    samples_weights = [weights[label] for _, label in train_dataset.samples]
    
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    # Initialize model
    detector = get_detector()
    detector.to(device)
    
    # Ensure classification head is trainable
    for param in detector.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.classifier.parameters(), lr=learning_rate)

    detector.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits, _ = detector(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    # Save fine-tuned weights
    save_path = "backend/ml/weights/vit_4class_finetuned.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(detector.state_dict(), save_path)
    logger.info(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 4-class ForensicLens ViT Detector")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to 'train' directory containing 'Real', 'GAN-based', 'Diffusion-based', 'Face-swap' folders.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    train_model(args.dataset_dir, args.epochs, args.batch_size, args.lr)
