import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from obstacle_classifier import ObstacleClassifier
from PIL import Image
import numpy as np
from setup_data import setup_training_data


class ObstacleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['static', 'dynamic']

        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)

            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train_obstacle_classifier():
    print("=== OBS CLASSIFIER TRAINING ===")
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists('data/obstacles/train/static') or len(os.listdir('data/obstacles/train/static')) == 0:
        print("Training data not found. Generating training data...")
        setup_training_data()

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Wider scale range
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),  # Add vertical flip
            transforms.RandomRotation(degrees=15),  # Add rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color augmentation
            transforms.RandomGrayscale(p=0.1),  # Occasional grayscale
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/obstacles'

    train_dataset = ObstacleDataset(
        os.path.join(data_dir, 'train'),
        transform=data_transforms['train']
    )

    val_dataset = ObstacleDataset(
        os.path.join(data_dir, 'val'),
        transform=data_transforms['val']
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Classes: {train_dataset.classes}")

    classifier = ObstacleClassifier(use_gpu=use_gpu)
    # Tăng training epochs và improve learning strategy
    classifier.train(train_loader, val_loader, num_epochs=25, learning_rate=0.001)  # 25 epochs
    print("=== TRAINING COMPLETED ===")
    print("Model saved to: models/googlenet_obstacle_classifier.pth")

    # Test final accuracy
    print("\n=== FINAL ACCURACY TEST ===")
    classifier.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(classifier.device)
            labels = labels.to(classifier.device)

            outputs = classifier.model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = 100 * correct / total
    print(f"Final Validation Accuracy: {final_accuracy:.2f}%")

    if final_accuracy >= 95:
        print("✓ EXCELLENT: Accuracy >= 95%")
    elif final_accuracy >= 90:
        print("✓ GOOD: Accuracy >= 90%")
    elif final_accuracy >= 80:
        print("⚠ ACCEPTABLE: Accuracy >= 80%")
    else:
        print("❌ POOR: Accuracy < 80% - Need more training data or adjustments")


if __name__ == "__main__":
    train_obstacle_classifier()