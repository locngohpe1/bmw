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
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    if not os.path.exists('data/obstacles/train/static') or len(os.listdir('data/obstacles/train/static')) == 0:
        setup_training_data()

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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

    classifier = ObstacleClassifier(use_gpu=use_gpu)
    classifier.train(train_loader, val_loader, num_epochs=10, learning_rate=0.001)


if __name__ == "__main__":
    train_obstacle_classifier()