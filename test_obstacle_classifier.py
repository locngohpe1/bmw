import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from obstacle_classifier import ObstacleClassifier


class ObstacleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset để huấn luyện phân loại vật cản
        root_dir: thư mục chứa 2 thư mục con 'static' và 'dynamic'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['static', 'dynamic']

        self.images = []
        self.labels = []

        # Đọc dữ liệu từ các thư mục
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

        # Đọc và chuyển đổi hình ảnh
        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train_obstacle_classifier():
    # Kiểm tra GPU
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"Using device: {device}")

    # Chuẩn bị dữ liệu
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

    # Đường dẫn đến thư mục dữ liệu
    data_dir = 'data/obstacles'

    # Tạo datasets và dataloaders
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

    # Tạo và huấn luyện mô hình
    classifier = ObstacleClassifier(use_gpu=use_gpu)
    classifier.train(train_loader, val_loader, num_epochs=10, learning_rate=0.001)

    print("Training complete!")


if __name__ == "__main__":
    train_obstacle_classifier()