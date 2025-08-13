import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os


class ObstacleClassifier:
    def __init__(self, use_gpu=True):
        self.device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = models.googlenet(pretrained=True)

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        model_path = "models/googlenet_obstacle_classifier.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['static', 'dynamic']

    def classify(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        class_name = self.classes[predicted.item()]
        confidence = probabilities[predicted.item()].item()

        return class_name, confidence

    def detect_and_classify_from_image(self, image, confidence_threshold=0.7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            obstacle_img = image[y:y + h, x:x + w]
            class_name, confidence = self.classify(obstacle_img)

            if confidence > confidence_threshold:
                center_pos = (x + w // 2, y + h // 2)
                obstacles.append((center_pos, class_name, confidence))

        return obstacles

    def train(self, train_dataloader, val_dataloader, num_epochs=10, learning_rate=0.001):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Learning rate: {learning_rate}")
        print("-" * 50)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if epoch_start_time:
                epoch_start_time.record()

            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataloader.dataset)

            self.model.eval()
            val_accuracy = 0.0
            val_loss = 0.0
            correct_predictions = 0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    val_accuracy += torch.sum(predictions == labels).item()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    correct_predictions += torch.sum(predictions == labels).item()

            val_accuracy = val_accuracy / len(val_dataloader.dataset)
            val_loss = val_loss / len(val_dataloader.dataset)

            # Print epoch results
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"  Train Loss: {epoch_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

            if epoch_start_time and torch.cuda.is_available():
                epoch_end_time = torch.cuda.Event(enable_timing=True)
                epoch_end_time.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
                print(f"  Epoch Time: {epoch_time:.2f}s")

            print("-" * 50)

        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.model.state_dict(), "models/googlenet_obstacle_classifier.pth")