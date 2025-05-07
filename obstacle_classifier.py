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
        print(f"Using device: {self.device}")

        # Tải mô hình GoogLeNet đã được huấn luyện trước
        self.model = models.googlenet(pretrained=True)

        # Điều chỉnh lớp output cho 2 lớp (tĩnh/động)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        # Tải trọng số nếu có
        model_path = "models/googlenet_obstacle_classifier.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Loaded pretrained obstacle classifier model")

        self.model.to(self.device)
        self.model.eval()

        # Tiền xử lý hình ảnh
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Hai lớp: vật cản tĩnh (0) và vật cản động (1)
        self.classes = ['static', 'dynamic']

    def classify(self, image):
        """Phân loại vật cản là tĩnh hay động từ hình ảnh"""
        if isinstance(image, np.ndarray):
            # Chuyển từ mảng numpy sang PIL Image
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
        """
        Phát hiện vật cản trong ảnh và phân loại chúng
        Trả về danh sách (vị trí, loại, độ tin cậy)
        """
        # Chuyển đổi ảnh sang grayscale để phát hiện vật cản
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        # Tìm contour của vật cản
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for contour in contours:
            # Lọc ra các contour quá nhỏ
            if cv2.contourArea(contour) < 100:
                continue

            # Lấy bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Cắt vật cản từ ảnh gốc
            obstacle_img = image[y:y + h, x:x + w]

            # Phân loại vật cản
            class_name, confidence = self.classify(obstacle_img)

            if confidence > confidence_threshold:
                center_pos = (x + w // 2, y + h // 2)
                obstacles.append((center_pos, class_name, confidence))

        return obstacles

    def train(self, train_dataloader, val_dataloader, num_epochs=10, learning_rate=0.001):
        """Huấn luyện hoặc tinh chỉnh mô hình với dữ liệu vật cản cụ thể"""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

        for epoch in range(num_epochs):
            # Train phase
            self.model.train()
            running_loss = 0.0

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
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

            # Validation phase
            self.model.eval()
            val_accuracy = 0.0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    val_accuracy += torch.sum(predictions == labels).item()

            val_accuracy = val_accuracy / len(val_dataloader.dataset)
            print(f'Validation Accuracy: {val_accuracy:.4f}')

        # Lưu mô hình đã huấn luyện
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.model.state_dict(), "models/googlenet_obstacle_classifier.pth")
        print("Model saved")