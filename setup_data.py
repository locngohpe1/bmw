# setup_data.py
import os
import cv2
import numpy as np
from PIL import Image


def setup_training_data():
    """Tạo synthetic 2D obstacle data cho GoogLeNet training"""

    print("Tạo 2D synthetic data cho obstacle classification...")

    # Tạo cấu trúc thư mục
    os.makedirs('data/obstacles/train/static', exist_ok=True)
    os.makedirs('data/obstacles/train/dynamic', exist_ok=True)
    os.makedirs('data/obstacles/val/static', exist_ok=True)
    os.makedirs('data/obstacles/val/dynamic', exist_ok=True)

    # Generate static obstacle patterns (furniture-like)
    for i in range(50):  # Tăng số lượng data
        img_static = create_static_pattern_2d()
        Image.fromarray(img_static).save(f'data/obstacles/train/static/static_2d_{i}.jpg')
        if i < 10:
            Image.fromarray(img_static).save(f'data/obstacles/val/static/static_2d_val_{i}.jpg')

    # Generate dynamic obstacle patterns (human-like movement)
    for i in range(50):
        img_dynamic = create_dynamic_pattern_2d()
        Image.fromarray(img_dynamic).save(f'data/obstacles/train/dynamic/dynamic_2d_{i}.jpg')
        if i < 10:
            Image.fromarray(img_dynamic).save(f'data/obstacles/val/dynamic/dynamic_2d_val_{i}.jpg')


def create_static_pattern_2d():
    """Tạo pattern cho static obstacle (furniture, walls)"""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200

    # Rectangular furniture pattern
    x_pos = np.random.randint(20, 120)
    y_pos = np.random.randint(20, 120)
    width = np.random.randint(60, 100)
    height = np.random.randint(40, 80)

    # Solid, consistent colors (furniture-like)
    color = [np.random.randint(80, 150)] * 3  # Grayscale furniture
    img[y_pos:y_pos + height, x_pos:x_pos + width] = color

    # Add sharp edges (static objects have clear boundaries)
    cv2.rectangle(img, (x_pos, y_pos), (x_pos + width, y_pos + height), (50, 50, 50), 2)

    return img


def create_dynamic_pattern_2d():
    """Tạo pattern cho dynamic obstacle (người, động vật)"""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 180

    # Human-like shape pattern
    x_center = np.random.randint(60, 164)
    y_center = np.random.randint(60, 164)

    # Body (elliptical shape)
    cv2.ellipse(img, (x_center, y_center), (25, 40), 0, 0, 360, (120, 80, 60), -1)

    # Head
    cv2.circle(img, (x_center, y_center - 35), 15, (140, 100, 80), -1)

    # Motion blur effect (characteristic of moving objects)
    kernel = np.ones((7, 3), np.float32) / 21
    img = cv2.filter2D(img, -1, kernel)

    # Color variation (movement creates lighting changes)
    noise = np.random.randint(-20, 20, size=(224, 224, 3))
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    return img


print("✅ Đã tạo xong dữ liệu mẫu để huấn luyện và test!")
print(f"📁 Static images: data/obstacles/train/static/ (50 files)")
print(f"📁 Dynamic images: data/obstacles/train/dynamic/ (50 files)")
print(f"📁 Validation data: data/obstacles/val/ folders")


if __name__ == "__main__":
    setup_training_data()