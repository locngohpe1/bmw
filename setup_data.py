# setup_data.py
import os
import cv2
import numpy as np
from PIL import Image

def setup_training_data():
    os.makedirs('data/obstacles/train/static', exist_ok=True)
    os.makedirs('data/obstacles/train/dynamic', exist_ok=True)
    os.makedirs('data/obstacles/val/static', exist_ok=True)
    os.makedirs('data/obstacles/val/dynamic', exist_ok=True)

    for i in range(200):
        img_static = create_static_pattern_2d()
        Image.fromarray(img_static).save(f'data/obstacles/train/static/static_2d_{i}.jpg')
        if i < 40:
            Image.fromarray(img_static).save(f'data/obstacles/val/static/static_2d_val_{i}.jpg')

    for i in range(200):
        img_dynamic = create_dynamic_pattern_2d()
        Image.fromarray(img_dynamic).save(f'data/obstacles/train/dynamic/dynamic_2d_{i}.jpg')
        if i < 40:
            Image.fromarray(img_dynamic).save(f'data/obstacles/val/dynamic/dynamic_2d_val_{i}.jpg')

def create_static_pattern_2d():
    img = np.ones((224, 224, 3), dtype=np.uint8) * 60

    furniture_types = np.random.randint(0, 3)

    if furniture_types == 0:
        cv2.rectangle(img, (40, 80), (184, 120), (100, 90, 80), -1)
        cv2.rectangle(img, (50, 120), (60, 160), (80, 70, 60), -1)
        cv2.rectangle(img, (174, 120), (184, 160), (80, 70, 60), -1)

    elif furniture_types == 1:
        cv2.rectangle(img, (60, 100), (164, 130), (120, 100, 80), -1)
        cv2.rectangle(img, (60, 60), (164, 100), (100, 80, 60), -1)
        cv2.rectangle(img, (65, 130), (75, 170), (80, 60, 40), -1)
        cv2.rectangle(img, (154, 130), (164, 170), (80, 60, 40), -1)

    else:
        cv2.rectangle(img, (50, 70), (174, 150), (90, 80, 70), -1)
        cv2.line(img, (112, 75), (112, 145), (60, 50, 40), 3)
        cv2.circle(img, (80, 110), 3, (150, 140, 130), -1)
        cv2.circle(img, (144, 110), 3, (150, 140, 130), -1)

    cv2.rectangle(img, (40, 60), (184, 170), (180, 180, 180), 3)

    for y in range(70, 160, 6):
        cv2.line(img, (45, y), (179, y), (110, 100, 90), 1)

    img[:, :, 2] = np.clip(img[:, :, 2] - 30, 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] - 10, 0, 255)

    return img

def create_dynamic_pattern_2d():
    img = np.ones((224, 224, 3), dtype=np.uint8) * 220

    x_center = np.random.randint(80, 144)
    y_center = np.random.randint(100, 124)

    head_color = (240, 200, 160)
    cv2.circle(img, (x_center, y_center - 60), 25, head_color, -1)

    hair_color = (60, 40, 20)
    cv2.ellipse(img, (x_center, y_center - 75), (20, 15), 0, 0, 360, hair_color, -1)

    body_colors = [(100, 150, 200), (150, 100, 200), (200, 100, 150)]
    body_color = body_colors[np.random.randint(0, 3)]
    cv2.ellipse(img, (x_center, y_center), (35, 70), 0, 0, 360, body_color, -1)

    arm_color = (220, 180, 140)
    cv2.ellipse(img, (x_center - 40, y_center - 15), (15, 35), 45, 0, 360, arm_color, -1)
    cv2.ellipse(img, (x_center + 40, y_center - 15), (15, 35), -45, 0, 360, arm_color, -1)

    leg_color = (80, 100, 120)
    cv2.rectangle(img, (x_center - 18, y_center + 50), (x_center + 18, y_center + 100), leg_color, -1)

    motion_kernel = np.array([[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]], dtype=np.float32)
    img = cv2.filter2D(img, -1, motion_kernel)

    organic_noise = np.random.randint(-25, 25, size=(224, 224, 3))
    img = np.clip(img.astype(np.int32) + organic_noise, 0, 255).astype(np.uint8)

    img[:, :, 0] = np.clip(img[:, :, 0] + 20, 0, 255)

    return img

if __name__ == "__main__":
    setup_training_data()