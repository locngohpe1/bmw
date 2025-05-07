# setup_data.py
import os
import numpy as np
from PIL import Image


def setup_training_data():
    """Tạo cấu trúc thư mục và tạo dữ liệu mẫu để huấn luyện"""

    print("Bắt đầu tạo cấu trúc thư mục và dữ liệu mẫu...")

    # Tạo cấu trúc thư mục
    os.makedirs('data/obstacles/train/static', exist_ok=True)
    os.makedirs('data/obstacles/train/dynamic', exist_ok=True)
    os.makedirs('data/obstacles/val/static', exist_ok=True)
    os.makedirs('data/obstacles/val/dynamic', exist_ok=True)
    os.makedirs('data/obstacles/test', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Đã tạo các thư mục cần thiết.")

    # Tạo một số hình ảnh giả để test
    # Ví dụ hình ảnh tĩnh (đồ nội thất, tường, etc.)
    print("Đang tạo hình ảnh vật cản tĩnh...")
    for i in range(20):
        # Tạo hình ảnh giả cho vật cản tĩnh
        img_static = np.ones((224, 224, 3), dtype=np.uint8) * 200  # màu xám nhạt

        # Vẽ một số hình dạng ngẫu nhiên để tăng tính đa dạng
        x_pos = np.random.randint(30, 150)
        y_pos = np.random.randint(30, 150)
        width = np.random.randint(40, 100)
        height = np.random.randint(40, 100)

        # Vẽ hình chữ nhật
        img_static[y_pos:y_pos + height, x_pos:x_pos + width] = [120, 120, 120]

        # Thêm nhiễu ngẫu nhiên để tạo sự đa dạng
        noise = np.random.randint(0, 30, size=(224, 224, 3), dtype=np.uint8)
        img_static = np.clip(img_static.astype(np.int32) + noise - 15, 0, 255).astype(np.uint8)

        # Lưu hình ảnh
        Image.fromarray(img_static).save(f'data/obstacles/train/static/static_{i}.jpg')
        if i < 5:  # tạo vài hình cho validation
            Image.fromarray(img_static).save(f'data/obstacles/val/static/static_val_{i}.jpg')

    # Ví dụ hình ảnh động (người, động vật, etc.)
    print("Đang tạo hình ảnh vật cản động...")
    for i in range(20):
        # Tạo hình ảnh giả cho vật cản động
        img_dynamic = np.ones((224, 224, 3), dtype=np.uint8) * 180  # màu xám nhạt

        # Vẽ hình dạng khác với vật cản tĩnh - sử dụng màu đỏ để thể hiện vật cản động
        x_pos = np.random.randint(30, 150)
        y_pos = np.random.randint(30, 150)
        size = np.random.randint(30, 60)

        # Vẽ hình tròn
        for y in range(224):
            for x in range(224):
                if ((x - x_pos) ** 2 + (y - y_pos) ** 2) < size ** 2:
                    img_dynamic[y, x] = [200, 50, 50]  # màu đỏ

        # Thêm nhiễu ngẫu nhiên để tạo sự đa dạng
        noise = np.random.randint(0, 30, size=(224, 224, 3), dtype=np.uint8)
        img_dynamic = np.clip(img_dynamic.astype(np.int32) + noise - 15, 0, 255).astype(np.uint8)

        # Lưu hình ảnh
        Image.fromarray(img_dynamic).save(f'data/obstacles/train/dynamic/dynamic_{i}.jpg')
        if i < 5:  # tạo vài hình cho validation
            Image.fromarray(img_dynamic).save(f'data/obstacles/val/dynamic/dynamic_val_{i}.jpg')

    # Tạo một số ảnh test
    print("Đang tạo hình ảnh test...")
    for i in range(5):
        # Tạo hình vật cản tĩnh
        img_test_static = np.ones((224, 224, 3), dtype=np.uint8) * 190
        img_test_static[60:160, 60:160] = [110, 110, 110]  # hình vuông
        noise = np.random.randint(0, 20, size=(224, 224, 3), dtype=np.uint8)
        img_test_static = np.clip(img_test_static.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img_test_static).save(f'data/obstacles/test/test_static_{i}.jpg')

        # Tạo hình vật cản động
        img_test_dynamic = np.ones((224, 224, 3), dtype=np.uint8) * 170

        # Vẽ một hình tam giác
        for y in range(80, 180):
            width = int((y - 80) * 0.8)
            img_test_dynamic[y, 112 - width:112 + width] = [190, 60, 60]

        noise = np.random.randint(0, 20, size=(224, 224, 3), dtype=np.uint8)
        img_test_dynamic = np.clip(img_test_dynamic.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(img_test_dynamic).save(f'data/obstacles/test/test_dynamic_{i}.jpg')

    print("Đã tạo xong dữ liệu mẫu để huấn luyện và test!")
    print(f"- 20 hình vật cản tĩnh trong data/obstacles/train/static")
    print(f"- 20 hình vật cản động trong data/obstacles/train/dynamic")
    print(f"- 5 hình vật cản tĩnh trong data/obstacles/val/static")
    print(f"- 5 hình vật cản động trong data/obstacles/val/dynamic")
    print(f"- 10 hình test trong data/obstacles/test")


if __name__ == "__main__":
    setup_training_data()