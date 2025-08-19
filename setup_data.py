# setup_data.py
import os
import cv2
import numpy as np
from PIL import Image


def setup_training_data():
    print("=== SETUP TRAINING DATA ===")
    print("Creating directories...")
    os.makedirs('data/obstacles/train/static', exist_ok=True)
    os.makedirs('data/obstacles/train/dynamic', exist_ok=True)
    os.makedirs('data/obstacles/val/static', exist_ok=True)
    os.makedirs('data/obstacles/val/dynamic', exist_ok=True)

    # Tăng số lượng training data và diversity
    print(f"Generating 1000 STATIC training samples + 200 validation...")
    static_train_count = 0
    static_val_count = 0

    for i in range(1000):  # Tăng từ 200 lên 1000
        img_static = create_static_pattern_2d()
        Image.fromarray(img_static).save(f'data/obstacles/train/static/static_2d_{i}.jpg')
        static_train_count += 1
        if i % 200 == 0:
            print(f"  Generated {i} static training samples...")
        if i < 200:  # Tăng validation samples
            Image.fromarray(img_static).save(f'data/obstacles/val/static/static_2d_val_{i}.jpg')
            static_val_count += 1

    print(f"✓ STATIC: {static_train_count} training + {static_val_count} validation samples")
    print(f"Generating 1000 DYNAMIC training samples + 200 validation...")
    dynamic_train_count = 0
    dynamic_val_count = 0

    for i in range(1000):  # Tăng từ 200 lên 1000
        img_dynamic = create_dynamic_pattern_2d()
        Image.fromarray(img_dynamic).save(f'data/obstacles/train/dynamic/dynamic_2d_{i}.jpg')
        dynamic_train_count += 1
        if i % 200 == 0:
            print(f"  Generated {i} dynamic training samples...")
        if i < 200:  # Tăng validation samples
            Image.fromarray(img_dynamic).save(f'data/obstacles/val/dynamic/dynamic_2d_val_{i}.jpg')
            dynamic_val_count += 1

    print(f"✓ DYNAMIC: {dynamic_train_count} training + {dynamic_val_count} validation samples")

    total_train = static_train_count + dynamic_train_count
    total_val = static_val_count + dynamic_val_count

    print("=== DATA GENERATION COMPLETED ===")
    print(f"Total training samples: {total_train}")
    print(f"Total validation samples: {total_val}")

    # Sample analysis
    print("\n=== SAMPLE ANALYSIS ===")
    test_static = create_static_pattern_2d()
    test_dynamic = create_dynamic_pattern_2d()

    # Analyze color distribution
    static_avg_colors = np.mean(test_static, axis=(0, 1))
    dynamic_avg_colors = np.mean(test_dynamic, axis=(0, 1))

    print(f"Static avg colors (B,G,R): {static_avg_colors}")
    print(f"Dynamic avg colors (B,G,R): {dynamic_avg_colors}")
    print(f"Color difference (B,G,R): {dynamic_avg_colors - static_avg_colors}")

    # Check distinctiveness với threshold rõ ràng
    red_diff = dynamic_avg_colors[0] - static_avg_colors[0]  # Dynamic should have more RED
    green_diff = static_avg_colors[1] - dynamic_avg_colors[1]  # Static should have more GREEN
    blue_diff = abs(dynamic_avg_colors[2] - static_avg_colors[2])  # Should be different

    print(f"RED difference (Dynamic - Static): {red_diff:.1f}")
    print(f"GREEN difference (Static - Dynamic): {green_diff:.1f}")
    print(f"BLUE difference (absolute): {blue_diff:.1f}")

    if red_diff > 30:
        print("✓ EXCELLENT: Dynamic has much more RED (human skin tone)")
    elif red_diff > 15:
        print("✓ GOOD: Dynamic has more RED")
    else:
        print("⚠ WARNING: Need more RED in dynamic samples")

    if green_diff > 15:
        print("✓ EXCELLENT: Static has much more GREEN (furniture tone)")
    elif green_diff > 5:
        print("✓ GOOD: Static has more GREEN")
    else:
        print("⚠ WARNING: Need more GREEN in static samples")

    if blue_diff > 20:
        print("✓ GOOD: Sufficient BLUE difference")
    else:
        print("⚠ MODERATE: BLUE difference could be larger")

    # Overall distinctiveness score
    distinctiveness_score = (red_diff + green_diff + blue_diff) / 3
    print(f"\n🎯 Overall Distinctiveness Score: {distinctiveness_score:.1f}")
    if distinctiveness_score > 40:
        print("🚀 EXCELLENT: Very distinctive patterns!")
    elif distinctiveness_score > 25:
        print("✅ GOOD: Sufficiently distinctive")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Patterns may be too similar")


def create_static_pattern_2d():
    img = np.ones((224, 224, 3), dtype=np.uint8) * 60

    # Safety function để đảm bảo color values trong range 0-255
    def safe_color(color, offset=0):
        return tuple(max(0, min(255, c + offset)) for c in color)

    # Tạo nhiều variation furniture types với distinctive patterns
    furniture_types = np.random.randint(0, 6)  # Tăng từ 3 lên 6 types
    base_colors = [(100, 90, 80), (120, 100, 80), (90, 80, 70), (110, 95, 85)]
    select_color = base_colors[np.random.randint(0, len(base_colors))]

    if furniture_types == 0:  # Table
        cv2.rectangle(img, (40, 80), (184, 120), select_color, -1)
        cv2.rectangle(img, (50, 120), (60, 160), safe_color(select_color, -20), -1)
        cv2.rectangle(img, (174, 120), (184, 160), safe_color(select_color, -20), -1)

    elif furniture_types == 1:  # Chair
        cv2.rectangle(img, (60, 100), (164, 130), select_color, -1)
        cv2.rectangle(img, (60, 60), (164, 100), safe_color(select_color, -20), -1)
        cv2.rectangle(img, (65, 130), (75, 170), safe_color(select_color, -30), -1)
        cv2.rectangle(img, (154, 130), (164, 170), safe_color(select_color, -30), -1)

    elif furniture_types == 2:  # Cabinet
        cv2.rectangle(img, (50, 70), (174, 150), select_color, -1)
        cv2.line(img, (112, 75), (112, 145), safe_color(select_color, -40), 3)
        cv2.circle(img, (80, 110), 3, safe_color(select_color, 50), -1)
        cv2.circle(img, (144, 110), 3, safe_color(select_color, 50), -1)

    elif furniture_types == 3:  # Shelf với nhiều geometric patterns
        cv2.rectangle(img, (40, 60), (184, 80), select_color, -1)
        cv2.rectangle(img, (40, 100), (184, 120), select_color, -1)
        cv2.rectangle(img, (40, 140), (184, 160), select_color, -1)
        for x in range(50, 174, 30):
            cv2.line(img, (x, 60), (x, 160), safe_color(select_color, -50), 2)

    elif furniture_types == 4:  # Desk với sharp edges
        cv2.rectangle(img, (30, 90), (194, 130), select_color, -1)
        cv2.rectangle(img, (35, 130), (45, 170), safe_color(select_color, -25), -1)
        cv2.rectangle(img, (179, 130), (189, 170), safe_color(select_color, -25), -1)
        # Thêm keyboard/monitor geometric shapes
        cv2.rectangle(img, (70, 95), (120, 110), safe_color(select_color, -40), -1)
        cv2.rectangle(img, (130, 95), (170, 125), safe_color(select_color, -30), -1)

    else:  # TV stand với complex geometry
        cv2.rectangle(img, (50, 80), (174, 100), select_color, -1)
        cv2.rectangle(img, (60, 100), (164, 140), safe_color(select_color, -15), -1)
        cv2.rectangle(img, (70, 140), (154, 160), safe_color(select_color, -25), -1)
        # TV screen (very geometric)
        cv2.rectangle(img, (80, 60), (144, 80), (30, 30, 30), -1)

    # Tăng cường geometric patterns và sharp edges
    cv2.rectangle(img, (40, 60), (184, 170), (200, 200, 200), 4)  # Thicker border

    # Thêm nhiều geometric lines
    for y in range(70, 160, 4):  # Denser lines
        cv2.line(img, (45, y), (179, y), safe_color(select_color, -30), 1)

    # Thêm cross-hatch pattern cho furniture
    for x in range(50, 174, 8):
        cv2.line(img, (x, 65), (x, 165), safe_color(select_color, -40), 1)

    # Tạo COLD furniture colors: LOW Red, MAXIMUM Green, LOW Blue
    img[:, :, 2] = np.clip(img[:, :, 2] - 50, 0, 255)  # Giảm RED mạnh
    img[:, :, 0] = np.clip(img[:, :, 0] - 30, 0, 255)  # Giảm BLUE mạnh
    img[:, :, 1] = np.clip(img[:, :, 1] + 120, 0, 255)  # Tăng GREEN cực mạnh (furniture = xanh lá)

    return img


def create_dynamic_pattern_2d():
    img = np.ones((224, 224, 3), dtype=np.uint8) * 220

    # Tăng variation cho human-like patterns
    x_center = np.random.randint(70, 154)  # Wider range
    y_center = np.random.randint(90, 134)  # Wider range

    # Diverse human poses
    pose_type = np.random.randint(0, 4)

    if pose_type == 0:  # Standing human
        head_color = (240, 200, 160)
        cv2.circle(img, (x_center, y_center - 60), 25, head_color, -1)

        hair_colors = [(60, 40, 20), (80, 60, 40), (40, 30, 20), (100, 80, 60)]
        hair_color = hair_colors[np.random.randint(0, len(hair_colors))]
        cv2.ellipse(img, (x_center, y_center - 75), (22, 18), 0, 0, 360, hair_color, -1)

        body_colors = [(100, 150, 200), (150, 100, 200), (200, 100, 150), (120, 180, 100)]
        body_color = body_colors[np.random.randint(0, len(body_colors))]
        cv2.ellipse(img, (x_center, y_center), (35, 70), 0, 0, 360, body_color, -1)

        arm_color = (220, 180, 140)
        cv2.ellipse(img, (x_center - 40, y_center - 15), (15, 35), 45, 0, 360, arm_color, -1)
        cv2.ellipse(img, (x_center + 40, y_center - 15), (15, 35), -45, 0, 360, arm_color, -1)

        leg_color = (80, 100, 120)
        cv2.rectangle(img, (x_center - 18, y_center + 50), (x_center + 18, y_center + 100), leg_color, -1)

    elif pose_type == 1:  # Walking human
        head_color = (235, 195, 155)
        cv2.ellipse(img, (x_center, y_center - 55), (28, 30), 15, 0, 360, head_color, -1)  # Tilted head

        cv2.ellipse(img, (x_center - 5, y_center - 70), (25, 20), 15, 0, 360, (70, 50, 30), -1)

        cv2.ellipse(img, (x_center + 5, y_center + 5), (38, 75), -10, 0, 360, (110, 160, 210), -1)  # Tilted body

        # Arms in motion
        cv2.ellipse(img, (x_center - 45, y_center - 25), (18, 40), 60, 0, 360, (225, 185, 145), -1)
        cv2.ellipse(img, (x_center + 35, y_center - 5), (16, 38), -30, 0, 360, (225, 185, 145), -1)

        # Legs in stride
        cv2.ellipse(img, (x_center - 10, y_center + 60), (20, 50), 20, 0, 360, (85, 105, 125), -1)
        cv2.ellipse(img, (x_center + 15, y_center + 65), (18, 45), -15, 0, 360, (85, 105, 125), -1)

    elif pose_type == 2:  # Sitting human
        cv2.circle(img, (x_center, y_center - 30), 30, (245, 205, 165), -1)
        cv2.ellipse(img, (x_center, y_center - 45), (25, 15), 0, 0, 360, (65, 45, 25), -1)

        # Sitting torso
        cv2.ellipse(img, (x_center, y_center + 20), (40, 60), 0, 0, 180, (105, 155, 205), -1)

        # Arms resting
        cv2.ellipse(img, (x_center - 35, y_center), (12, 25), 0, 0, 360, (215, 175, 135), -1)
        cv2.ellipse(img, (x_center + 35, y_center), (12, 25), 0, 0, 360, (215, 175, 135), -1)

        # Legs folded
        cv2.ellipse(img, (x_center - 20, y_center + 60), (25, 35), 0, 0, 360, (75, 95, 115), -1)
        cv2.ellipse(img, (x_center + 20, y_center + 60), (25, 35), 0, 0, 360, (75, 95, 115), -1)

    else:  # Pet-like (cat/dog)
        # Body
        cv2.ellipse(img, (x_center, y_center), (45, 25), 0, 0, 360, (160, 140, 120), -1)
        # Head
        cv2.ellipse(img, (x_center - 35, y_center - 5), (20, 18), 0, 0, 360, (170, 150, 130), -1)
        # Tail
        cv2.ellipse(img, (x_center + 40, y_center - 10), (15, 35), 45, 0, 360, (150, 130, 110), -1)
        # Legs
        for leg_x in [x_center - 25, x_center - 5, x_center + 15, x_center + 35]:
            cv2.ellipse(img, (leg_x, y_center + 20), (8, 20), 0, 0, 360, (140, 120, 100), -1)

    # Tăng cường motion blur và organic variation
    motion_kernels = [
        np.array([[0.1, 0.2, 0.4, 0.2, 0.1]], dtype=np.float32),
        np.array([[0.05, 0.15, 0.3, 0.3, 0.15, 0.05]], dtype=np.float32),
        np.array([[0.1], [0.2], [0.4], [0.2], [0.1]], dtype=np.float32)  # Vertical motion
    ]
    motion_kernel = motion_kernels[np.random.randint(0, len(motion_kernels))]
    img = cv2.filter2D(img, -1, motion_kernel)

    # Tăng organic noise variation
    organic_noise = np.random.randint(-35, 35, size=(224, 224, 3))  # Tăng từ -25,25
    img = np.clip(img.astype(np.int32) + organic_noise, 0, 255).astype(np.uint8)

    # Tạo WARM human colors: HIGH Red, MINIMAL Green, MEDIUM Blue
    img[:, :, 0] = np.clip(img[:, :, 0] + 40, 0, 255)  # Tăng RED mạnh (human skin)
    img[:, :, 1] = np.clip(img[:, :, 1] - 50, 0, 255)  # Giảm GREEN mạnh (human ít xanh lá)
    img[:, :, 2] = np.clip(img[:, :, 2] + 15, 0, 255)  # BLUE trung bình

    return img


if __name__ == "__main__":
    setup_training_data()