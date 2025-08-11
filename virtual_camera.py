import numpy as np
import pygame as pg
import cv2
import math
import time


class VirtualCamera:
    def __init__(self, grid_map, epsilon=8):
        self.grid_map = grid_map
        self.epsilon = epsilon
        self.camera_view_distance = 5  # Khoảng cách camera có thể nhìn thấy (đơn vị ô lưới)
        self.dynamic_obstacles_manager = None  # Reference to dynamic obstacles manager

        # Motion tracking for unknown obstacle analysis
        self._position_history = {}  # Track position changes over time
        self._obstacle_history = {}  # Store temporal features

    def capture_image(self, robot_pos, direction):
        """Chụp ảnh xung quanh robot trong phạm vi nhìn thấy được"""
        # Tăng resolution cho GoogLeNet (224x224 minimum)
        view_width = self.camera_view_distance * 2 + 1
        view_height = self.camera_view_distance * 2 + 1

        # Tạo ảnh resolution cao cho deep learning
        high_res_epsilon = 32  # Tăng từ 8 lên 32 cho detail tốt hơn
        image = np.ones((view_height * high_res_epsilon, view_width * high_res_epsilon, 3), dtype=np.uint8) * 255

        return image

    def capture_obstacle_roi(self, obstacle_pos, obstacle_size):
        """Enhanced ROI với motion-aware texture"""
        roi_size = 224
        x, y = obstacle_pos

        # Detect motion information for texture generation
        motion_info = self._analyze_obstacle_motion(x, y)

        if motion_info['is_dynamic']:
            roi_image = self._create_motion_aware_dynamic_texture(roi_size, motion_info)
            print(f"📸 Generated MOTION-AWARE DYNAMIC texture for {obstacle_pos}")
        else:
            roi_image = self._create_static_texture(roi_size)
            print(f"📸 Generated STATIC texture for {obstacle_pos}")

        return roi_image

    def _analyze_obstacle_motion(self, x, y):
        """Analyze motion và check actual dynamic obstacle presence"""
        motion_info = {
            'is_dynamic': False,
            'velocity': (0, 0),
            'direction': 0,
            'speed': 0
        }

        # FIRST: Check if there's actually a dynamic obstacle here
        actual_dynamic_obstacle = False
        if self.dynamic_obstacles_manager and self.dynamic_obstacles_manager.obstacles:
            for obstacle in self.dynamic_obstacles_manager.obstacles:
                obstacle_center = obstacle['pos']
                distance = math.sqrt((x - obstacle_center[0])**2 + (y - obstacle_center[1])**2)
                if distance <= 1.5:  # Within obstacle area
                    actual_dynamic_obstacle = True
                    # Get actual velocity from obstacle
                    actual_velocity = obstacle.get('velocity', (0, 0))
                    motion_info['velocity'] = actual_velocity
                    motion_info['speed'] = math.sqrt(actual_velocity[0]**2 + actual_velocity[1]**2)
                    if motion_info['speed'] > 0:
                        motion_info['direction'] = math.atan2(actual_velocity[1], actual_velocity[0])
                    break

        # If there's a dynamic obstacle here, mark as dynamic regardless of apparent motion
        if actual_dynamic_obstacle:
            motion_info['is_dynamic'] = True
            print(f"🚶 CONFIRMED dynamic obstacle at ({x},{y}) - velocity: {motion_info['velocity']}")
        else:
            print(f"🪑 NO dynamic obstacle at ({x},{y}) - treating as static")

        return motion_info

    def _create_motion_aware_dynamic_texture(self, size, motion_info):
        """Create EXTREMELY distinctive dynamic texture - NEVER confuse with static"""
        img = np.ones((size, size, 3), dtype=np.uint8) * 255  # Bright background

        # HUMAN SKIN SIGNATURE - Very distinctive
        center_x, center_y = size // 2, size // 2

        # Head with realistic skin tone (KEY FEATURE)
        cv2.circle(img, (center_x, center_y - 60), 35, (240, 200, 160), -1)
        # Hair
        cv2.ellipse(img, (center_x, center_y - 80), (30, 20), 0, 0, 360, (80, 60, 40), -1)

        # Body with clothing texture
        cv2.ellipse(img, (center_x, center_y + 10), (45, 80), 0, 0, 360, (100, 150, 200), -1)

        # Arms with skin tone
        cv2.ellipse(img, (center_x - 35, center_y - 20), (12, 40), 45, 0, 360, (220, 180, 140), -1)
        cv2.ellipse(img, (center_x + 35, center_y - 20), (12, 40), -45, 0, 360, (220, 180, 140), -1)

        # Legs
        cv2.rectangle(img, (center_x - 15, center_y + 70), (center_x + 15, center_y + 120), (60, 100, 150), -1)

        # CRITICAL: Add organic texture variation (human characteristic)
        organic_noise = np.random.randint(-15, 15, size=(size, size, 3))
        img = np.clip(img.astype(np.int32) + organic_noise, 0, 255).astype(np.uint8)

        # CRITICAL: Motion blur (movement signature)
        motion_kernel = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]], dtype=np.float32)
        img = cv2.filter2D(img, -1, motion_kernel)

        # FINAL SIGNATURE: Ensure dominant colors are warm (skin-like)
        img[:, :, 0] = np.clip(img[:, :, 0] + 30, 0, 255)  # More red
        img[:, :, 1] = np.clip(img[:, :, 1] + 20, 0, 255)  # More green

        print(f"🚶 HUMAN texture: skin tones + motion blur + organic variation")
        return img
    def _create_static_texture(self, size):
        """Create EXTREMELY distinctive static texture - NEVER confuse with dynamic"""
        img = np.ones((size, size, 3), dtype=np.uint8) * 40  # Dark base

        # FURNITURE SIGNATURE - Very geometric
        # Table/chair pattern
        cv2.rectangle(img, (30, 30), (size-30, size-30), (80, 80, 80), -1)
        cv2.rectangle(img, (50, 50), (size-50, size-50), (120, 120, 120), -1)
        cv2.rectangle(img, (70, 70), (size-70, size-70), (60, 60, 60), -1)

        # Add wood grain texture (horizontal lines)
        for y in range(40, size-40, 8):
            cv2.line(img, (40, y), (size-40, y), (100, 90, 70), 2)

        # CRITICAL: Sharp geometric edges (furniture characteristic)
        cv2.rectangle(img, (30, 30), (size-30, size-30), (150, 150, 150), 4)
        cv2.rectangle(img, (50, 50), (size-50, size-50), (180, 180, 180), 2)

        # CRITICAL: Cold color palette (opposite of skin)
        img[:, :, 2] = np.clip(img[:, :, 2] - 20, 0, 255)  # Less red
        img[:, :, 1] = np.clip(img[:, :, 1] - 10, 0, 255)  # Less green

        # NO motion blur, NO organic variation
        print(f"🪑 FURNITURE texture: geometric + sharp edges + cold colors")
        return img

    def _create_dynamic_texture(self, size):
        """Dynamic obstacles: organic, varied patterns - VERY DISTINCTIVE"""
        img = np.ones((size, size, 3), dtype=np.uint8) * 180

        # VERY distinctive human pattern
        center_x, center_y = size // 2, size // 2

        # Body - elliptical with skin tone
        cv2.ellipse(img, (center_x, center_y), (40, 70), 0, 0, 360, (220, 170, 140), -1)

        # Head - circle above body
        cv2.circle(img, (center_x, center_y - 50), 25, (240, 190, 160), -1)

        # Strong motion blur (characteristic of movement)
        motion_kernel = np.array([[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]], dtype=np.float32)
        img = cv2.filter2D(img, -1, motion_kernel)

        # Add temporal noise (movement creates variation)
        noise = np.random.randint(-40, 40, size=(size, size, 3))
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        print(f"🚶 Created DYNAMIC texture with motion blur and skin tones")
        return img

    def _generate_realistic_texture(self, x, y):
        """Generate more realistic textures based on surrounding context"""
        roi_size = 224
        roi_image = np.ones((roi_size, roi_size, 3), dtype=np.uint8) * 128

        # ✅ IMPROVED: Context-aware texture generation
        # Sample larger neighborhood for context
        context_radius = 5
        static_count = 0
        dynamic_count = 0

        for i in range(-context_radius, context_radius + 1):
            for j in range(-context_radius, context_radius + 1):
                map_x, map_y = x + i, y + j
                if (0 <= map_x < len(self.grid_map.map) and
                        0 <= map_y < len(self.grid_map.map[0])):
                    cell_value = self.grid_map.map[map_x, map_y]
                    if cell_value in (1, 'o'):
                        static_count += 1
                    elif cell_value == 'd':
                        dynamic_count += 1

        # Generate texture based on context
        if static_count > dynamic_count:
            # Static-dominant area: sharp, geometric patterns
            roi_image = self._create_static_texture(roi_size)
        else:
            # Dynamic-dominant area: organic, irregular patterns
            roi_image = self._create_dynamic_texture(roi_size)

        return roi_image

    def detect_dynamic_obstacles(self, current_image, previous_image):
        """
        Phát hiện vật cản di chuyển bằng cách so sánh 2 frame liên tiếp
        Trả về danh sách các vật cản di chuyển với vị trí
        """
        if previous_image is None:
            return []

        # Thêm simple frame skipping để giảm false positive
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
        self._frame_counter += 1
        if self._frame_counter % 2 != 0:  # Skip every other frame
            return []

        # Chuyển đổi ảnh sang grayscale
        gray_current = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)

        # Tìm sự khác biệt giữa hai frame
        diff = cv2.absdiff(gray_current, gray_prev)
        _, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)  # Tăng lên 80

        # Thêm morphological operations mạnh hơn
        kernel = np.ones((7, 7), np.uint8)  # Tăng kernel size
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Áp dụng các phép biến đổi hình thái học để loại bỏ nhiễu
        kernel2 = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel2, iterations=2)
        thresh = cv2.erode(thresh, kernel2, iterations=1)

        # Tìm contour của các vật cản di chuyển
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc ra các contour quá nhỏ
        dynamic_obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) < 200:  # Tăng lên 200
                continue

            # Lấy bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Thêm check tỷ lệ width/height để loại bỏ noise
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 5 or aspect_ratio < 0.2:  # Loại bỏ shape quá dài/rộng
                continue

            # Thêm temporal consistency check
            # Chỉ accept detections có movement pattern hợp lý
            center_x, center_y = x + w // 2, y + h // 2

            # Skip nếu detection ở border (thường là artifact)
            image_h, image_w = current_image.shape[:2]
            if center_x < 20 or center_x > image_w - 20 or center_y < 20 or center_y > image_h - 20:
                continue

            # Tính vị trí tương đối so với robot
            rel_row = y // self.epsilon - self.camera_view_distance
            rel_col = x // self.epsilon - self.camera_view_distance

            dynamic_obstacles.append(((rel_row, rel_col), (w, h)))

        return dynamic_obstacles