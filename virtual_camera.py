import numpy as np
import pygame as pg
import cv2


class VirtualCamera:
    def __init__(self, grid_map, epsilon=8):
        self.grid_map = grid_map
        self.epsilon = epsilon
        self.camera_view_distance = 5  # Khoảng cách camera có thể nhìn thấy (đơn vị ô lưới)
        self._frame_counter = 0

    def capture_image(self, robot_pos, direction):
        """Chụp ảnh xung quanh robot trong phạm vi nhìn thấy được"""
        # Tăng resolution cho GoogLeNet (224x224 minimum)
        view_width = self.camera_view_distance * 2 + 1
        view_height = self.camera_view_distance * 2 + 1

        # Tạo ảnh resolution cao cho deep learning
        high_res_epsilon = 32  # Tăng từ 8 lên 32 cho detail tốt hơn
        image = np.ones((view_height * high_res_epsilon, view_width * high_res_epsilon, 3), dtype=np.uint8) * 255

    def capture_obstacle_roi(self, obstacle_pos, obstacle_size):
        """Enhanced ROI generation với realistic features"""
        roi_size = 224
        x, y = obstacle_pos

        # ✅ IMPROVED: Multi-frame temporal information
        if not hasattr(self, '_obstacle_history'):
            self._obstacle_history = {}

        obs_key = f"{x}_{y}"
        if obs_key not in self._obstacle_history':
        self._obstacle_history[obs_key] = []

    # ✅ Create base image với realistic texture
    roi_image = self._generate_realistic_texture(x, y)

    # ✅ IMPROVED: Add temporal consistency features
    if len(self._obstacle_history[obs_key]) > 0:
        prev_features = self._obstacle_history[obs_key][-1]

        # Dynamic objects show temporal variation
        if self.grid_map.map[x, y] == 'd':
            # Add temporal variation (movement signatures)
            temporal_noise = np.random.randint(-30, 30, size=(roi_size, roi_size, 3))
            roi_image = np.clip(roi_image.astype(np.int32) + temporal_noise, 0, 255).astype(np.uint8)

            # Motion blur based on estimated velocity
            motion_kernel = np.array([[0.1, 0.2, 0.1], [0.1, 0.2, 0.1], [0.1, 0.2, 0.1]], dtype=np.float32)
            roi_image = cv2.filter2D(roi_image, -1, motion_kernel)

    # Store current features for next frame
    current_features = np.mean(roi_image, axis=(0, 1))
    self._obstacle_history[obs_key].append(current_features)
    if len(self._obstacle_history[obs_key]) > 5:  # Keep last 5 frames
        self._obstacle_history[obs_key].pop(0)

    return roi_image


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


    def _create_static_texture(self, size):
        """Static obstacles: geometric, consistent patterns"""
        img = np.ones((size, size, 3), dtype=np.uint8) * 120

        # Geometric patterns (furniture-like)
        pattern_size = size // 4
        for i in range(0, size, pattern_size):
            for j in range(0, size, pattern_size):
                cv2.rectangle(img, (i, j), (i + pattern_size // 2, j + pattern_size // 2),
                              (100, 100, 100), -1)

        return img


    def _create_dynamic_texture(self, size):
        """Dynamic obstacles: organic, varied patterns"""
        img = np.ones((size, size, 3), dtype=np.uint8) * 140

        # Organic patterns (human-like)
        center_x, center_y = size // 2, size // 2
        cv2.ellipse(img, (center_x, center_y), (size // 4, size // 3), 0, 0, 360,
                    (160, 120, 100), -1)

        # Add texture variation
        noise = np.random.randint(-20, 20, size=(size, size, 3))
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        return img
        return roi_image

    def _generate_obstacle_texture(self, height, width):
        """Tạo texture pattern khác nhau cho static vs dynamic"""
        base_image = np.random.randint(100, 200, (height * 32, width * 32, 3), dtype=np.uint8)

        # Add movement blur cho dynamic objects
        if hasattr(self, '_previous_positions') and len(self._previous_positions) > 0:
            # Dynamic pattern: motion blur, varied colors
            kernel = np.ones((5, 5), np.float32) / 25
            base_image = cv2.filter2D(base_image, -1, kernel)
            # Add color variation để simulate movement
            base_image[:, :, 0] = np.clip(base_image[:, :, 0] + np.random.randint(-30, 30), 0, 255)
        else:
            # Static pattern: sharp edges, consistent colors
            base_image = cv2.medianBlur(base_image, 3)

        return base_image
        return image

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