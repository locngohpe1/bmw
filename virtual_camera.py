import numpy as np
import pygame as pg
import cv2
import math
import time
class VirtualCamera:
    def __init__(self, grid_map, epsilon=8):
        self.grid_map = grid_map
        self.epsilon = epsilon
        self.camera_view_distance = 5
        self.dynamic_obstacles_manager = None
        self._position_history = {}
        self._obstacle_history = {}

    def capture_image(self, robot_pos, direction):
        view_width = self.camera_view_distance * 2 + 1
        view_height = self.camera_view_distance * 2 + 1
        high_res_epsilon = 32
        image = np.ones((view_height * high_res_epsilon, view_width * high_res_epsilon, 3), dtype=np.uint8) * 255
        return image

    def capture_obstacle_roi(self, obstacle_pos, obstacle_size):
        roi_size = 224
        x, y = obstacle_pos

        motion_info = self._analyze_obstacle_motion(x, y)

        if motion_info['is_dynamic']:
            roi_image = self._create_motion_aware_dynamic_texture(roi_size, motion_info)
        else:
            roi_image = self._create_static_texture(roi_size)

        return roi_image

    def _analyze_obstacle_motion(self, x, y):
        motion_info = {
            'is_dynamic': False,
            'velocity': (0, 0),
            'direction': 0,
            'speed': 0
        }

        actual_dynamic_obstacle = False
        if self.dynamic_obstacles_manager and self.dynamic_obstacles_manager.obstacles:
            for obstacle in self.dynamic_obstacles_manager.obstacles:
                obstacle_center = obstacle['pos']
                distance = math.sqrt((x - obstacle_center[0])**2 + (y - obstacle_center[1])**2)
                if distance <= 1.5:
                    actual_dynamic_obstacle = True
                    actual_velocity = obstacle.get('velocity', (0, 0))
                    motion_info['velocity'] = actual_velocity
                    motion_info['speed'] = math.sqrt(actual_velocity[0]**2 + actual_velocity[1]**2)
                    if motion_info['speed'] > 0:
                        motion_info['direction'] = math.atan2(actual_velocity[1], actual_velocity[0])
                    break

        if actual_dynamic_obstacle:
            motion_info['is_dynamic'] = True

        return motion_info

    def _create_motion_aware_dynamic_texture(self, size, motion_info):
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        center_x, center_y = size // 2, size // 2

        cv2.circle(img, (center_x, center_y - 60), 35, (240, 200, 160), -1)
        cv2.ellipse(img, (center_x, center_y - 80), (30, 20), 0, 0, 360, (80, 60, 40), -1)
        cv2.ellipse(img, (center_x, center_y + 10), (45, 80), 0, 0, 360, (100, 150, 200), -1)
        cv2.ellipse(img, (center_x - 35, center_y - 20), (12, 40), 45, 0, 360, (220, 180, 140), -1)
        cv2.ellipse(img, (center_x + 35, center_y - 20), (12, 40), -45, 0, 360, (220, 180, 140), -1)
        cv2.rectangle(img, (center_x - 15, center_y + 70), (center_x + 15, center_y + 120), (60, 100, 150), -1)

        organic_noise = np.random.randint(-15, 15, size=(size, size, 3))
        img = np.clip(img.astype(np.int32) + organic_noise, 0, 255).astype(np.uint8)

        motion_kernel = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]], dtype=np.float32)
        img = cv2.filter2D(img, -1, motion_kernel)

        img[:, :, 0] = np.clip(img[:, :, 0] + 30, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] + 20, 0, 255)

        return img

    def _create_static_texture(self, size):
        img = np.ones((size, size, 3), dtype=np.uint8) * 40

        cv2.rectangle(img, (30, 30), (size-30, size-30), (80, 80, 80), -1)
        cv2.rectangle(img, (50, 50), (size-50, size-50), (120, 120, 120), -1)
        cv2.rectangle(img, (70, 70), (size-70, size-70), (60, 60, 60), -1)

        for y in range(40, size-40, 8):
            cv2.line(img, (40, y), (size-40, y), (100, 90, 70), 2)

        cv2.rectangle(img, (30, 30), (size-30, size-30), (150, 150, 150), 4)
        cv2.rectangle(img, (50, 50), (size-50, size-50), (180, 180, 180), 2)

        img[:, :, 2] = np.clip(img[:, :, 2] - 20, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] - 10, 0, 255)

        return img

    def _create_dynamic_texture(self, size):
        img = np.ones((size, size, 3), dtype=np.uint8) * 180
        center_x, center_y = size // 2, size // 2

        cv2.ellipse(img, (center_x, center_y), (40, 70), 0, 0, 360, (220, 170, 140), -1)
        cv2.circle(img, (center_x, center_y - 50), 25, (240, 190, 160), -1)

        motion_kernel = np.array([[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]], dtype=np.float32)
        img = cv2.filter2D(img, -1, motion_kernel)

        noise = np.random.randint(-40, 40, size=(size, size, 3))
        img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

        return img

    def detect_dynamic_obstacles(self, current_image, previous_image):
        if previous_image is None:
            return []

        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
        self._frame_counter += 1
        if self._frame_counter % 2 != 0:
            return []

        gray_current = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_current, gray_prev)
        _, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)

        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        kernel2 = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel2, iterations=2)
        thresh = cv2.erode(thresh, kernel2, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dynamic_obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) < 200:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue

            center_x, center_y = x + w // 2, y + h // 2

            image_h, image_w = current_image.shape[:2]
            if center_x < 20 or center_x > image_w - 20 or center_y < 20 or center_y > image_h - 20:
                continue

            rel_row = y // self.epsilon - self.camera_view_distance
            rel_col = x // self.epsilon - self.camera_view_distance

            dynamic_obstacles.append(((rel_row, rel_col), (w, h)))

        return dynamic_obstacles