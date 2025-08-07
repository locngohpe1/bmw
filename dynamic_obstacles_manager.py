import pygame as pg
import numpy as np
import random
import math
from grid_map import EPSILON


class DynamicObstaclesManager:
    def __init__(self, grid_map, num_obstacles=2, speed_factor=0.5):
        self.grid_map = grid_map
        self.obstacles = []  # list of {pos, velocity, size, color, id}
        self.epsilon = EPSILON
        self.num_obstacles = num_obstacles
        self.next_id = 1
        self.speed_factor = speed_factor

        # đoạn code sửa
        self.human_icon = pg.image.load('assets/human_icon2.png')  # Load icon người
        self.human_icon = pg.transform.scale(self.human_icon, (16, 26))  # Resize icon

        # Không tự khởi tạo vật cản động nữa
        # self.initialize_obstacles()

    def initialize_obstacles(self):
        """Khởi tạo vật cản động từ danh sách manual từ grid_map"""
        # Lấy danh sách vật cản động từ grid_map nếu có
        if hasattr(self.grid_map, 'dynamic_obstacles'):
            for manual_obs in self.grid_map.dynamic_obstacles:
                pos = manual_obs['pos']

                # Tạo vận tốc ngẫu nhiên
                base_velocity = (
                    random.uniform(-0.03, 0.03),
                    random.uniform(-0.03, 0.03)
                )

                # Đảm bảo vận tốc base không quá nhỏ
                max_attempts = 30
                attempt = 0
                while (abs(base_velocity[0]) < 0.02 and abs(base_velocity[1]) < 0.02) and attempt < max_attempts:
                    base_velocity = (
                        random.uniform(-0.03, 0.03),
                        random.uniform(-0.03, 0.03)
                    )
                    attempt += 1

                # Nếu vẫn quá nhỏ, set một giá trị mặc định
                if abs(base_velocity[0]) < 0.02 and abs(base_velocity[1]) < 0.02:
                    base_velocity = (0.025, 0.025)

                # Áp dụng speed factor
                velocity = (
                    base_velocity[0] * max(self.speed_factor, 0.1),
                    base_velocity[1] * max(self.speed_factor, 0.1)
                )

                obstacle = {
                    'id': manual_obs['id'],
                    'pos': pos,
                    'velocity': velocity,
                    'size': manual_obs.get('size', 1.0),
                    'color': (255, 0, 0),
                    'exact_pos': (pos[0] + 0.5, pos[1] + 0.5)
                }

                self.obstacles.append(obstacle)
                self._mark_obstacle_cells(pos, obstacle['size'])
                self.next_id += 1

        print(f"Created {len(self.obstacles)} manual dynamic obstacles")

    def _clear_obstacle_cells(self, center_pos, size):
        """Xóa tất cả cells mà vật cản chiếm"""
        radius = int(max(size) / 2) if isinstance(size, tuple) else int(size / 2)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row, col = center_pos[0] + dr, center_pos[1] + dc
                if (0 <= row < len(self.grid_map.map) and
                        0 <= col < len(self.grid_map.map[0]) and
                        self.grid_map.map[row, col] == 'd'):
                    self.grid_map.map[row, col] = 0

    def _mark_obstacle_cells(self, center_pos, size):
        if isinstance(size, tuple):
            max_dim = max(size)
        else:
            max_dim = size
        radius = int(max_dim / 2)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row, col = center_pos[0] + dr, center_pos[1] + dc
                if (0 <= row < len(self.grid_map.map) and
                        0 <= col < len(self.grid_map.map[0]) and
                        self.grid_map.map[row, col] not in (1, 'o', 'e')):
                    self.grid_map.map[row, col] = 'd'
    pass

    def update(self, delta_time):
        """Cập nhật vị trí vật cản động theo thời gian"""
        if not self.obstacles:
            return  # Không có vật cản động để cập nhật

        map_width = len(self.grid_map.map[0])
        map_height = len(self.grid_map.map)

        for obstacle in self.obstacles:
            old_pos = obstacle['pos']
            old_exact = obstacle['exact_pos']

            # Tính vị trí mới
            new_x = old_exact[0] + obstacle['velocity'][0] * delta_time * 15
            new_y = old_exact[1] + obstacle['velocity'][1] * delta_time * 15

            size = obstacle.get('size', 1.0)
            obstacle_radius = max(size) / 2 if isinstance(size, tuple) else size / 2

            if new_x - obstacle_radius < 0 or new_x + obstacle_radius >= map_height:
                obstacle['velocity'] = (-obstacle['velocity'][0], obstacle['velocity'][1])
                new_x = old_exact[0]

            if new_y - obstacle_radius < 0 or new_y + obstacle_radius >= map_width:
                obstacle['velocity'] = (obstacle['velocity'][0], -obstacle['velocity'][1])
                new_y = old_exact[1]

            new_cell = (int(new_x), int(new_y))
            collision_with_static = False
            radius = int(max(size) / 2) if isinstance(size, tuple) else int(size / 2)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    check_row = new_cell[0] + dr
                    check_col = new_cell[1] + dc
                    if 0 <= check_row < map_height and 0 <= check_col < map_width:
                        if self.grid_map.map[check_row, check_col] in (1, 'o'):
                            collision_with_static = True
                            break
                if collision_with_static:
                    break

            if collision_with_static and new_cell != old_pos:
                obstacle['velocity'] = (-obstacle['velocity'][0], -obstacle['velocity'][1])
                new_x = old_exact[0]
                new_y = old_exact[1]
                new_cell = old_pos

            obstacle['exact_pos'] = (new_x, new_y)
            obstacle['pos'] = (int(new_x), int(new_y))

            if old_pos != obstacle['pos']:
                self._clear_obstacle_cells(old_pos, size)
                self._mark_obstacle_cells(obstacle['pos'], size)
                size_str = str(size)
                print(f"Dynamic obstacle {obstacle['id']} (size={size_str}) moved to {obstacle['pos']}")

    def draw(self, surface):
        """Vẽ các vật cản động lên bề mặt pygame"""
        for obstacle in self.obstacles:
            x = obstacle['exact_pos'][1] * self.epsilon
            y = obstacle['exact_pos'][0] * self.epsilon
            icon_w, icon_h = self.human_icon.get_size()
            draw_x = int(x + (self.epsilon - icon_w) / 2)
            draw_y = int(y + (self.epsilon - icon_h) / 2)
            surface.blit(self.human_icon, (draw_x, draw_y))

    def get_obstacle_info(self, obstacle_id):
        """Get thông tin chi tiết của vật cản theo ID"""
        for obstacle in self.obstacles:
            if obstacle['id'] == obstacle_id:
                return obstacle
        return None

    def get_all_obstacle_positions(self):
        """Trả về tất cả vị trí của các vật cản động"""
        positions = []
        for obstacle in self.obstacles:
            # Trả về tất cả cells mà vật cản chiếm
            radius = int(max(obstacle['size']) / 2) if isinstance(obstacle['size'], tuple) else int(obstacle['size'] / 2)
            center_pos = obstacle['pos']
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    row, col = center_pos[0] + dr, center_pos[1] + dc
                    if (0 <= row < len(self.grid_map.map) and
                            0 <= col < len(self.grid_map.map[0])):
                        positions.append((row, col))
        return positions