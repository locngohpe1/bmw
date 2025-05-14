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

        # Khởi tạo các vật cản động
        self.initialize_obstacles()

    def initialize_obstacles(self):
        """Tạo các vật cản động ban đầu ở vị trí ngẫu nhiên"""
        free_cells = []

        # Tìm các ô trống
        for row in range(len(self.grid_map.map)):
            for col in range(len(self.grid_map.map[0])):
                if self.grid_map.map[row, col] == 0:  # Ô trống
                    # Không đặt vật cản gần vị trí sạc pin
                    if math.dist((row, col), self.grid_map.battery_pos) > 5:
                        free_cells.append((row, col))

        # Tạo số lượng vật cản theo cấu hình
        for _ in range(min(self.num_obstacles, len(free_cells) // 2)):  # Giới hạn số lượng vật cản nếu không đủ ô trống
            if not free_cells:
                break

            # Chọn vị trí ngẫu nhiên
            pos = random.choice(free_cells)
            free_cells.remove(pos)

            # Tạo vận tốc ngẫu nhiên nhưng lớn hơn để dễ thấy
            velocity = (
                random.uniform(-0.15, 0.15) * self.speed_factor,
                random.uniform(-0.15, 0.15) * self.speed_factor
            )

            # Đảm bảo vận tốc không quá nhỏ
            while abs(velocity[0]) < 0.05 and abs(velocity[1]) < 0.05:
                velocity = (
                    random.uniform(-0.15, 0.15) * self.speed_factor,
                    random.uniform(-0.15, 0.15) * self.speed_factor
                )

            # Kích thước vật cản (đơn vị lưới) - có thể thay đổi ở đây
            size = 0.9  # Tăng kích thước để dễ thấy hơn (có thể điều chỉnh)

            # Màu sắc (đỏ đậm)
            color = (255, 0, 0)

            # Thêm vật cản mới
            obstacle = {
                'id': f"dyn_{self.next_id}",
                'pos': pos,
                'velocity': velocity,
                'size': size,
                'color': color,
                'exact_pos': (pos[0] + 0.5, pos[1] + 0.5)  # Vị trí chính xác (giữa ô)
            }

            self.obstacles.append(obstacle)
            # Đánh dấu vị trí ban đầu với size
            self._mark_obstacle_cells(pos, size)

            self.next_id += 1

        print(f"Created {len(self.obstacles)} dynamic obstacles")

    def _clear_obstacle_cells(self, center_pos, size):
        """Xóa tất cả cells mà vật cản chiếm"""
        radius = int(size / 2)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row, col = center_pos[0] + dr, center_pos[1] + dc
                if (0 <= row < len(self.grid_map.map) and
                        0 <= col < len(self.grid_map.map[0]) and
                        self.grid_map.map[row, col] == 'd'):
                    self.grid_map.map[row, col] = 0

    def _mark_obstacle_cells(self, center_pos, size):
        """Đánh dấu tất cả cells mà vật cản chiếm"""
        radius = int(size / 2)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row, col = center_pos[0] + dr, center_pos[1] + dc
                if (0 <= row < len(self.grid_map.map) and
                        0 <= col < len(self.grid_map.map[0]) and
                        self.grid_map.map[row, col] not in (1, 'o', 'e')):
                    self.grid_map.map[row, col] = 'd'

    def update(self, delta_time):
        """Cập nhật vị trí vật cản động theo thời gian"""
        map_width = len(self.grid_map.map[0])
        map_height = len(self.grid_map.map)

        for obstacle in self.obstacles:
            # Lưu vị trí cũ
            old_pos = obstacle['pos']
            old_exact = obstacle['exact_pos']

            # Tính vị trí mới
            new_x = old_exact[0] + obstacle['velocity'][0] * delta_time * 15  # Tăng tốc độ lên 15 lần
            new_y = old_exact[1] + obstacle['velocity'][1] * delta_time * 15

            # Kiểm tra va chạm với biên và đổi hướng nếu cần
            obstacle_radius = obstacle['size'] / 2
            if new_x - obstacle_radius < 0 or new_x + obstacle_radius >= map_height:
                obstacle['velocity'] = (-obstacle['velocity'][0], obstacle['velocity'][1])
                new_x = old_exact[0]

            if new_y - obstacle_radius < 0 or new_y + obstacle_radius >= map_width:
                obstacle['velocity'] = (obstacle['velocity'][0], -obstacle['velocity'][1])
                new_y = old_exact[1]

            # Kiểm tra va chạm với vật cản tĩnh (check area của vật cản)
            new_cell = (int(new_x), int(new_y))
            collision_with_static = False

            # Check collision for entire obstacle area
            radius = int(obstacle['size'] / 2)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    check_row, check_col = new_cell[0] + dr, new_cell[1] + dc
                    if (0 <= check_row < map_height and 0 <= check_col < map_width):
                        if self.grid_map.map[check_row, check_col] in (1, 'o'):
                            collision_with_static = True
                            break
                if collision_with_static:
                    break

            if collision_with_static and new_cell != old_pos:
                # Đổi hướng khi gặp vật cản
                obstacle['velocity'] = (-obstacle['velocity'][0], -obstacle['velocity'][1])
                new_x = old_exact[0]
                new_y = old_exact[1]
                new_cell = old_pos

            # Cập nhật vị trí
            obstacle['exact_pos'] = (new_x, new_y)
            obstacle['pos'] = (int(new_x), int(new_y))

            # Update grid map markings only if position changed
            if old_pos != obstacle['pos']:
                # Clear old position(s) based on size
                self._clear_obstacle_cells(old_pos, obstacle['size'])

                # Mark new position(s) based on size
                self._mark_obstacle_cells(obstacle['pos'], obstacle['size'])

                # In thông báo để kiểm tra
                print(f"Dynamic obstacle {obstacle['id']} (size={obstacle['size']}) moved to {obstacle['pos']}")

    def draw(self, surface):
        """Vẽ các vật cản động lên bề mặt pygame"""
        for obstacle in self.obstacles:
            # Tính tọa độ pixel
            x = obstacle['exact_pos'][1] * self.epsilon
            y = obstacle['exact_pos'][0] * self.epsilon

            # Vẽ hình tròn đại diện cho vật cản động với viền đen
            radius = int(obstacle['size'] * self.epsilon / 2)
            center_x = int(x + self.epsilon / 2)
            center_y = int(y + self.epsilon / 2)

            # Vẽ obstacle với size thực tế
            pg.draw.circle(surface, obstacle['color'], (center_x, center_y), radius)
            pg.draw.circle(surface, (0, 0, 0), (center_x, center_y), radius, 2)  # Viền đen

            # Vẽ ID để debug (optional)
            # font = pg.font.Font(None, 16)
            # text = font.render(obstacle['id'], True, (255, 255, 255))
            # surface.blit(text, (center_x - 10, center_y - 5))

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
            radius = int(obstacle['size'] / 2)
            center_pos = obstacle['pos']
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    row, col = center_pos[0] + dr, center_pos[1] + dc
                    if (0 <= row < len(self.grid_map.map) and
                            0 <= col < len(self.grid_map.map[0])):
                        positions.append((row, col))
        return positions