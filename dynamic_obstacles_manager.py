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

            # Kích thước vật cản (đơn vị lưới)
            size = 0.9  # Tăng kích thước để dễ thấy hơn

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
            # Đánh dấu ngay vị trí ban đầu là vật cản động
            self.grid_map.map[pos] = 'd'

            self.next_id += 1

        print(f"Created {len(self.obstacles)} dynamic obstacles")

    def update(self, delta_time):
        """Cập nhật vị trí vật cản động theo thời gian"""
        map_width = len(self.grid_map.map[0])
        map_height = len(self.grid_map.map)

        for obstacle in self.obstacles:
            # Lưu vị trí cũ
            old_pos = obstacle['pos']
            old_exact = obstacle['exact_pos']

            # Tính vị trí mới
            new_x = old_exact[0] + obstacle['velocity'][0] * delta_time * 10  # Tăng tốc độ lên 10 lần
            new_y = old_exact[1] + obstacle['velocity'][1] * delta_time * 10

            # Kiểm tra va chạm với biên và đổi hướng nếu cần
            if new_x < 0 or new_x >= map_height:
                obstacle['velocity'] = (-obstacle['velocity'][0], obstacle['velocity'][1])
                new_x = old_exact[0]

            if new_y < 0 or new_y >= map_width:
                obstacle['velocity'] = (obstacle['velocity'][0], -obstacle['velocity'][1])
                new_y = old_exact[1]

            # Kiểm tra va chạm với vật cản tĩnh
            new_cell = (int(new_x), int(new_y))
            if new_cell != old_pos and (self.grid_map.map[new_cell] == 1 or self.grid_map.map[new_cell] == 'o'):
                # Đổi hướng khi gặp vật cản
                obstacle['velocity'] = (-obstacle['velocity'][0], -obstacle['velocity'][1])
                new_x = old_exact[0]
                new_y = old_exact[1]
                new_cell = old_pos

            # Cập nhật vị trí
            obstacle['exact_pos'] = (new_x, new_y)
            obstacle['pos'] = (int(new_x), int(new_y))

            if old_pos != obstacle['pos']:
                # Xóa dấu hiệu vật cản ở vị trí cũ
                if self.grid_map.map[old_pos] == 'd':
                    self.grid_map.map[old_pos] = 0

                # Đánh dấu vị trí mới là vật cản động CHỈ khi không phải vật cản tĩnh
                if self.grid_map.map[obstacle['pos']] not in (1, 'o', 'e'):
                    self.grid_map.map[obstacle['pos']] = 'd'

                    # In thông báo để kiểm tra
                    print(f"Dynamic obstacle moved to {obstacle['pos']}")

    def draw(self, surface):
        """Vẽ các vật cản động lên bề mặt pygame"""
        for obstacle in self.obstacles:
            # Tính tọa độ pixel
            x = obstacle['exact_pos'][1] * self.epsilon
            y = obstacle['exact_pos'][0] * self.epsilon

            # Vẽ hình tròn đại diện cho vật cản động với viền đen
            radius = int(obstacle['size'] * self.epsilon / 2)
            pg.draw.circle(surface, obstacle['color'], (int(x + self.epsilon / 2), int(y + self.epsilon / 2)), radius)
            pg.draw.circle(surface, (0, 0, 0), (int(x + self.epsilon / 2), int(y + self.epsilon / 2)), radius,
                           2)  # Viền đen

