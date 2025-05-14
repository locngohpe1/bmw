import math
import numpy as np
import time


class DynamicObstacleHandler:
    def __init__(self):
        self.dynamic_obstacles = {}  # Dictionary lưu thông tin vật cản động {id: {pos, velocity, size, ...}}
        self.last_update_time = time.time()

    def register_obstacle(self, obstacle_id, position, velocity=None):
        """Đăng ký một vật cản động mới"""
        current_time = time.time()

        if velocity is None:
            # Nếu không có thông tin vận tốc, giả định vận tốc là 0
            velocity = (0, 0)

        self.dynamic_obstacles[obstacle_id] = {
            'position': position,
            'velocity': velocity,
            'size': 1.0,  # Default size
            'history': [(position, current_time)],
            'last_seen': current_time
        }

    def update_obstacle(self, obstacle_id, new_position):
        """Cập nhật vị trí vật cản và tính vận tốc"""
        current_time = time.time()

        if obstacle_id in self.dynamic_obstacles:
            old_position = self.dynamic_obstacles[obstacle_id]['position']
            old_time = self.dynamic_obstacles[obstacle_id]['last_seen']

            # Tính vận tốc dựa trên khoảng thời gian
            time_diff = current_time - old_time
            if time_diff > 0:
                velocity = (
                    (new_position[0] - old_position[0]) / time_diff,
                    (new_position[1] - old_position[1]) / time_diff
                )
            else:
                velocity = self.dynamic_obstacles[obstacle_id]['velocity']

            # Cập nhật thông tin
            self.dynamic_obstacles[obstacle_id]['position'] = new_position
            self.dynamic_obstacles[obstacle_id]['velocity'] = velocity
            self.dynamic_obstacles[obstacle_id]['history'].append((new_position, current_time))
            self.dynamic_obstacles[obstacle_id]['last_seen'] = current_time

    def remove_old_obstacles(self, max_age=5.0):
        """Xóa các vật cản quá lâu không được cập nhật"""
        current_time = time.time()
        obstacles_to_remove = []

        for obstacle_id, data in self.dynamic_obstacles.items():
            age = current_time - data['last_seen']
            if age > max_age:
                obstacles_to_remove.append(obstacle_id)
                # Optional: Log removed obstacles
                # print(f"Removed aged obstacle {obstacle_id}, age: {age:.1f}s")

        for obstacle_id in obstacles_to_remove:
            del self.dynamic_obstacles[obstacle_id]

    def predict_collision(self, robot_pos, robot_direction, robot_speed, obstacle_id):
        """Dự đoán có va chạm với vật cản động không"""
        if obstacle_id not in self.dynamic_obstacles:
            return False, None

        obstacle = self.dynamic_obstacles[obstacle_id]
        obstacle_pos = obstacle['position']
        obstacle_vel = obstacle['velocity']

        # Đơn vị hóa vector hướng robot
        dir_norm = math.sqrt(robot_direction[0] ** 2 + robot_direction[1] ** 2)
        if dir_norm < 1e-6:  # Tránh chia cho 0
            return False, None

        norm_dir = (robot_direction[0] / dir_norm, robot_direction[1] / dir_norm)

        # Vận tốc robot theo vector đơn vị
        robot_vel = (norm_dir[0] * robot_speed, norm_dir[1] * robot_speed)

        # Vectơ tương đối giữa robot và vật cản
        rel_vel = (robot_vel[0] - obstacle_vel[0], robot_vel[1] - obstacle_vel[1])

        # Nếu vận tốc tương đối quá nhỏ, coi như không có va chạm
        rel_speed = math.sqrt(rel_vel[0] ** 2 + rel_vel[1] ** 2)
        if rel_speed < 1e-6:
            return False, None

        # Vector từ robot đến vật cản
        rel_pos = (obstacle_pos[0] - robot_pos[0], obstacle_pos[1] - robot_pos[1])

        # Tính thời gian đến gần nhất
        t_closest = -(rel_pos[0] * rel_vel[0] + rel_pos[1] * rel_vel[1]) / (rel_vel[0] ** 2 + rel_vel[1] ** 2)

        # Nếu thời gian âm, vật cản đang đi xa khỏi robot
        if t_closest < 0:
            return False, None

        # Tính khoảng cách gần nhất
        closest_pos = (
            robot_pos[0] + robot_vel[0] * t_closest,
            robot_pos[1] + robot_vel[1] * t_closest
        )

        obstacle_future_pos = (
            obstacle_pos[0] + obstacle_vel[0] * t_closest,
            obstacle_pos[1] + obstacle_vel[1] * t_closest
        )

        # Tính khoảng cách gần nhất
        closest_distance = math.sqrt(
            (closest_pos[0] - obstacle_future_pos[0]) ** 2 +
            (closest_pos[1] - obstacle_future_pos[1]) ** 2
        )

        # Ngưỡng khoảng cách an toàn - tính theo size của vật cản
        obstacle_size = obstacle.get('size', 1.0)
        robot_size = 1.0  # Assume robot cũng có size
        safety_distance = (obstacle_size + robot_size) / 2 + 0.5  # Buffer thêm 0.5

        if closest_distance < safety_distance:
            # Sẽ có va chạm
            return True, (closest_pos, t_closest)

        return False, None

    def apply_waiting_rule(self, robot_pos, robot_direction, robot_speed):
        """Áp dụng waiting rule cho robot"""
        min_time_to_collision = float('inf')
        collision_point = None
        colliding_obstacle = None

        # Kiểm tra va chạm với tất cả vật cản động
        for obstacle_id, obstacle in self.dynamic_obstacles.items():
            will_collide, collision_info = self.predict_collision(
                robot_pos, robot_direction, robot_speed, obstacle_id)

            if will_collide:
                intersection_point, t_closest = collision_info

                if t_closest < min_time_to_collision:
                    min_time_to_collision = t_closest
                    collision_point = intersection_point
                    colliding_obstacle = obstacle_id

        if collision_point is not None:
            # Tính toán thời gian cần chờ và vị trí dừng
            distance_to_stop = min_time_to_collision * robot_speed * 0.7  # Dừng trước điểm va chạm

            # Đơn vị hóa vector hướng
            dir_norm = math.sqrt(robot_direction[0] ** 2 + robot_direction[1] ** 2)
            if dir_norm < 1e-6:
                return False, None

            norm_dir = (robot_direction[0] / dir_norm, robot_direction[1] / dir_norm)

            stop_position = (
                int(robot_pos[0] + distance_to_stop * norm_dir[0]),
                int(robot_pos[1] + distance_to_stop * norm_dir[1])
            )

            # Thời gian chờ: thời gian để vật cản đi qua + một khoảng an toàn
            obstacle_speed = math.sqrt(
                self.dynamic_obstacles[colliding_obstacle]['velocity'][0] ** 2 +
                self.dynamic_obstacles[colliding_obstacle]['velocity'][1] ** 2
            )

            # Nếu vật cản đứng yên hoặc di chuyển quá chậm
            if obstacle_speed < 0.1:
                wait_time = 0  # Không chờ, tìm đường khác
                return False, None

            # Thêm logic động để tính wait time
            base_wait = min_time_to_collision * 0.8  # 80% của thời gian collision
            safety_buffer = 0.3  # Buffer an toàn

            # Điều chỉnh theo size của vật cản
            obstacle_size = self.dynamic_obstacles[colliding_obstacle].get('size', 1.0)
            size_factor = 1.0 + (obstacle_size - 1.0) * 0.3  # Bigger obstacles need more wait time

            wait_time = max(0.3, min(2.0, (base_wait + safety_buffer) * size_factor))

            return True, (stop_position, wait_time)

        return False, None