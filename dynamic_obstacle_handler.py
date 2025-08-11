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
                if hasattr(self, 'map') and hasattr(self.map, 'clear_dynamic'):
                    self.map.clear_dynamic(data['position'])  # clear cell trên bản đồ nếu có
                obstacles_to_remove.append(obstacle_id)
        for obstacle_id in obstacles_to_remove:
            del self.dynamic_obstacles[obstacle_id]

    def predict_collision(self, robot_pos, robot_direction, robot_speed, obstacle_id):
        """Dự đoán va chạm với improved velocity model"""
        if obstacle_id not in self.dynamic_obstacles:
            return False, None

        obstacle = self.dynamic_obstacles[obstacle_id]
        obstacle_pos = obstacle['position']
        obstacle_vel = obstacle['velocity']

        # ✅ IMPROVED: Tính velocity uncertainty cho human movement
        vel_uncertainty = 0.3  # Human movement uncertainty factor

        # ✅ IMPROVED: Non-linear velocity model
        # Humans tend to slow down near obstacles
        distance_to_robot = math.sqrt((obstacle_pos[0] - robot_pos[0]) ** 2 +
                                      (obstacle_pos[1] - robot_pos[1]) ** 2)

        if distance_to_robot < 3.0:  # Close to robot
            # Humans slow down when near robots
            deceleration_factor = max(0.3, distance_to_robot / 3.0)
            adjusted_obstacle_vel = (obstacle_vel[0] * deceleration_factor,
                                     obstacle_vel[1] * deceleration_factor)
        else:
            adjusted_obstacle_vel = obstacle_vel

        # ✅ IMPROVED: Add velocity uncertainty
        vel_magnitude = math.sqrt(adjusted_obstacle_vel[0] ** 2 + adjusted_obstacle_vel[1] ** 2)
        if vel_magnitude > 0:
            uncertainty_x = vel_uncertainty * vel_magnitude * (np.random.random() - 0.5)
            uncertainty_y = vel_uncertainty * vel_magnitude * (np.random.random() - 0.5)
            final_obstacle_vel = (adjusted_obstacle_vel[0] + uncertainty_x,
                                  adjusted_obstacle_vel[1] + uncertainty_y)
        else:
            final_obstacle_vel = adjusted_obstacle_vel

        # Continue with existing collision calculation using final_obstacle_vel
        dir_norm = math.sqrt(robot_direction[0] ** 2 + robot_direction[1] ** 2)
        if dir_norm < 1e-6:
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
        raw_size = obstacle.get('size', 1.0)
        if isinstance(raw_size, tuple):
            obstacle_size = max(raw_size)
        else:
            obstacle_size = raw_size
        robot_size = 1.0  # Assume robot cũng có size
        safety_distance = (obstacle_size + robot_size) / 2 + 0.3  # Buffer thêm 0.3

        if closest_distance < safety_distance:
            # Sẽ có va chạm
            return True, (closest_pos, t_closest)

        return False, None

    def apply_waiting_rule(self, robot_pos, robot_direction, robot_speed):
        """Áp dụng waiting rule """
        min_collision_time = float('inf')
        collision_info = None

        for obstacle_id, obstacle in self.dynamic_obstacles.items():
            obstacle_pos = obstacle['position']
            obstacle_vel = obstacle['velocity']

            # Tính vector từ robot đến obstacle
            rel_pos = (obstacle_pos[0] - robot_pos[0], obstacle_pos[1] - robot_pos[1])
            d = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)

            if d < 0.1:  # Too close
                continue

            # Normalize robot direction
            dir_norm = math.sqrt(robot_direction[0]**2 + robot_direction[1]**2)
            if dir_norm < 1e-6:
                continue
            robot_dir_normalized = (robot_direction[0]/dir_norm, robot_direction[1]/dir_norm)

            # Robot velocity vector
            vr = (robot_dir_normalized[0] * robot_speed, robot_dir_normalized[1] * robot_speed)
            vo = obstacle_vel

            # Paper formula (24): α = θr - θo
            # Calculate angle between robot and obstacle directions
            robot_angle = math.atan2(robot_dir_normalized[1], robot_dir_normalized[0])
            obstacle_angle = math.atan2(vo[1], vo[0]) if (vo[0]**2 + vo[1]**2) > 1e-6 else 0
            alpha = robot_angle - obstacle_angle

            # Paper formula (25): Cosine theorem for collision time
            # (vr*t)² + (vo*t)² - d² = 2*vr*vo*t*cos(α)
            vr_mag = math.sqrt(vr[0]**2 + vr[1]**2)
            vo_mag = math.sqrt(vo[0]**2 + vo[1]**2)

            if vr_mag < 1e-6 or vo_mag < 1e-6:
                continue

            # Quadratic equation: at² + bt + c = 0
            # where (vr² + vo² - 2*vr*vo*cos(α))t² - d² = 0
            a = vr_mag**2 + vo_mag**2 - 2*vr_mag*vo_mag*math.cos(alpha)
            c = -d**2

            if abs(a) < 1e-6:  # Linear case
                continue

            discriminant = -4*a*c
            if discriminant < 0:  # No collision
                continue

            t0 = math.sqrt(discriminant) / (2*abs(a))  # Collision time

            if t0 < min_collision_time and t0 > 0:
                min_collision_time = t0

                # Paper formula (26): Sr = vr*t0 - Lr (robot stopping distance)
                Lr = 0.5  # Robot length
                Sr = vr_mag * t0 - Lr

                if Sr > 0:
                    stop_pos = (
                        int(robot_pos[0] + Sr * robot_dir_normalized[0]),
                        int(robot_pos[1] + Sr * robot_dir_normalized[1])
                    )

                    # Paper formula (27): Wait time calculation
                    safety_buffer = 1.0
                    wait_time = t0 + safety_buffer

                    collision_info = (stop_pos, wait_time)

        if collision_info:
            return True, collision_info
        return False, None