import math
import numpy as np
import pygame as pg
import time
import csv
import torch
import threading
import contextlib
import argparse

from a_star import GridMapGraph, a_star_search
from logic import Logic, Q
from grid_map import Grid_Map, EPSILON
from obstacle_classifier import ObstacleClassifier
from dynamic_obstacle_handler import DynamicObstacleHandler
from virtual_camera import VirtualCamera
from dynamic_obstacles_manager import DynamicObstaclesManager

# Xử lý tham số dòng lệnh
parser = argparse.ArgumentParser(description='Robot Coverage Path Planning with Dynamic Obstacles')
parser.add_argument('--map', type=str, default='map/real_map/denmark.txt', help='Path to map file')
#parser.add_argument('--dynamic', type=int, default=3, help='Number of dynamic obstacles')
parser.add_argument('--speed', type=float, default=0.1, help='Speed of dynamic obstacles')
parser.add_argument('--energy', type=float, default=1000, help='Energy capacity')
args = parser.parse_args()

# coverage:             1 unit of energy / cell width
# advance & retreat:    0.5 unit of energy / cell width
ENERGY_CAPACITY = args.energy

ui = Grid_Map()
ui.read_map(args.map)
ENVIRONMENT, battery_pos = ui.edit_map()
# ui.save_map('map/empty_map.txt')

ROW_COUNT = len(ENVIRONMENT)
COL_COUNT = len(ENVIRONMENT[0])

FPS = 40

total_travel_length = 0
coverage_length, retreat_length, advance_length = 0, 0, 0
return_charge_count = 1
count_cell_go_through = 1
deadlock_count = 0
extreme_deadlock_count = 0
dynamic_wait_count = 0  # Đếm số lần robot phải chờ do vật cản động
dynamic_obstacles = None  # Global reference cho dynamic obstacles manager
execute_time = time.time()
total_coverage_cells = 0  # Tổng số cells đã được coverage (có thể overlap)
total_free_cells = 0     # Tổng số free cells trong environment
# Find special area
from optimization import get_special_area

special_areas = get_special_area(ENVIRONMENT)

# Pre-calculate return path to charge station from each cell in ENVIRONMENT
from optimization import return_path_matrix, get_return_path

return_matrix = return_path_matrix(ENVIRONMENT, battery_pos)


def check_valid_pos(pos):
    row, col = pos
    if row < 0 or row >= ROW_COUNT: return False
    if col < 0 or col >= COL_COUNT: return False
    return True


class Robot:
    def __init__(self, battery_pos, map_row_count, map_col_count):
        self.logic = Logic(map_row_count, map_col_count, grid_map=ui)
        '''
        map:
            'u': unvisited
            'e': explored
            'o': obstacle (static)
            'd': dynamic obstacle (new)
        '''
        self.mode = "NORMAL"  # chế độ mặc định ban đầu
        self.map = None
        self.current_pos = battery_pos

        # The angle between the robot direction and left to right axis in rad [0, 2pi)
        # (up direction at the start)
        self.angle = math.pi / 2

        self.battery_pos = battery_pos
        self.energy = ENERGY_CAPACITY
        self.estimated_return_energy = 0

        self.move_status = 0  # 0: normal coverage, 1: retreat, 2: charge, 3: advance
        self.cache_path = []  # store temporary path (e.g.: retreat, advance)

        # New components for dynamic obstacle handling
        self.use_gpu = torch.cuda.is_available()
        self.obstacle_classifier = ObstacleClassifier(use_gpu=self.use_gpu)
        self.dynamic_obstacle_handler = DynamicObstacleHandler()
        self.virtual_camera = VirtualCamera(ui, EPSILON)

        # Obstacle tracking state
        self.classified_obstacles = {}  # {pos: ('static'/'dynamic', confidence)}
        self.dynamic_obstacle_ids = {}  # {pos: id}
        self.next_obstacle_id = 1

        # Waiting state for dynamic obstacles
        self.waiting = False
        self.wait_time = 0
        self.wait_start_time = 0
        self.wait_reason = ""  # Lý do chờ đợi để hiển thị

        # Previous camera image for motion detection
        self.previous_camera_image = None

        # Essential tracking only
        self.total_moves = 0
        self.detected_positions = set()  # Cần thiết cho detect_and_classify_obstacles

    def set_map(self, environment):
        row_count, col_count = len(environment), len(environment[0])
        self.map = np.full((row_count, col_count), 'u')

        for x in range(len(environment)):
            for y in range(len(environment[0])):
                if environment[x, y] == 1:
                    self.map[x, y] = 'o'

        self.logic.set_weight_map(environment)

    def run(self):
        global FPS, deadlock_count, extreme_deadlock_count, dynamic_wait_count
        clock = pg.time.Clock()
        run = True
        pause = False
        coverage_finish = False

        # Biến theo dõi thời gian cho vật cản động
        last_time = time.time()

        while run:
            # ✅ LOCK mechanism cho thread safety
            with threading.Lock() if hasattr(self, '_update_lock') else contextlib.nullcontext():
                # Tính delta time cho vật cản động
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time

                # ✅ ATOMIC UPDATE: Snapshot obstacle state
                obstacle_snapshot = self._get_obstacle_snapshot()

                # Cập nhật vật cản động với snapshot
                dynamic_obstacles.update(delta_time)

            ui.draw()

            # Vẽ thêm vật cản động nếu có
            if 'dynamic_obstacles' in globals():
                dynamic_obstacles.draw(ui.WIN)
            pg.display.flip()

            # Show thêm thông so cho doi
            if self.waiting:
                waiting_text = f"Waiting: {self.wait_reason} ({round(self.wait_time - (current_time - self.wait_start_time), 1)}s)"
                waiting_img = pg.font.SysFont(None, 24).render(waiting_text, True, (255, 0, 0))
                ui.WIN.blit(waiting_img, (10, 10))

            clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:  # pause
                        pause = not pause
                        pg.image.save(ui.WIN, 'tmp/screenshot.png')
                    elif event.key == pg.K_LEFT:  # slow down
                        FPS /= 2
                    elif event.key == pg.K_RIGHT:  # speed up
                        FPS *= 2
                if event.type == pg.QUIT:
                    run = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_UP:
                        for obs in dynamic_obstacles.obstacles:
                            vx, vy = obs['velocity']
                            obs['velocity'] = (vx * 2, vy * 2)
                        print("↑ Tăng vận tốc vật cản động ×2")

                    elif event.key == pg.K_DOWN:
                        for obs in dynamic_obstacles.obstacles:
                            vx, vy = obs['velocity']
                            obs['velocity'] = (vx / 2, vy / 2)
                        print("↓ Giảm vận tốc vật cản động ÷2")

            if pause:
                continue

            if self.waiting:
                current_time = time.time()
                if current_time - self.wait_start_time >= self.wait_time:
                    self.waiting = False
                    print("Waiting complete, continuing movement")
                else:
                    # Still waiting
                    continue

            if self.logic.state == Q.FINISH:
                if not coverage_finish:
                    coverage_finish = True
                    self.retreat()
                    self.charge()

                    global execute_time
                    execute_time = time.time() - execute_time

                    print('Coverage Finish')

                # FN (do nothing until close window)
                continue

            # Detect and classify obstacles
            self.detect_and_classify_obstacles()

            # Cập nhật thông tin vật cản động từ dynamic_obstacles_manager
            if 'dynamic_obstacles' in globals():
                for obstacle in dynamic_obstacles.obstacles:
                    pos = obstacle['pos']
                    obstacle_id = obstacle['id']

                    # Update với size information
                    if obstacle_id not in self.dynamic_obstacle_handler.dynamic_obstacles:
                        self.dynamic_obstacle_handler.register_obstacle(obstacle_id, pos,
                                                                        obstacle.get('velocity', (0, 0)))
                        self.dynamic_obstacle_handler.dynamic_obstacles[obstacle_id]['size'] = obstacle.get('size', 1.0)
                    else:
                        self.dynamic_obstacle_handler.update_obstacle(obstacle_id, pos)
                        self.dynamic_obstacle_handler.dynamic_obstacles[obstacle_id]['size'] = obstacle.get('size', 1.0)

                    # Đánh dấu vị trí là vật cản động trong bản đồ
                    if self.map[pos] not in ('o', 'e'):  # Không ghi đè lên vật cản tĩnh hoặc ô đã thăm
                        self.map[pos] = 'd'
                    # Lưu thông tin phân loại
                    self.classified_obstacles[pos] = ('dynamic', 0.95)
                    self.dynamic_obstacle_ids[pos] = obstacle_id

            # Remove old dynamic obstacles
            self.dynamic_obstacle_handler.remove_old_obstacles()

            wp = self.logic.get_wp(self.current_pos)
            if len(wp) == 0:
                continue
            selected_cell = self.select_from_wp(wp)

            if selected_cell == self.current_pos:
                self.task()
            else:
                # CP 0
                if self.logic.state == Q.NORMAL:
                    # ✅ ENERGY CHECK TRƯỚC - Logic đúng
                    if self.check_enough_energy(selected_cell) == False:
                        self.charge_planning()
                        continue
                    # ✅ SAU ĐÓ MỚI CHECK COLLISION
                    if self.check_dynamic_collision(selected_cell):
                        dynamic_wait_count += 1
                        continue
                    self.move_to(selected_cell)

                # CP l (l > 0)
                elif self.logic.state == Q.DEADLOCK:
                    path, dist = self.logic.cache_path, self.logic.cache_dist
                    print(f"Deadlock ({round(dist, 2)})")

                    deadlock_count += 1
                    if dist > math.sqrt(ROW_COUNT ** 2 + COL_COUNT ** 2) / 4:
                        extreme_deadlock_count += 1

                    self.follow_path_plan(path, time_delay=0.05, check_energy=True, stop_on_unexpored=True)

    def _get_obstacle_snapshot(self):
        """Get atomic snapshot of current obstacle state"""
        snapshot = {
            'positions': {},
            'velocities': {},
            'sizes': {}
        }

        if hasattr(self, 'dynamic_obstacles') and dynamic_obstacles.obstacles:
            for obs in dynamic_obstacles.obstacles:
                obs_id = obs['id']
                snapshot['positions'][obs_id] = obs['pos']
                snapshot['velocities'][obs_id] = obs.get('velocity', (0, 0))
                snapshot['sizes'][obs_id] = obs.get('size', 1.0)

        return snapshot

    def select_from_wp(self, wp):
        new_wp = self.get_better_wp(wp)
        if len(new_wp) > 0: wp = new_wp

        return min(wp, key=self.travel_cost)

    def task(self):
        global total_coverage_cells
        current_pos = self.current_pos
        self.map[current_pos] = 'e'
        self.logic.update_explored(current_pos)
        ui.task(current_pos)
        total_coverage_cells += 1  # Đếm mỗi lần task (coverage)

    def move_to(self, pos):
        global total_travel_length, coverage_length, retreat_length, advance_length
        dist = energy = math.dist(self.current_pos, pos)

        if self.move_status in (1, 3):  # retreat or advance cost half energy as coverage
            energy = 0.5 * energy

        if self.energy < energy:
            raise Exception('Robot run out of battery')
        self.energy -= energy

        self.rotate_to(pos)
        self.current_pos = pos

        # Increment move counter an toàn
        if hasattr(self, 'total_moves'):
            self.total_moves += 1

        if self.move_status == 0:
            ui.move_to(pos)
            coverage_length += dist
        elif self.move_status == 1:
            ui.move_retreat(pos)
            retreat_length += dist
        elif self.move_status == 3:
            ui.move_advance(pos)
            advance_length += dist

        total_travel_length += dist
        ui.set_energy_display(self.energy)

    def travel_cost(self, pos_to):
        pos_from = self.current_pos
        turn_angle = abs(self.angle - self.get_angle(pos_to))
        if turn_angle > math.pi:  # always take the smaller angle to turn
            turn_angle = 2 * math.pi - turn_angle
        travel_dist = math.dist(pos_from, pos_to)

        # cost of travel distance, turning rad
        cost = 2 * travel_dist + 1 * turn_angle
        return cost

    def get_angle(self, pos_to):
        pos_from = self.current_pos
        vecto = (pos_to[0] - pos_from[0], pos_to[1] - pos_from[1])
        angle = - np.arctan2(vecto[0], vecto[1])
        return angle % (2 * math.pi)

    def rotate_to(self, pos_to):
        self.angle = self.get_angle(pos_to)


    def check_enough_energy(self, wp):
        return_dist_from_wp = return_matrix[wp][1]
        basic_energy = math.dist(self.current_pos, wp) + 0.5 * return_dist_from_wp

        # ✅ THÊM: Estimate waiting energy cost for dynamic obstacles
        waiting_energy_buffer = 0

        # Check potential dynamic obstacles on path to wp
        path_cells = self._get_path_cells(self.current_pos, wp)
        for cell in path_cells:
            if hasattr(self, 'map') and self.map[cell] == 'd':
                waiting_energy_buffer += 2.0  # Energy cost for potential waiting

        # ✅ THÊM: Buffer cho potential detours
        dynamic_detour_buffer = basic_energy * 0.2  # 20% buffer for detours

        total_expected_energy = basic_energy + waiting_energy_buffer + dynamic_detour_buffer
        return self.energy >= total_expected_energy


    def _get_path_cells(self, start, end):
        """Estimate cells on direct path"""
        cells = []
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steps = max(dx, dy)

        if steps == 0:
            return [start]

        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            cells.append((x, y))

        return cells

    def charge_planning(self):
        global return_charge_count
        return_charge_count += 1
        # retreat
        self.retreat()

        # charge
        self.charge()
        time.sleep(0.1)

        # advance
        self.advance()

        # coverage
        self.move_status = 0

    def retreat(self):
        return_path = get_return_path(return_matrix, self.current_pos)
        self.cache_path = return_path  # save for reuse in advance path

        self.move_status = 1
        ui.set_charge_path(return_path)
        self.follow_path_plan(return_path, time_delay=0.05)

    def charge(self):
        self.move_status = 2
        self.energy = ENERGY_CAPACITY

    def advance(self):
        self.move_status = 3
        advance_path = list(reversed(self.cache_path))
        ui.set_charge_path(advance_path)
        self.follow_path_plan(advance_path, time_delay=0.05)

    def follow_path_plan(self, path, time_delay=0, check_energy=False, stop_on_unexpored=False):
        is_retreat = self.mode == "RETREAT"
        wait_loops = 0
        max_wait_loops = 50
        clock = pg.time.Clock()  # giới hạn tốc độ vòng lặp nếu cần

        for pos in path:
            print(f"\n🔁 [RETREAT STEP] Next pos: {pos}, energy left: {self.energy:.2f}")

            if self.map[pos] == 'd':
                print(f"⚠️  Cell {pos} is marked as dynamic obstacle in map")

            # Cập nhật vật cản động mỗi bước
            delta_time = clock.get_time() / 1000.0
            dynamic_obstacles.update(delta_time)

            while check_energy and not self.check_enough_energy(pos):
                if is_retreat:
                    print(f"⚠️ Not enough energy during retreat at {pos} — skipping this step")
                    return
                else:
                    self.charge_planning()
            # Áp dụng waiting rule nếu có vật cản động
            while self.check_dynamic_collision(pos):
                delta_time = clock.tick(FPS) / 1000.0  # cập nhật đúng mỗi frame
                dynamic_obstacles.update(delta_time)
                ui.draw()
                dynamic_obstacles.draw(ui.WIN)
                if self.waiting:
                    wait_remain = round(self.wait_time - (time.time() - self.wait_start_time), 1)
                    wait_text = f"Waiting: {self.wait_reason} ({wait_remain}s)"
                    wait_img = pg.font.SysFont(None, 24).render(wait_text, True, (255, 0, 0))
                    ui.WIN.blit(wait_img, (10, 10))

                pg.display.flip()
                pg.time.delay(100)

                if self.waiting and time.time() - self.wait_start_time >= self.wait_time:
                    self.waiting = False
                    print("✅ Done waiting — will try to move again")

                if is_retreat:
                    wait_loops += 1
                    if wait_loops > max_wait_loops:
                        print("⛔ Retreat waiting timeout — skipping this cell")
                        break

            # Di chuyển bình thường
            self.move_to(pos)
            wait_loops = 0  # Reset sau mỗi bước để tránh tích luỹ sai
            ui.draw()
            if 'dynamic_obstacles' in globals():
                dynamic_obstacles.draw(ui.WIN)

            if stop_on_unexpored and self.logic.weight_map[pos] > 0:
                return

    def get_better_wp(self, wp):
        if len(wp) == 1: return wp

        new_wp = []
        x_up, y_up = min(wp, key=lambda x: x[0])
        x_down, y_down = max(wp, key=lambda x: x[0])

        if not check_valid_pos((x_up - 1, y_up)) or self.map[(x_up - 1, y_up)] in ('o', 'e', 'd'):
            new_wp.append((x_up, y_up))
        if not check_valid_pos((x_down + 1, y_down)) or self.map[(x_down + 1, y_down)] in ('o', 'e', 'd'):
            new_wp.append((x_down, y_down))
        return new_wp

    def set_special_areas(self, special_areas):
        self.logic.set_special_areas(special_areas)
        self.set_inner_special_areas(special_areas)

    def set_inner_special_areas(self, special_areas):
        candidate_areas = get_special_area(ENVIRONMENT, reverse_dir=True)
        for parent_region in special_areas:
            for child_region in candidate_areas:
                if not (set(child_region.cell_list) <= set(parent_region.cell_list)): continue

                floor_weight = -1
                for pos in parent_region.cell_list:
                    if pos[1] == child_region.max_y:
                        floor_weight = self.logic.weight_map[pos] + 2
                        break

                for x, y in child_region.cell_list:
                    self.logic.weight_map[x, y] = floor_weight + (child_region.max_y - y)

    def detect_and_classify_obstacles(self):
        """Detect and classify obstacles using GoogLeNet + virtual camera"""
        # Skip detection một số frames để giảm false positive
        if not hasattr(self, '_detection_skip_counter'):
            self._detection_skip_counter = 0
        self._detection_skip_counter += 1
        if self._detection_skip_counter % 3 != 0:
            return

        # Get direction from robot angle
        direction = (math.cos(self.angle), math.sin(self.angle))

        # Capture high-res image from virtual camera
        current_image = self.virtual_camera.capture_image(self.current_pos, direction)

        # Detect moving obstacles using frame differencing
        if self.previous_camera_image is not None:
            dynamic_obstacles_detected = self.virtual_camera.detect_dynamic_obstacles(
                current_image, self.previous_camera_image
            )

            # NEW: Classify each detected obstacle using GoogLeNet
            for (rel_row, rel_col), (width, height) in dynamic_obstacles_detected:
                # Convert relative position to absolute
                abs_row = self.current_pos[0] + rel_row
                abs_col = self.current_pos[1] + rel_col

                if not check_valid_pos((abs_row, abs_col)):
                    continue

                # Capture ROI of the specific obstacle
                obstacle_roi = self.virtual_camera.capture_obstacle_roi((abs_row, abs_col), (height, width))

                # Classify using GoogLeNet
                class_name, confidence = self.obstacle_classifier.classify(obstacle_roi)

                print(f"🔍 GoogLeNet Classification: {class_name} (confidence: {confidence:.3f})")

                # Only accept high-confidence predictions
                if confidence > 0.75:  # Threshold cho accuracy
                    pos_key = (abs_row, abs_col)

                    if class_name == 'dynamic' and pos_key not in self.detected_positions:
                        self.detected_positions.add(pos_key)
                        self.map[pos_key] = 'd'

                        # Register with dynamic obstacle handler
                        obstacle_id = f"googlet_{self.next_obstacle_id}"
                        self.next_obstacle_id += 1
                        self.dynamic_obstacle_ids[pos_key] = obstacle_id
                        self.dynamic_obstacle_handler.register_obstacle(obstacle_id, pos_key)

                        # Save classification result
                        self.classified_obstacles[pos_key] = (class_name, confidence)

                    elif class_name == 'static':
                        self.map[pos_key] = 'o'  # Mark as static obstacle
                        self.classified_obstacles[pos_key] = (class_name, confidence)
                else:
                    print(f"⚠️ Low confidence detection ignored: {confidence:.3f}")
        self.previous_camera_image = current_image


    def check_dynamic_collision(self, target_pos):
        """Unified collision check với consistent state management"""

        # ✅ STEP 1: GoogLeNet classification (ground truth)
        obstacle_roi = self.virtual_camera.capture_obstacle_roi(target_pos, (2, 2))
        class_name, confidence = self.obstacle_classifier.classify(obstacle_roi)

        print(f"🔍 Real-time GoogLeNet: {target_pos} -> {class_name} ({confidence:.3f})")

        # ✅ STEP 2: Update map state based on AI classification
        if confidence > 0.75:  # High confidence threshold
            if class_name == 'dynamic':
                self.map[target_pos] = 'd'  # Update map to dynamic
                self.classified_obstacles[target_pos] = ('dynamic', confidence)
            elif class_name == 'static':
                self.map[target_pos] = 'o'  # Update map to static
                self.classified_obstacles[target_pos] = ('static', confidence)
                self.logic.weight_map[target_pos] = -1  # ← THÊM dòng này
                print(f"🚫 GoogLeNet detected STATIC obstacle - blocking movement")
                return False  # Static obstacle - block movement, no waiting
        else:
            print(f"⚠️ Low confidence GoogLeNet result - using fallback logic")
            # Continue to fallback logic below

        # ✅ STEP 3: Collision logic dựa trên updated map state
        if self.map[target_pos] == 'd':  # Now based on AI classification
            # Find corresponding dynamic obstacle
            is_real_dynamic = False
            obstacle = None

            for obs in dynamic_obstacles.obstacles:
                obstacle_center = obs['pos']
                distance = math.sqrt((target_pos[0] - obstacle_center[0]) ** 2 +
                                     (target_pos[1] - obstacle_center[1]) ** 2)

                if distance <= 1.5:  # Within obstacle range
                    is_real_dynamic = True
                    obstacle = obs
                    break

            if is_real_dynamic:
                obstacle_size = obstacle.get('size', 1.0)
                if isinstance(obstacle_size, tuple):
                    obstacle_size = max(obstacle_size)

                wait_time = 0.5 + (obstacle_size - 1.0) * 0.5

                self.waiting = True
                self.wait_time = wait_time
                self.wait_start_time = time.time()
                self.wait_reason = f"AI-classified DYNAMIC (conf={confidence:.2f}, size={obstacle_size:.1f})"

                print(f"🤖 AI-based waiting: {wait_time:.1f}s for dynamic obstacle")
                return True
            else:
                # Clean up stale dynamic marking
                if check_valid_pos(target_pos) and self.map[target_pos] == 'd':
                    self.map[target_pos] = 0
                    print(f"🧹 Cleaned up stale dynamic marking at {target_pos}")
                return False

        # ✅ STEP 4: Fallback - Original logic for manual obstacles
        if self.map[target_pos] in (1, 'o'):
            return False  # Static obstacle - không chờ

        # ✅ STEP 5: Calculate movement direction for waiting rule
        direction = (target_pos[0] - self.current_pos[0], target_pos[1] - self.current_pos[1])
        distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)

        if distance < 1e-6:  # If distance is almost zero
            return False

        # Robot speed (in cells/second)
        robot_speed = 1.0

        # ✅ STEP 6: Check and apply waiting rule if needed
        need_wait, wait_info = self.dynamic_obstacle_handler.apply_waiting_rule(
            self.current_pos, direction, robot_speed
        )

        if need_wait:
            stop_position, wait_time = wait_info
            self.wait_reason = "Collision predicted by velocity model"
            print(f"🔄 Velocity-based waiting: {wait_time:.2f} seconds")

            # Only move to stop position if different from current position
            if stop_position != self.current_pos:
                # Tactical movement to stop position
                dist = math.dist(self.current_pos, stop_position)
                self.energy -= 0.5 * dist  # Half energy for tactical movement
                self.rotate_to(stop_position)
                self.current_pos = stop_position
                ui.update_vehicle_pos(stop_position)
                ui.set_energy_display(self.energy)

            # Start waiting
            self.waiting = True
            self.wait_time = wait_time
            self.wait_start_time = time.time()
            return True

        return False
def main():
    global dynamic_obstacles  # Khai báo global
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.set_map(ENVIRONMENT)
    robot.set_special_areas(special_areas)
    # Tính tổng số free cells (S_free) trong environment
    global total_free_cells
    total_free_cells = np.sum(ENVIRONMENT == 0)  # Đếm số cells = 0 (free space)
    print(f"Total free cells in environment: {total_free_cells}")

    # Khởi tạo trình quản lý vật cản động với manual obstacles từ ui
    dynamic_obstacles = DynamicObstaclesManager(ui, num_obstacles=0, speed_factor=args.speed)

    # Khởi tạo các vật cản manual từ grid_map only if
    if hasattr(ui, 'dynamic_obstacles') and ui.dynamic_obstacles:
        dynamic_obstacles.initialize_obstacles()
        print(f"Initialized {len(ui.dynamic_obstacles)} manual dynamic obstacles")
    print("Using BWave Framework with Dynamic Obstacles")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    print(f"Created {len(dynamic_obstacles.obstacles)} manual dynamic obstacles")

    global execute_time
    execute_time = time.time()
    robot.run()

    print('\nCoverage:\t', coverage_length)
    print('Retreat:\t', retreat_length)
    print('Advance:\t', advance_length)
    print('-' * 8)
    print('Total:', total_travel_length)
    print('Time: ', execute_time)

    # ===== BWave Framework Metrics (theo đúng Paper) =====
    print('\n' + '=' * 50)
    print('BWAVE FRAMEWORK METRICS')
    print('=' * 50)

    # 1. Total Path Length (đã có)
    print(f'1. Total Path Length: {total_travel_length:.2f}')

    # 2. Overlap Rate (theo công thức BWave paper)
    if total_free_cells > 0:
        bwave_overlap_rate = (total_coverage_cells / total_free_cells - 1) * 100
        print(f'2. Overlap Rate: {bwave_overlap_rate:.2f}%')
    else:
        print('2. Overlap Rate: 0.00%')

    # 3. Number of Returns
    print(f'3. Number of Returns: {return_charge_count}')

    # 4. Number of Deadlocks (total và extreme)
    print(f'4. Number of Deadlocks: {deadlock_count} (extreme: {extreme_deadlock_count})')

    # 5. Execution Time
    print(f'5. Execution Time: {execute_time:.3f}s')

    print('=' * 50)

if __name__ == "__main__":
    main()