import math
import numpy as np
import pygame as pg
import time
import csv
import torch
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
parser.add_argument('--map', type=str, default='map/real_map/Cantwell.txt', help='Path to map file')
parser.add_argument('--dynamic', type=int, default=3, help='Number of dynamic obstacles')
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

FPS = 80

total_travel_length = 0
coverage_length, retreat_length, advance_length = 0, 0, 0
return_charge_count = 1
count_cell_go_through = 1
deadlock_count = 0
extreme_deadlock_count = 0
dynamic_wait_count = 0  # Đếm số lần robot phải chờ do vật cản động
execute_time = time.time()

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
        self.logic = Logic(map_row_count, map_col_count)
        '''
        map:
            'u': unvisited
            'e': explored
            'o': obstacle (static)
            'd': dynamic obstacle (new)
        '''
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

        # Debug tracking
        self.detected_positions = set()  # Track đã phát hiện
        self.debug_dynamic_count = 0  # Debug counter
        self.cleaned_dynamic_cells = 0
        self.false_positive_count = 0
        self.start_time = time.time()
        self.total_moves = 0

    def set_map(self, environment):
        row_count, col_count = len(environment), len(environment[0])
        self.map = np.full((row_count, col_count), 'u')

        for x in range(len(environment)):
            for y in range(len(environment[0])):
                if environment[x, y] == 1:
                    self.map[x, y] = 'o'

        self.logic.set_weight_map(environment)

    def run(self):
        # Ensure all attributes exist (backward compatibility)
        if not hasattr(self, 'cleaned_dynamic_cells'):
            self.cleaned_dynamic_cells = 0
        if not hasattr(self, 'false_positive_count'):
            self.false_positive_count = 0
        if not hasattr(self, 'total_moves'):
            self.total_moves = 0

        global FPS, deadlock_count, extreme_deadlock_count, dynamic_wait_count
        clock = pg.time.Clock()
        run = True
        pause = False
        coverage_finish = False

        # Biến theo dõi thời gian cho vật cản động
        last_time = time.time()

        while run:
            # Tính delta time cho vật cản động
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Cập nhật vật cản động
            dynamic_obstacles.update(delta_time)

            ui.draw()

            # Vẽ thêm vật cản động
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
            for obstacle in dynamic_obstacles.obstacles:
                pos = obstacle['pos']
                obstacle_id = obstacle['id']

                # Update với size information
                if obstacle_id not in self.dynamic_obstacle_handler.dynamic_obstacles:
                    self.dynamic_obstacle_handler.register_obstacle(obstacle_id, pos, obstacle.get('velocity', (0, 0)))
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
            if len(wp) == 0: continue
            selected_cell = self.select_from_wp(wp)

            if selected_cell == self.current_pos:
                self.task()
            else:
                # CP 0
                if self.logic.state == Q.NORMAL:
                    # Check for potential collision with dynamic obstacles
                    if self.check_dynamic_collision(selected_cell):
                        # Collision detected, waiting implemented
                        dynamic_wait_count += 1
                        continue

                    if self.check_enough_energy(selected_cell) == False:
                        self.charge_planning()
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

    def select_from_wp(self, wp):
        new_wp = self.get_better_wp(wp)
        if len(new_wp) > 0: wp = new_wp

        return min(wp, key=self.travel_cost)

    def task(self):
        current_pos = self.current_pos
        self.map[current_pos] = 'e'
        self.logic.update_explored(current_pos)
        ui.task(current_pos)

    def move_to(self, pos):
        global total_travel_length, coverage_length, retreat_length, advance_length, count_cell_go_through
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
        if self.move_status == 0:  # coverage
            count_cell_go_through += 1

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
        expected_energy = math.dist(self.current_pos, wp) + 0.5 * return_dist_from_wp
        if self.energy < expected_energy:
            return False
        else:
            return True

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
        clock = pg.time.Clock()
        for pos in path:
            clock.tick(FPS / 4)

            while check_energy == True and self.check_enough_energy(pos) == False:
                self.charge_planning()

            # Kiểm tra vật cản động trước khi di chuyển
            while self.check_dynamic_collision(pos):
                ui.draw()
                dynamic_obstacles.draw(ui.WIN)
                # Hiển thị thông tin chờ đợi
                if self.waiting:
                    waiting_text = f"Waiting: {self.wait_reason} ({round(self.wait_time - (time.time() - self.wait_start_time), 1)}s)"
                    waiting_img = pg.font.SysFont(None, 24).render(waiting_text, True, (255, 0, 0))
                    ui.WIN.blit(waiting_img, (10, 10))
                pg.display.flip()
                time.sleep(0.1)

                # Cập nhật vật cản động
                current_time = time.time()
                if self.waiting and current_time - self.wait_start_time >= self.wait_time:
                    self.waiting = False

            self.move_to(pos)
            ui.draw()
            dynamic_obstacles.draw(ui.WIN)

            # Check for dynamic obstacles along the path
            if self.check_dynamic_collision(pos):
                # Wait for obstacle to pass
                while self.waiting:
                    ui.draw()
                    dynamic_obstacles.draw(ui.WIN)
                    current_time = time.time()
                    if current_time - self.wait_start_time >= self.wait_time:
                        self.waiting = False
                    pg.time.delay(100)

            if stop_on_unexpored:
                if self.logic.weight_map[pos] > 0: return

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
        """Detect and classify obstacles using the virtual camera"""
        # Skip detection một số frames để giảm false positive
        if not hasattr(self, '_detection_skip_counter'):
            self._detection_skip_counter = 0
        self._detection_skip_counter += 1
        if self._detection_skip_counter % 3 != 0:
            return

        # Get direction from robot angle
        direction = (math.cos(self.angle), math.sin(self.angle))

        # Capture image from virtual camera
        current_image = self.virtual_camera.capture_image(self.current_pos, direction)

        # Detect moving obstacles
        if self.previous_camera_image is not None:
            dynamic_obstacles_detected = self.virtual_camera.detect_dynamic_obstacles(
                current_image, self.previous_camera_image
            )

            # Update dynamic obstacle information
            for (rel_row, rel_col), (width, height) in dynamic_obstacles_detected:
                # Bỏ qua detections quá gần robot (có thể là noise)
                distance_from_robot = math.sqrt(rel_row ** 2 + rel_col ** 2)
                if distance_from_robot < 2:  # Ignore detections trong bán kính 2 cells
                    continue

                # Convert relative position to absolute
                abs_row = self.current_pos[0] + rel_row
                abs_col = self.current_pos[1] + rel_col

                # Check if position is valid
                if not check_valid_pos((abs_row, abs_col)):
                    continue

                # Tránh duplicate counting
                pos_key = (abs_row, abs_col)
                if pos_key not in self.detected_positions and self.map[pos_key] != 'o':
                    self.detected_positions.add(pos_key)
                    self.map[pos_key] = 'd'
                    self.debug_dynamic_count += 1

                    # Get obstacle ID or create new
                    if pos_key in self.dynamic_obstacle_ids:
                        obstacle_id = self.dynamic_obstacle_ids[pos_key]
                        # Update obstacle position
                        self.dynamic_obstacle_handler.update_obstacle(obstacle_id, pos_key)
                    else:
                        # Register new obstacle
                        obstacle_id = f"obs_{self.next_obstacle_id}"
                        self.next_obstacle_id += 1
                        self.dynamic_obstacle_ids[pos_key] = obstacle_id
                        self.dynamic_obstacle_handler.register_obstacle(obstacle_id, pos_key)

                    # Save obstacle type
                    self.classified_obstacles[pos_key] = ('dynamic', 0.9)
        self.previous_camera_image = current_image

    def check_dynamic_collision(self, target_pos):
        """Check for collision with dynamic obstacles when moving to target_pos"""
        # Kiểm tra nếu vị trí đích là vật cản tĩnh thì KHÔNG áp dụng waiting rule
        if self.map[target_pos] in (1, 'o'):
            return False  # Vật cản tĩnh - không chờ

        # Nếu vị trí đích là vật cản động được tạo thủ công, thì áp dụng waiting rule
        if self.map[target_pos] == 'd':
            # Kiểm tra age của detection để tránh chờ obstacles cũ
            current_time = time.time()
            is_real_dynamic = False
            obstacle = None

            # Check trong vùng của vật cản based on size
            for obs in dynamic_obstacles.obstacles:
                obstacle_size = obs.get('size', 1.0)
                radius = int(obstacle_size / 2)
                obstacle_center = obs['pos']

                # Check if target_pos is within obstacle area
                distance = math.sqrt((target_pos[0] - obstacle_center[0]) ** 2 +
                                     (target_pos[1] - obstacle_center[1]) ** 2)
                if distance <= radius + 0.5:  # Include safety margin
                    is_real_dynamic = True
                    obstacle = obs
                    break

            # Chỉ wait nếu là real dynamic obstacle gần đây
            if is_real_dynamic:
                # Thời gian chờ tỉ lệ với size của vật cản
                obstacle_size = obstacle.get('size', 1.0) if obstacle else 1.0
                wait_time = 1.5 + (obstacle_size - 1.0) * 0.5  # Longer wait for bigger obstacles

                self.waiting = True
                self.wait_time = wait_time
                self.wait_start_time = time.time()
                self.wait_reason = f"Large obstacle (size={obstacle_size:.1f}) at target ({target_pos})"
                return True
            else:
                # Clean up stale dynamic marking với bounds checking
                if check_valid_pos(target_pos) and self.map[target_pos] == 'd':
                    self.cleaned_dynamic_cells += 1
                    self.false_positive_count += 1
                    self.map[target_pos] = 0
                return False

        # Calculate movement direction
        direction = (target_pos[0] - self.current_pos[0], target_pos[1] - self.current_pos[1])
        distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)

        if distance < 1e-6:  # If distance is almost zero
            return False

        # Robot speed (in cells/second)
        robot_speed = 100.0

        # Check if target position is already occupied by a dynamic obstacle
        if self.map[target_pos] == 'd':
            self.waiting = True
            self.wait_time = 2.0  # Chờ 2 giây nếu vị trí đích đã bị chiếm
            self.wait_start_time = time.time()
            self.wait_reason = f"Obstacle at target ({target_pos})"
            print(f"Target position {target_pos} occupied by dynamic obstacle. Waiting...")
            return True

        # Check and apply waiting rule if needed
        need_wait, wait_info = self.dynamic_obstacle_handler.apply_waiting_rule(
            self.current_pos, direction, robot_speed
        )

        if need_wait:
            stop_position, wait_time = wait_info
            self.wait_reason = "Collision predicted"
            print(f"Dynamic obstacle detected! Waiting for {wait_time:.2f} seconds")

            # Only move to stop position if different from current position
            if stop_position != self.current_pos:
                # Use move_to to go to the stop position
                # We don't use the original move_to to avoid triggering coverage
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
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.set_map(ENVIRONMENT)
    robot.set_special_areas(special_areas)

    # Khởi tạo trình quản lý vật cản động
    global dynamic_obstacles
    dynamic_obstacles = DynamicObstaclesManager(ui, num_obstacles=5, speed_factor=2.0)
    print("Map layout:")
    for row in range(len(ENVIRONMENT)):
        for col in range(len(ENVIRONMENT[0])):
            if ENVIRONMENT[row, col] == 1:
                print(f"Static obstacle at ({row}, {col})")

    print("\nDynamic obstacles:")
    for obs in dynamic_obstacles.obstacles:
        print(f"Dynamic obstacle at {obs['pos']} with velocity {obs['velocity']}")
    # Show information about obstacle classification
    print("Using obstacle classification with GoogLeNet")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    print(f"Dynamic obstacles: {args.dynamic}")

    global execute_time
    execute_time = time.time()
    robot.run()

    print('\nCoverage:\t', coverage_length)
    print('Retreat:\t', retreat_length)
    print('Advance:\t', advance_length)
    print('-' * 8)
    print('Total:', total_travel_length)

    overlap_rate = (count_cell_go_through / np.sum(robot.map == 'e') - 1) * 100
    print('\nOverlap rate: ', overlap_rate)
    print('Number Of Return: ', return_charge_count)
    print('Number of extreme deadlock:', extreme_deadlock_count, '/', deadlock_count)
    print('Number of dynamic obstacle waits:', dynamic_wait_count)

    # Add statistics about dynamic obstacles
    dynamic_count = sum(1 for val in robot.classified_obstacles.values() if val[0] == 'dynamic')
    print(f'Dynamic obstacles detected: {dynamic_count}')
    print(
        f'Waiting events: {sum(1 for t in robot.dynamic_obstacle_handler.dynamic_obstacles.values() if len(t["history"]) > 2)}')

    print('Time: ', execute_time)

    # Debug information
    print(f'Debug: Total dynamic detections: {robot.debug_dynamic_count}')
    print(f'Debug: Unique positions detected: {len(robot.detected_positions)}')
    print(
        f'Debug: Detection ratio: {robot.debug_dynamic_count / len(robot.detected_positions) if robot.detected_positions else 0:.2f}')
    print(f'Debug: Detection per frame: {robot.debug_dynamic_count / count_cell_go_through:.2f}')
    print(
        f'Debug: Avg detections per position: {robot.debug_dynamic_count / len(robot.detected_positions) if robot.detected_positions else 0:.2f}')

    # Debug để hiểu discrepancy
    total_classified_dynamic = sum(1 for v in robot.classified_obstacles.values() if v[0] == 'dynamic')
    print(f'Debug: Classified as dynamic: {total_classified_dynamic}')
    print(
        f'Debug: Dynamic vs classified ratio: {robot.debug_dynamic_count / total_classified_dynamic if total_classified_dynamic > 0 else 0:.2f}')

    # Clean up statistics
    actual_map_dynamic = np.sum(robot.map == 'd')
    print(f'Debug: Final dynamic cells in map: {actual_map_dynamic}')

    # Summary metrics
    print("\n=== SUMMARY ===")
    print(f"Efficiency Score: {((1 - overlap_rate / 100) * 100):.1f}% (lower overlap = better)")
    print(f"Speed Improvement: {execute_time:.1f}s total")
    print(f"Dynamic Handling: {dynamic_wait_count} waits, {robot.debug_dynamic_count} unique detections")
    print(
        f"Energy Efficiency: {return_charge_count} charges, avg distance per charge: {coverage_length / return_charge_count:.1f}")

    # Performance analysis
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    print(
        f"False positive rate: {(robot.false_positive_count / robot.debug_dynamic_count * 100) if robot.debug_dynamic_count > 0 else 0:.1f}%")
    print(f"Cleaned dynamic cells: {robot.cleaned_dynamic_cells}")

    # Safe access với fallback
    total_moves = getattr(robot, 'total_moves', count_cell_go_through) or 1  # Prevent division by zero
    print(f"Average move time: {execute_time / total_moves:.3f}s per move")

    # Safe calculation
    expected_detections = total_moves * 0.07
    if expected_detections > 0:
        print(f"Detection efficiency: {robot.debug_dynamic_count / expected_detections:.2f} (target ~1.0)")
    else:
        print("Detection efficiency: N/A (no moves recorded)")


if __name__ == "__main__":
    main()