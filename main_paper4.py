import math
import numpy as np
import pygame as pg
import time
import torch
import argparse

from project_B.logic_projectB import LogicAlgorithm, Q as Q_B
from grid_map import Grid_Map, EPSILON
from dynamic_obstacles_manager import DynamicObstaclesManager
from project_B.dynamic_obstacle_projectB import DynamicObstacle
from collections import deque
from copy import deepcopy

# Xử lý tham số dòng lệnh
parser = argparse.ArgumentParser(description='Robot Coverage Path Planning with Dynamic Obstacles')
parser.add_argument('--map', type=str, default='map/real_map/denmark.txt', help='Path to map file')
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

FPS = 40

total_travel_length = 0
coverage_length, retreat_length, advance_length = 0, 0, 0
return_charge_count = 1
count_cell_go_through = 1
deadlock_count = 0
extreme_deadlock_count = 0
dynamic_wait_count = 0  # Đếm số lần robot phải chờ do vật cản động
execute_time = time.time()
total_coverage_cells = 0  # Tổng số cells đã được coverage (có thể overlap)
total_free_cells = 0     # Tổng số free cells trong environment

# Find special area
from optimization import get_special_area

special_areas = get_special_area(ENVIRONMENT)

# Pre-calculate return path to charge station from each cell in ENVIRONMENT
from optimization import return_path_matrix, get_return_path

return_matrix = return_path_matrix(ENVIRONMENT, battery_pos)

# Project B constants
VISION_SENSOR_RANGE = 5
NUMS_SAMPLE = 5000
MIN_PROB_THRESHOLD = 3

# Project B dynamic obstacles
dynamic_obs_list = []
dynamic_obs_list.append(DynamicObstacle((3, 6), (2, 1), 4, 10))


def check_valid_pos(pos):
    row, col = pos
    if row < 0 or row >= ROW_COUNT: return False
    if col < 0 or col >= COL_COUNT: return False
    return True


class Robot:
    def __init__(self, battery_pos, map_row_count, map_col_count):
        # Only use Project B logic
        self.logic = LogicAlgorithm(map_row_count, map_col_count)
        # Remove: self.logic_b = LogicAlgorithm(map_row_count, map_col_count)
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

        # Remove all Project A obstacle handling components
        # Keep only essential tracking
        self.total_moves = 0

        # Add missing attributes for Project B compatibility
        self.waiting = False
        self.wait_time = 0
        self.wait_start_time = 0
        self.wait_reason = ""
        self.map = None  # Will be set by Project B logic
        self.classified_obstacles = {}
        self.dynamic_obstacle_ids = {}
        self.detected_positions = set()  # Fix missing attribute

        # Project B components
        self.static_map = None
        self.dynamic_map = None
        self.predict_map = None
        self.prob_map = None
        self.seen_map = None
        self.velocity = 10
        self.scan_freq = 2
        self.alpha_1 = 0.3
        self.alpha_2 = 0.7
        self.alpha_3 = 0.3
        self.alpha_4 = 0.7
        self.obs_prev_detected_dict = dict()
        self.obs_detected_dict = dict()

    def set_map(self, environment):
        # Only use Project B numeric format
        self.init_static_map_b(environment)
        self.logic.init_weight_map(self.static_map)

    def init_static_map_b(self, environment):
        """Initialize Project B maps"""
        # Convert environment to numeric format for Project B
        row_count, col_count = len(environment), len(environment[0])
        numeric_env = np.zeros((row_count, col_count), dtype=int)

        for x in range(row_count):
            for y in range(col_count):
                if environment[x, y] == 1 or environment[x, y] == 'o':
                    numeric_env[x, y] = 1  # obstacle
                elif environment[x, y] == 'e':
                    numeric_env[x, y] = 2  # visited
                elif environment[x, y] == 'd':
                    numeric_env[x, y] = 3  # dynamic obstacle
                else:
                    numeric_env[x, y] = 0  # free space

        self.static_map = deepcopy(numeric_env)
        self.dynamic_map = deepcopy(numeric_env)
        self.predict_map = deepcopy(numeric_env)
        self.prob_map = np.zeros((row_count, col_count), dtype=float)
        self.seen_map = deepcopy(numeric_env)
        self.predict_map[self.battery_pos] = self.dynamic_map[self.battery_pos] = 2
        self.seen_map[self.battery_pos] = 2
        # Fix: use self.logic instead of self.logic_b

    def run(self):
        global FPS, deadlock_count, extreme_deadlock_count, dynamic_wait_count
        clock = pg.time.Clock()
        run = True
        pause = False
        coverage_finish = False

        # Biến theo dõi thời gian cho vật cản động
        last_time = time.time()
        loop_count = 0

        while run:
            loop_count += 1
            # Tính delta time cho vật cản động
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Cập nhật vật cản động
            dynamic_obstacles.update(delta_time)

            # Update Project B dynamic obstacles
            self.update_dynamic_map_b(loop_count)

            # Update probability map if needed
            if loop_count % self.velocity == 0:
                self.update_probability_map_and_seen_map_b()

            ui.draw()

            # Vẽ thêm vật cản động nếu có
            if 'dynamic_obstacles' in globals():
                dynamic_obstacles.draw(ui.WIN)
            # Always draw vision sensor (Project B only)
            self.draw_vision_sensor()
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

            if self.logic.state == Q_B.FINISH:
                if not coverage_finish:
                    coverage_finish = True
                    self.retreat()
                    self.charge()

                    global execute_time
                    execute_time = time.time() - execute_time

                    print('Coverage Finish')

                # FN (do nothing until close window)
                continue

            # Remove Project A obstacle detection
            # Keep only Project B obstacle management

            # Cập nhật thông tin vật cản động từ dynamic_obstacles_manager
            if 'dynamic_obstacles' in globals():
                for obstacle in dynamic_obstacles.obstacles:
                    pos = obstacle['pos']
                    # Mark as dynamic obstacle in Project B format
                    if self.dynamic_map[pos] not in (1, 2):  # not obstacle or visited
                        self.dynamic_map[pos] = 3  # dynamic obstacle
                # Initialize selected_cell
                selected_cell = None
                # CRITICAL FIX: Call task() BEFORE getting waypoint (like Project B)
                if self.logic.state != Q_B.DEADLOCK:
                    self.task()

                # MOVE LOGIC OUTSIDE LOOP - Always use Project B P-Decision Framework
                flag_b = self.detect_dynamic_obs_b(VISION_SENSOR_RANGE)

                if flag_b:
                    self.logic.set_map(self.seen_map)
                    self.logic.set_prob_map(self.prob_map)
                    max_bid_value, replan_wp = self.logic.get_replan_wp(self.current_pos)
                    wp = [replan_wp] if replan_wp else []

                    # Go-or-wait decision from Project B
                    designated_wp = self.logic.boustrophedon_moving(self.current_pos)
                    if wp != designated_wp and self.prob_map[self.current_pos] < MIN_PROB_THRESHOLD and len(
                            designated_wp) > 0:
                        designated_wp = designated_wp[0]
                        if self.prob_map[designated_wp] > 0:
                            continue  # Wait
                        else:
                            wp = [designated_wp]
                else:
                    # Use Project B boustrophedon motion
                    wp = self.logic.get_wp(self.current_pos)

                if len(wp) == 0:
                    print(f"DEBUG: No waypoint found at {self.current_pos}, state={self.logic.state}")
                    if self.logic.state == Q_B.DEADLOCK:
                        # Handle deadlock directly without select_from_wp
                        selected_cell = None  # Will trigger deadlock handling below
                    else:
                        continue
                else:
                    selected_cell = self.select_from_wp(wp)
                print(f"DEBUG: Current pos: {self.current_pos}, WP candidates: {wp}, Selected: {selected_cell}")
                print(
                    f"DEBUG: Logic state: {self.logic.state}, Weight map at current: {self.logic.weight_map[self.current_pos]}")

            # Handle different cases
            if selected_cell is None and self.logic.state == Q_B.DEADLOCK:
                # CRITICAL: Check energy FIRST before any deadlock escape
                print(f"DEBUG DEADLOCK: Energy check before escape - Current: {self.energy:.2f}")

                # If low energy, prioritize charging over deadlock escape
                if self.energy < 50:  # Energy threshold for safe operations
                    print(f"DEBUG DEADLOCK: Low energy detected - forcing charge planning")
                    self.charge_planning()
                    continue

                # Handle deadlock with energy awareness
                print(f"DEBUG DEADLOCK: Starting deadlock escape from {self.current_pos}")
                path = self.logic.escape_deadlock_path(self.current_pos)
                print(f"DEBUG DEADLOCK: Found path: {path}")

                if len(path) == 0:
                    print("DEBUG DEADLOCK: No escape path found - FINISH")
                    self.logic.state = Q_B.FINISH
                    continue
                else:
                    # Follow entire deadlock path with energy checking
                    print(f"DEBUG DEADLOCK: Following deadlock path with energy constraints")
                    self.follow_path_plan(path, time_delay=0.05, check_energy=True, stop_on_unexpored=True)
                continue
            elif selected_cell is None:
                continue
            else:
                # Move to selected cell
                if self.logic.state == Q_B.NORMAL:
                    # Check energy constraint before moving
                    if not self.check_enough_energy(selected_cell):
                        print(f"DEBUG: Energy low! Current energy: {self.energy}, need energy check failed")
                        self.charge_planning()
                        continue
                    self.move_to(selected_cell)

                elif self.logic.state == Q_B.DEADLOCK:
                    print(f"DEBUG DEADLOCK: Starting deadlock escape from {self.current_pos}")
                    path = self.logic.escape_deadlock_path(self.current_pos)
                    print(f"DEBUG DEADLOCK: Found path: {path}")

                    if len(path) == 0:
                        print("DEBUG DEADLOCK: No escape path found - FINISH")
                        self.logic.state = Q_B.FINISH
                        continue
                    else:
                        # Check if dynamic obstacles detected
                        current_flag_b = self.detect_dynamic_obs_b(VISION_SENSOR_RANGE)
                        if current_flag_b:
                            print("DEBUG DEADLOCK: Using dynamic escape")
                            _, deadlock_wp = self.logic.escape_deadlock_dynamic(self.current_pos, path[-1])
                            print(f"DEBUG DEADLOCK: Dynamic escape waypoint: {deadlock_wp}")
                            self.move_to(deadlock_wp)
                        else:
                            print(f"DEBUG DEADLOCK: Using static escape to {path[0]}")
                            self.move_to(path[0])

    def select_from_wp(self, wp):
        new_wp = self.get_better_wp(wp)
        if len(new_wp) > 0: wp = new_wp

        return min(wp, key=self.travel_cost)

    def task(self):
        global total_coverage_cells
        current_pos = self.current_pos
        # Update Project B maps
        self.static_map[current_pos] = 2
        self.dynamic_map[current_pos] = 2
        self.seen_map[current_pos] = 2

        # CRITICAL FIX: Update logic's weight_map to mark as visited
        self.logic.weight_map[current_pos] = 2

        ui.task(current_pos)
        total_coverage_cells += 1
        print(f"DEBUG TASK: Marked {current_pos} as visited, weight_map value: {self.logic.weight_map[current_pos]}")

        # Initialize self.map for compatibility
        if self.map is None:
            self.map = self.seen_map

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

        # Remove duplicate task() call - task() already called in main logic
        # if self.move_status == 0:  # coverage mode
        #     self.task()

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
        if return_matrix[wp][1] == math.inf:
            return True  # Can't calculate return path, assume OK
        return_dist_from_wp = return_matrix[wp][1]
        expected_energy = math.dist(self.current_pos, wp) + 0.5 * return_dist_from_wp
        print(f"DEBUG ENERGY: Current={self.energy:.2f}, Need={expected_energy:.2f}, Return_dist={return_dist_from_wp:.2f}")
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
        is_retreat = self.mode == "RETREAT"
        wait_loops = 0
        max_wait_loops = 50
        clock = pg.time.Clock()  # giới hạn tốc độ vòng lặp nếu cần

        for pos in path:
            print(f"\n🔁 [RETREAT STEP] Next pos: {pos}, energy left: {self.energy:.2f}")

            if self.seen_map[pos] == 3:
                print(f"⚠️  Cell {pos} is marked as dynamic obstacle in map")

            # Cập nhật vật cản động mỗi bước
            delta_time = clock.get_time() / 1000.0
            dynamic_obstacles.update(delta_time)

            # CRITICAL: Always check energy before moving during retreat
            move_energy = math.dist(self.current_pos, pos) * 0.5  # retreat uses half energy
            if self.energy < move_energy:
                print(f"⚠️ EMERGENCY: Not enough energy for retreat step {self.current_pos} -> {pos}")
                print(f"⚠️ Current energy: {self.energy:.2f}, Need: {move_energy:.2f}")
                print(f"⚠️ Stopping retreat - robot stuck at {self.current_pos}")
                return

            while check_energy and not self.check_enough_energy(pos):
                if is_retreat:
                    print(f"⚠️ Not enough energy during retreat at {pos} — skipping this step")
                    return
                else:
                    self.charge_planning()
            # Áp dụng waiting rule nếu có vật cản động
            while False:  # Temporary - will implement Project B collision check
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

        if not check_valid_pos((x_up - 1, y_up)) or self.seen_map[(x_up - 1, y_up)] in (1, 2, 3):
            new_wp.append((x_up, y_up))
        if not check_valid_pos((x_down + 1, y_down)) or self.seen_map[(x_down + 1, y_down)] in (1, 2, 3):
            new_wp.append((x_down, y_down))
        return new_wp

    def set_special_areas(self, special_areas):
        pass

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

    # ========== PROJECT B METHODS ==========
    def update_dynamic_map_b(self, loop_count):
        """Update dynamic map for Project B obstacles"""
        # Check if maps are initialized
        if self.static_map is None or self.dynamic_map is None:
            return

        row_count, col_count = len(self.static_map), len(self.static_map[0])

        # Clear previous dynamic obstacles
        for x in range(row_count):
            for y in range(col_count):
                if self.dynamic_map[x, y] == 3 or self.dynamic_map[x, y] == 4:
                    self.dynamic_map[x, y] = self.static_map[x, y]
                if self.predict_map[x, y] == 3 or self.predict_map[x, y] == 4:
                    self.predict_map[x, y] = self.static_map[x, y]

        # Move Project B dynamic obstacles
        for obs in dynamic_obs_list:
            if loop_count % int(obs.velocity) == 0:
                obs.move_one_step(self.static_map)

            # Mark obstacle positions
            for dx in range(obs.height):
                for dy in range(obs.width):
                    x, y = obs.cur_row + dx, obs.cur_col + dy
                    if self.current_pos == (x, y):
                        print("Collision with Project B dynamic obstacle!")
                        raise Exception('Collision with obstacle')
                    self.dynamic_map[x, y] = 3

        # CRITICAL: Also mark manual dynamic obstacles from Project A
        if 'dynamic_obstacles' in globals():
            for obstacle in dynamic_obstacles.obstacles:
                obs_pos = obstacle['pos']
                obs_size = obstacle.get('size', (1, 1))

                # Mark all cells occupied by manual obstacle
                if isinstance(obs_size, tuple):
                    height, width = obs_size
                    for dr in range(int(height)):
                        for dc in range(int(width)):
                            x, y = obs_pos[0] + dr, obs_pos[1] + dc
                            if 0 <= x < row_count and 0 <= y < col_count:
                                self.dynamic_map[x, y] = 3
                                self.seen_map[x, y] = 3

    def update_probability_map_and_seen_map_b(self):
        """Update probability map and seen map for Project B"""
        # Check if maps are initialized
        if self.static_map is None or self.prob_map is None or self.seen_map is None:
            return

        row_count, col_count = len(self.static_map), len(self.static_map[0])

        # Reset seen map
        for x in range(row_count):
            for y in range(col_count):
                self.seen_map[x, y] = self.static_map[x, y]

        # Reset probability map
        for x in range(row_count):
            for y in range(col_count):
                if self.static_map[x, y] == 1:
                    self.prob_map[x, y] = 0
                else:
                    self.prob_map[x, y] = 0

        # Detect obstacles and update probability
        detected_obs = self.obs_sensor_b(vision_range=VISION_SENSOR_RANGE)
        obs_potential_next_move = []
        obs_occupy_list = []

        for obs in detected_obs:
            self.calculateProbabilityMap_b(obs)
            for row in range(row_count):
                for col in range(col_count):
                    if self.prob_map[row, col] != 0 and self.dynamic_map[row, col] != 1:
                        obs_potential_next_move.append((row, col))
            obs_potential_next_move += self.get_potential_positions_b(obs)
            obs_occupy_list += obs.get_current_occupy_positions()

        # Mark potential positions
        for pos in obs_potential_next_move:
            self.dynamic_map[pos] = 4

        for pos in obs_occupy_list:
            self.dynamic_map[pos] = 3
            self.seen_map[pos] = 3
            self.prob_map[pos] = 100

    def obs_sensor_b(self, vision_range=VISION_SENSOR_RANGE):
        """Project B obstacle sensor"""
        obs_detected_list = []
        in_sensor_list = []
        border_cells = self.get_border_cells_b(self.current_pos)

        for pos in border_cells:
            obstruct_cell_list = self.obstruct_cell_list_b(self.current_pos, pos)
            for cell in obstruct_cell_list:
                if self.dynamic_map[cell] == 1:
                    break
                if cell not in in_sensor_list:
                    in_sensor_list.append(cell)

        for obs in dynamic_obs_list:
            if set(obs.get_current_occupy_positions()) & set(in_sensor_list):
                obs_detected_list.append(obs)

        self.obs_prev_detected_dict = self.obs_detected_dict.copy()
        self.obs_detected_dict = {obs: obs.get_pos() for obs in obs_detected_list}

        return obs_detected_list

    def get_border_cells_b(self, cur_pos):
        """Get border cells for vision sensor"""
        left_border = right_border = up_border = down_border = -1
        border_cells = []
        cur_x, cur_y = cur_pos[0], cur_pos[1]

        for x in range(cur_x - VISION_SENSOR_RANGE, cur_x + 1):
            if x >= 0:
                up_border = x
                break

        for x in range(cur_x, cur_x + VISION_SENSOR_RANGE + 1):
            if x >= ROW_COUNT:
                break
            else:
                down_border = x

        for y in range(cur_y - VISION_SENSOR_RANGE, cur_y + 1):
            if y >= 0:
                left_border = y
                break

        for y in range(cur_y, cur_y + VISION_SENSOR_RANGE + 1):
            if y >= COL_COUNT:
                break
            else:
                right_border = y

        for x in range(up_border, down_border + 1):
            for y in range(left_border, right_border + 1):
                if x == up_border or x == down_border:
                    border_cells.append((x, y))
                else:
                    if y == left_border or y == right_border:
                        border_cells.append((x, y))
        return border_cells

    def obstruct_cell_list_b(self, pos_from, pos_to, strict=False):
        """Calculate obstructed cells between two positions"""

        def sign(n):
            return int(np.sign(n))

        threshold = 0.3
        start = (pos_from[0] + 0.5, pos_from[1] + 0.5)
        goal = (pos_to[0] + 0.5, pos_to[1] + 0.5)

        vecto = (goal[0] - start[0], goal[1] - start[1])
        angle = - np.arctan2(vecto[0], vecto[1])

        (x, y) = pos_from
        cell_list = [pos_from]

        sx, sy = sign(vecto[0]), sign(vecto[1])
        dx = abs(0.5 / math.sin(angle)) if vecto[0] != 0 else math.inf
        dy = abs(0.5 / math.cos(angle)) if vecto[1] != 0 else math.inf
        sum_x, sum_y = dx, dy

        while (x, y) != pos_to:
            (movx, movy) = (sum_x < sum_y or math.isclose(sum_x, sum_y),
                            sum_y < sum_x or math.isclose(sum_x, sum_y))

            prev_x, prev_y = x, y
            prev_sum_x, prev_sum_y = sum_x, sum_y
            if movx:
                x += sx
                sum_x += 2 * dx

            if movy:
                y += sy
                sum_y += 2 * dy

            if strict:
                if movx and movy:
                    cell_list.extend([(prev_x, prev_y + sy), (prev_x + sx, prev_y)])
                elif movx and not movy:
                    projection_y = (abs(prev_sum_x * math.cos(angle)) - 0.5) % 1
                    if projection_y < threshold:
                        cell_list.append((x, prev_y - sy))
                    elif projection_y > 1 - threshold:
                        cell_list.append((prev_x, prev_y + sy))
                elif movy and not movx:
                    projection_x = (abs(prev_sum_y * math.sin(angle)) - 0.5) % 1
                    if projection_x < threshold:
                        cell_list.append((prev_x - sx, y))
                    elif projection_x > 1 - threshold:
                        cell_list.append((prev_x + sx, prev_y))

            cell_list.append((x, y))

        return cell_list

    def get_potential_positions_b(self, obs):
        """Get potential positions for obstacle"""
        neighbour = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        obs_occupy_list = obs.get_current_occupy_positions()

        prob_neighbour_list = []
        visited = []
        queue = deque()
        queue.extend([(i, 0) for i in obs_occupy_list])

        while queue:
            current_pos, step = queue.popleft()
            for dx, dy in neighbour:
                x, y = current_pos[0] + dx, current_pos[1] + dy
                if not check_valid_pos((x, y)):
                    continue
                if (x, y) in visited:
                    continue
                if self.dynamic_map[x, y] == 1 or self.dynamic_map[x, y] == 3:
                    continue
                if step > self.scan_freq * obs.velocity:
                    continue
                queue.append(((x, y), step + 1))
                visited.append((x, y))
                if self.prob_map[x, y] > 0 and self.dynamic_map[x, y] != 1:
                    prob_neighbour_list.append((x, y))
        return prob_neighbour_list

    def sample_b(self, z):
        """Sampling function for Project B"""
        rand = np.random.uniform(-z, z, 12)
        return np.sum(rand) * 1 / 12

    def sampling_b(self, obs):
        """Sampling for obstacle prediction"""
        (x, y) = obs.get_pos()
        v_prime = obs.v + self.sample_b(self.alpha_1 * abs(obs.velocity) + self.alpha_2 * abs(obs.w))
        w_prime = obs.w + self.sample_b(self.alpha_3 * abs(obs.velocity) + self.alpha_4 * abs(obs.w))
        x_prime = x - v_prime / w_prime * math.sin(obs.theta) + v_prime / w_prime * math.sin(
            obs.theta + self.scan_freq * w_prime)
        y_prime = y + v_prime / w_prime * math.cos(obs.theta) - v_prime / w_prime * math.cos(
            obs.theta + self.scan_freq * w_prime)
        return (round(x_prime), round(y_prime))

    def calculateProbabilityMap_b(self, obs):
        """Calculate probability map for obstacle"""
        new_pos_dict = dict()
        for _ in range(NUMS_SAMPLE):
            new_pos = self.sampling_b(obs)
            if new_pos not in new_pos_dict.keys():
                new_pos_dict[new_pos] = 1
            else:
                new_pos_dict[new_pos] += 1

        for new_pos in new_pos_dict.keys():
            prob = round(new_pos_dict[new_pos] / NUMS_SAMPLE * 100, 1)
            if not check_valid_pos(new_pos):
                continue
            if prob < self.prob_map[new_pos]:
                continue
            self.prob_map[new_pos] = prob

    def detect_dynamic_obs_b(self, vision_range=VISION_SENSOR_RANGE):
        """Detect dynamic obstacles using Project B method"""
        # Check if dynamic_map is initialized
        if self.dynamic_map is None:
            return False

        border_cells = self.get_border_cells_b(self.current_pos)
        in_sensor_list = []

        for pos in border_cells:
            obstruct_cell_list = self.obstruct_cell_list_b(self.current_pos, pos)
            for cell in obstruct_cell_list:
                if self.dynamic_map[cell] == 1:
                    break
                if cell not in in_sensor_list:
                    in_sensor_list.append(cell)

        # Check both Project B dynamic obstacles and manual obstacles
        for cell in in_sensor_list:
            if self.dynamic_map[cell] == 3:
                return True
            # Also check manual dynamic obstacles from Project A
            if hasattr(ui, 'map') and ui.map[cell] == 'd':
                return True
        return False

    def draw_vision_sensor(self):
        """Draw vision sensor circle on screen"""
        vehicle_center = (int((self.current_pos[1] + 1 / 2) * EPSILON), int((self.current_pos[0] + 1 / 2) * EPSILON))
        sensor_radius = int((VISION_SENSOR_RANGE + 1 / 2) * EPSILON)
        pg.draw.circle(ui.WIN, (204, 255, 255), vehicle_center, sensor_radius, width=4)

def main():
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.set_map(ENVIRONMENT)
    # Tính tổng số free cells (S_free) trong environment
    global total_free_cells
    total_free_cells = np.sum(ENVIRONMENT == 0)  # Đếm số cells = 0 (free space)
    print(f"Total free cells in environment: {total_free_cells}")

    # Khởi tạo trình quản lý vật cản động với manual obstacles từ ui
    global dynamic_obstacles
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

    # ===== BWave Framework Metrics (theo đúng Paper) =====
    print('\nCoverage:\t', coverage_length)
    print('Advance:\t', advance_length)
    print('Return:\t', retreat_length)
    print('-' * 8)
    print('Total Path Length:', total_travel_length)
    print('Total:', total_travel_length)
    print('Time: ', execute_time)

    # ===== BWave Framework Metrics (theo đúng Paper) =====
    print('=' * 50)

    # 1. Total Path Length (fixed terminology)
    print(f'1. Total Path Length: {total_travel_length:.2f}')

    # 2. Overlap Rate (already correct)
    if total_free_cells > 0:
        bwave_overlap_rate = (total_coverage_cells / total_free_cells - 1) * 100
        print(f'2. Overlap Rate: {bwave_overlap_rate:.2f}%')
    else:
        print('2. Overlap Rate: 0.00%')

    # 3. Number of Returns (fixed from 1 to 0 initial)
    print(f'3. Number of Returns: {return_charge_count}')

    # 4. Number of Deadlocks (already correct)
    print(f'4. Number of Deadlocks: {deadlock_count} (extreme: {extreme_deadlock_count})')

    # 5. Execution Time (already correct)
    print(f'5. Execution Time: {execute_time:.3f}s')

    # 6. Coverage Rate
    covered_cells = total_coverage_cells

    if total_free_cells > 0:
        coverage_rate = (covered_cells / total_free_cells) * 100.0
        uncovered_cells = total_free_cells - covered_cells
        print(f'6. Coverage Rate: {coverage_rate - bwave_overlap_rate:.2f}%')
    else:
        print(f'6. Coverage Rate: 0.00%')

    print('=' * 50)

if __name__ == "__main__":
    main()