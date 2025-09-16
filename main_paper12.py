import math
import numpy as np
import pygame as pg
import time
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
from optimization import get_special_area, return_path_matrix, get_return_path

parser = argparse.ArgumentParser(description='Robot Coverage Path Planning with Dynamic Obstacles')
parser.add_argument('--map', type=str, default='map/real_map/ribera.txt', help='Path to map file')
parser.add_argument('--speed', type=float, default=1, help='Speed of dynamic obstacles')
parser.add_argument('--energy', type=float, default=1000, help='Energy capacity')
args = parser.parse_args()

ENERGY_CAPACITY = args.energy
FPS = 40

ui = Grid_Map()
ui.read_map(args.map)
ENVIRONMENT, battery_pos = ui.edit_map()
ROW_COUNT = len(ENVIRONMENT)
COL_COUNT = len(ENVIRONMENT[0])

total_travel_length = 0
coverage_length, retreat_length, advance_length = 0, 0, 0
return_charge_count = 0
deadlock_count = 0
extreme_deadlock_count = 0
dynamic_wait_count = 0
dynamic_obstacles = None
execute_time = time.time()
total_coverage_cells = 0
covered_positions = set()
blank_cells = 0
total_free_cells = 0
count_cell_go_through = 0

special_areas = get_special_area(ENVIRONMENT)
return_matrix = return_path_matrix(ENVIRONMENT, battery_pos)


def save_map(self, output_file):
    map = self.map
    with open(output_file, "w", encoding="utf-8") as f:
        col_count, row_count = len(map[0]), len(map) if map else (0, 0)
        f.write(str(col_count) + ' ' + str(row_count) + '\n')
        for row in map:
            line = [str(value) for value in row]
            line = " ".join(line)
            f.write(line + '\n')

        # Append dynamic obstacles section
        f.write("DYNAMIC OBSTACLES\n")
        for obs in self.dynamic_obstacles:
            pos = obs['pos']
            id_str = obs['id']
            size_str = obs['size_str']  # e.g., "2x1"
            shape_str = ",".join([f"({dr},{dc})" for dr, dc in obs['shape']])
            f.write(f"DYNAMIC {id_str} {pos[0]} {pos[1]} {size_str} {shape_str}\n")

def check_valid_pos(pos):
    row, col = pos
    if row < 0 or row >= ROW_COUNT: return False
    if col < 0 or col >= COL_COUNT: return False
    return True

class Robot:
    def __init__(self, initial_battery_pos, map_row_count, map_col_count):
        self.logic = Logic(map_row_count, map_col_count, grid_map=ui)
        self.mode = "NORMAL"
        self.map = None
        self.current_pos = battery_pos
        self.angle = math.pi / 2
        self.battery_pos = initial_battery_pos
        self.energy = ENERGY_CAPACITY
        self.move_status = 0
        self.cache_path = []

        self.use_gpu = torch.cuda.is_available()
        self.obstacle_classifier = ObstacleClassifier(use_gpu=self.use_gpu)
        self.dynamic_obstacle_handler = DynamicObstacleHandler()
        self.virtual_camera = VirtualCamera(ui, EPSILON)

        self.classified_obstacles = {}
        self.dynamic_obstacle_ids = {}
        self.next_obstacle_id = 1
        self.waiting = False
        self.wait_time = 0
        self.wait_start_time = 0
        self.wait_reason = ""
        self.previous_camera_image = None
        self.total_moves = 0
        self.detected_positions = set()

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
        last_time = time.time()

        while run:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            if dynamic_obstacles:
                dynamic_obstacles.update(delta_time)

            ui.draw()
            if dynamic_obstacles:
                dynamic_obstacles.draw(ui.WIN)
            pg.display.flip()

            if self.waiting:
                waiting_text = f"Waiting: {self.wait_reason} ({round(self.wait_time - (current_time - self.wait_start_time), 1)}s)"
                waiting_img = pg.font.SysFont(None, 24).render(waiting_text, True, (255, 0, 0))
                ui.WIN.blit(waiting_img, (10, 10))

            clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        pause = not pause
                        pg.image.save(ui.WIN, 'tmp/screenshot.png')
                    elif event.key == pg.K_LEFT:
                        FPS /= 2
                    elif event.key == pg.K_RIGHT:
                        FPS *= 2
                if event.type == pg.QUIT:
                    run = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_UP:
                        if dynamic_obstacles:
                            for obs in dynamic_obstacles.obstacles:
                                vx, vy = obs['velocity']
                                obs['velocity'] = (vx * 2, vy * 2)
                    elif event.key == pg.K_DOWN:
                        if dynamic_obstacles:
                            for obs in dynamic_obstacles.obstacles:
                                vx, vy = obs['velocity']
                                obs['velocity'] = (vx / 2, vy / 2)

            if pause:
                continue

            if self.waiting:
                current_time = time.time()
                if current_time - self.wait_start_time >= self.wait_time:
                    self.waiting = False
                else:
                    continue

            if self.logic.state == Q.FINISH:
                if not coverage_finish:
                    coverage_finish = True
                    self.retreat()
                    self.charge()
                    global execute_time
                    execute_time = time.time() - execute_time
                continue

            self.detect_and_classify_obstacles()

            if dynamic_obstacles:
                for obstacle in dynamic_obstacles.obstacles:
                    pos = obstacle['pos']
                    obstacle_id = obstacle['id']
                    if obstacle_id not in self.dynamic_obstacle_handler.dynamic_obstacles:
                        self.dynamic_obstacle_handler.register_obstacle(obstacle_id, pos,
                                                                        obstacle.get('velocity', (0, 0)))
                        self.dynamic_obstacle_handler.dynamic_obstacles[obstacle_id]['size'] = obstacle.get('size', 1.0)
                    else:
                        self.dynamic_obstacle_handler.update_obstacle(obstacle_id, pos)
                        self.dynamic_obstacle_handler.dynamic_obstacles[obstacle_id]['size'] = obstacle.get('size', 1.0)

            self.dynamic_obstacle_handler.remove_old_obstacles()

            wp = self.logic.get_wp(self.current_pos)
            if len(wp) == 0:
                continue
            selected_cell = self.select_from_wp(wp)

            if selected_cell == self.current_pos:
                self.task()
            else:
                if self.logic.state == Q.NORMAL:
                    if self.check_enough_energy(selected_cell) == False:
                        self.charge_planning()
                        continue

                    collision_result = self.check_dynamic_collision(selected_cell)
                    if collision_result == True:
                        dynamic_wait_count += 1
                        continue
                    elif collision_result == "BLOCKED":
                        continue

                    self.move_to(selected_cell)

                elif self.logic.state == Q.DEADLOCK:
                    path, dist = self.logic.cache_path, self.logic.cache_dist
                    deadlock_count += 1
                    if dist > math.sqrt(ROW_COUNT ** 2 + COL_COUNT ** 2) / 4:
                        extreme_deadlock_count += 1
                    self.follow_path_plan(path, time_delay=0.05, check_energy=True, stop_on_unexpored=True)

    def select_from_wp(self, wp):
        new_wp = self.get_better_wp(wp)
        if len(new_wp) > 0:
            wp = new_wp
        return min(wp, key=self.travel_cost)

    def task(self):
        global total_coverage_cells, covered_positions, count_cell_go_through  # ✅ NEW: Use covered_positions set
        current_pos = self.current_pos
        self.map[current_pos] = 'e'
        self.logic.update_explored(current_pos)
        ui.task(current_pos)
        total_coverage_cells += 1
        covered_positions.add(current_pos)  # ✅ NEW: Add to set (auto handles duplicates)
        count_cell_go_through += 1  # ✅ FIXED: Track coverage moves for correct overlap calculation

    def move_to(self, pos):
        global total_travel_length, coverage_length, retreat_length, advance_length
        dist = energy = math.dist(self.current_pos, pos)

        if self.move_status in (1, 3):
            energy = 0.5 * energy

        if self.energy < energy:
            raise Exception('Robot run out of battery')
        self.energy -= energy

        self.rotate_to(pos)
        self.current_pos = pos

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
        if turn_angle > math.pi:
            turn_angle = 2 * math.pi - turn_angle
        travel_dist = math.dist(pos_from, pos_to)
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

        waiting_energy_buffer = 0
        path_cells = self._get_path_cells(self.current_pos, wp)
        for cell in path_cells:
            if hasattr(self, 'map') and self.map[cell] == 'd':
                waiting_energy_buffer += 2.0

        dynamic_detour_buffer = basic_energy * 0.2
        total_expected_energy = basic_energy + waiting_energy_buffer + dynamic_detour_buffer
        return self.energy >= total_expected_energy

    def _get_path_cells(self, start, end):
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
        self.retreat()
        self.charge()
        time.sleep(0.1)
        self.advance()
        self.move_status = 0

    def retreat(self):
        return_path = get_return_path(return_matrix, self.current_pos)
        self.cache_path = return_path
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
        clock = pg.time.Clock()

        for pos in path:
            delta_time = clock.get_time() / 1000.0
            if dynamic_obstacles:
                dynamic_obstacles.update(delta_time)

            while check_energy and not self.check_enough_energy(pos):
                if is_retreat:
                    return
                else:
                    self.charge_planning()

            while self.check_dynamic_collision(pos):
                delta_time = clock.tick(FPS) / 1000.0
                if dynamic_obstacles:
                    dynamic_obstacles.update(delta_time)
                ui.draw()
                if dynamic_obstacles:
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

                if is_retreat:
                    wait_loops += 1
                    if wait_loops > max_wait_loops:
                        break

            self.move_to(pos)
            wait_loops = 0
            ui.draw()
            if dynamic_obstacles:
                dynamic_obstacles.draw(ui.WIN)

            if stop_on_unexpored and self.logic.weight_map[pos] > 0:
                return

    def get_better_wp(self, wp):
        if len(wp) == 1:
            return wp

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
                if not (set(child_region.cell_list) <= set(parent_region.cell_list)):
                    continue

                floor_weight = -1
                for pos in parent_region.cell_list:
                    if pos[1] == child_region.max_y:
                        floor_weight = self.logic.weight_map[pos] + 2
                        break

                for x, y in child_region.cell_list:
                    self.logic.weight_map[x, y] = floor_weight + (child_region.max_y - y)

    def detect_and_classify_obstacles(self):
        if not hasattr(self, '_detection_skip_counter'):
            self._detection_skip_counter = 0
        self._detection_skip_counter += 1
        if self._detection_skip_counter % 3 != 0:
            return

        direction = (math.cos(self.angle), math.sin(self.angle))
        current_image = self.virtual_camera.capture_image(self.current_pos, direction)

        if self.previous_camera_image is not None:
            dynamic_obstacles_detected = self.virtual_camera.detect_dynamic_obstacles(
                current_image, self.previous_camera_image
            )

            for (rel_row, rel_col), (width, height) in dynamic_obstacles_detected:
                abs_row = self.current_pos[0] + rel_row
                abs_col = self.current_pos[1] + rel_col

                if not check_valid_pos((abs_row, abs_col)):
                    continue

                if self.map[abs_row, abs_col] == 0:
                    continue

                obstacle_roi = self.virtual_camera.capture_obstacle_roi((abs_row, abs_col), (height, width))
                class_name, confidence = self.obstacle_classifier.classify(obstacle_roi)

                if confidence > 0.75:
                    pos_key = (abs_row, abs_col)

                    if class_name == 'dynamic' and pos_key not in self.detected_positions:
                        self.detected_positions.add(pos_key)
                        self.map[pos_key] = 'd'
                        obstacle_id = f"googlet_{self.next_obstacle_id}"
                        self.next_obstacle_id += 1
                        self.dynamic_obstacle_ids[pos_key] = obstacle_id
                        self.dynamic_obstacle_handler.register_obstacle(obstacle_id, pos_key)
                        self.classified_obstacles[pos_key] = (class_name, confidence)

                    elif class_name == 'static':
                        # Static obstacles - Update map
                        self.classified_obstacles[pos_key] = (class_name, confidence)

        self.previous_camera_image = current_image

    def check_dynamic_collision(self, target_pos):
        collision_detected = False
        closest_obstacle = None

        if dynamic_obstacles and dynamic_obstacles.obstacles:
            for obstacle in dynamic_obstacles.obstacles:
                obstacle_center = obstacle['pos']
                obstacle_size = obstacle.get('size', 1.0)

                if isinstance(obstacle_size, tuple):
                    height, width = obstacle_size

                    for dr in range(-height // 2, height // 2 + 1):
                        for dc in range(-width // 2, width // 2 + 1):
                            obstacle_cell = (obstacle_center[0] + dr, obstacle_center[1] + dc)

                            if (target_pos[0] == obstacle_cell[0] and
                                    target_pos[1] == obstacle_cell[1]):
                                collision_detected = True
                                closest_obstacle = obstacle
                                break
                        if collision_detected:
                            break
                else:
                    distance = math.sqrt((target_pos[0] - obstacle_center[0]) ** 2 +
                                         (target_pos[1] - obstacle_center[1]) ** 2)
                    if distance <= 1.0:
                        collision_detected = True
                        closest_obstacle = obstacle
                        break

        if not collision_detected:
            return False

        obstacle_roi = self.virtual_camera.capture_obstacle_roi(target_pos, (2, 2))
        class_name, confidence = self.obstacle_classifier.classify(obstacle_roi)

        # Only process dynamic obstacles since static map is completely known
        if confidence > 0.55 and class_name == 'dynamic':
            obstacle_size = closest_obstacle.get('size', 1.0)
            if isinstance(obstacle_size, tuple):
                obstacle_size = max(obstacle_size)

            wait_time = 1 + (obstacle_size - 1.0) * 0.5
            self.waiting = True
            self.wait_time = wait_time
            self.wait_start_time = time.time()
            self.wait_reason = f"AI-DYNAMIC (conf={confidence:.2f})"
            return True

        if self.map[target_pos] in (1, 'o'):
            return False

        direction = (target_pos[0] - self.current_pos[0], target_pos[1] - self.current_pos[1])
        distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)

        if distance < 1e-6:
            return False

        robot_speed = 1.0
        need_wait, wait_info = self.dynamic_obstacle_handler.apply_waiting_rule(
            self.current_pos, direction, robot_speed
        )

        if need_wait:
            stop_position, wait_time = wait_info
            self.wait_reason = "Collision predicted by velocity model"

            if stop_position != self.current_pos:
                dist = math.dist(self.current_pos, stop_position)
                self.energy -= 0.5 * dist
                self.rotate_to(stop_position)
                self.current_pos = stop_position
                ui.update_vehicle_pos(stop_position)
                ui.set_energy_display(self.energy)

            self.waiting = True
            self.wait_time = wait_time
            self.wait_start_time = time.time()
            return True

        return False

def main():
    global dynamic_obstacles
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.set_map(ENVIRONMENT)
    robot.set_special_areas(special_areas)

    global total_free_cells, blank_cells
    blank_cells = np.sum(ENVIRONMENT == 0)  # ✅ NEW: Calculate before dynamic obstacles
    total_free_cells = np.sum(ENVIRONMENT == 0)
    dynamic_obstacles = DynamicObstaclesManager(ui, num_obstacles=0, speed_factor=args.speed)

    if hasattr(ui, 'dynamic_obstacles') and ui.dynamic_obstacles:
        dynamic_obstacles.initialize_obstacles()
        robot.virtual_camera.dynamic_obstacles_manager = dynamic_obstacles

    for x in range(ROW_COUNT):
        for y in range(COL_COUNT):
            if ui.map[x, y] == 's':
                ui.map[x, y] = 0

    ui.draw_map()
    pg.display.flip()
    time.sleep(0.5)

    global execute_time
    execute_time = time.time()
    robot.run()

    print('\nCoverage:\t', coverage_length)
    print('Advance:\t', advance_length)
    print('Return:\t', retreat_length)
    print('-' * 8)
    print('Total Path Length:', total_travel_length)
    print('Time: ', execute_time)

    print('=' * 50)
    print(f'1. Total Path Length: {total_travel_length:.2f}')

    # ✅ FIXED: Use correct overlap formula to prevent negative values
    explored_cells = np.sum(robot.map == 'e')
    if explored_cells > 0:
        overlap_rate = (count_cell_go_through / explored_cells - 1) * 100
        print(f'2. Overlap Rate: {overlap_rate:.2f}%')
    else:
        print('2. Overlap Rate: 0.00%')
    print(f'3. Number of Returns: {return_charge_count}')
    print(f'4. Number of Deadlocks: {deadlock_count} (extreme: {extreme_deadlock_count})')
    print(f'5. Execution Time: {execute_time:.3f}s')
    # 6. Coverage Rate (NEW)
    cover_cells = len(covered_positions)  # ✅ NEW: Unique coverage cells only
    if blank_cells > 0:
        coverage_rate = (cover_cells / blank_cells) * 100
        if coverage_rate > 100:
            coverage_rate = 100
        print(f'6. Coverage Rate: {coverage_rate:.2f}%')
    else:
        print('6. Coverage Rate: 0.00%')

    print('=' * 50)

if __name__ == "__main__":
    main()