import math
import numpy as np
from collections import deque
import pygame as pg
import time
from copy import deepcopy

from a_star_projectB import GridMapGraph, a_star_search
from dynamic_obstacle_projectB import DynamicObstacle
from dynamic_obstacle_random_projectB import DynamicObstacleRandom
from logic_projectB import LogicAlgorithm, Q
from grid_map_projectB import Grid_Map

VISION_SENSOR_RANGE = 5

# MCTA Framework Configuration
ENERGY_CAPACITY = math.inf
MCTA_ENABLE_DYNAMIC_REPLANNING = True
MCTA_THREAT_THRESHOLD = 80  # Threshold for high-threat dynamic obstacles

ui = Grid_Map()
ui.read_map('map/scenario_2/test_1.txt')
ENVIRONMENT, battery_pos = ui.edit_map()
# ui.save_map('map/map_test.txt')

ROW_COUNT = len(ENVIRONMENT)
COL_COUNT = len(ENVIRONMENT[0])

FPS = 40

NUMS_SAMPLE = 5000
MIN_PROB_THRESHOLD = 3

total_travel_length = 0
coverage_length, retreat_length, advance_length = 0, 0, 0
coverage_ratio = 0
nums_cell_repetition = 0
repetition_rate = 0
return_charge_count = 1
count_waiting = 0

dynamic_obs_list: list[DynamicObstacle] = []
dynamic_obs_list.append(DynamicObstacle((3, 6), (2, 1), 4, 10))

consecutive_wait_obs_list = [0] * len(dynamic_obs_list)


def check_valid_pos(pos):
    row, col = pos
    if row < 0 or row >= ROW_COUNT: return False
    if col < 0 or col >= COL_COUNT: return False
    return True


def sign(n):
    return int(np.sign(n))


class Robot:
    def __init__(self, battery_pos, map_row_count, map_col_count):
        self.logic = LogicAlgorithm(map_row_count, map_col_count)

        # MCTA Framework State
        self.uav_id = 0
        self.mcta_mode = True  # Always use MCTA as primary

        # Maps for MCTA
        self.static_map = None
        self.dynamic_map = None
        self.predict_map = None
        self.prob_map = None
        self.seen_map = None
        self.threat_map = None

        self.current_pos = battery_pos
        self.angle = -math.pi / 2
        self.velocity = 10

        self.battery_pos = battery_pos
        self.energy = ENERGY_CAPACITY

        self.move_status = 0  # 0: normal coverage, 1: retreat, 2: charge, 3: advance
        self.cache_path = []
        self.repeated_cells = []

        # Dynamic obstacle detection
        self.obs_prev_detected_dict = dict()
        self.obs_detected_dict = dict()
        self.scan_freq = 2
        self.alpha_1 = 0.3
        self.alpha_2 = 0.7
        self.alpha_3 = 0.3
        self.alpha_4 = 0.7

        # MCTA Statistics
        self.mcta_auction_count = 0
        self.mcta_fallback_count = 0
        self.mcta_dynamic_replan_count = 0

    def init_static_map(self, environment):
        row_count, col_count = len(environment), len(environment[0])
        self.static_map = deepcopy(environment)
        self.dynamic_map = deepcopy(environment)
        self.predict_map = deepcopy(environment)
        self.prob_map = np.zeros((row_count, col_count))
        self.seen_map = deepcopy(environment)
        self.threat_map = np.zeros((row_count, col_count))

        # Initialize MCTA maps
        self.predict_map[self.battery_pos] = self.dynamic_map[self.battery_pos] = 2
        self.seen_map[self.battery_pos] = 2

        # Initialize logic with MCTA framework
        self.logic.init_weight_map(environment)
        self.logic.set_threat_map(self.threat_map)

        print("MCTA Framework initialized")

    def update_dynamic_map(self, loop_count):
        row_count, col_count = len(self.static_map), len(self.static_map[0])

        # Reset dynamic states
        for x in range(row_count):
            for y in range(col_count):
                if self.dynamic_map[x, y] == 3 or self.dynamic_map[x, y] == 4:
                    self.dynamic_map[x, y] = self.static_map[x, y]
                if self.predict_map[x, y] == 3 or self.predict_map[x, y] == 4:
                    self.predict_map[x, y] = self.static_map[x, y]

        # Update dynamic obstacles
        for obs in dynamic_obs_list:
            if loop_count % obs.velocity == 0:
                obs.move_one_step(self.static_map)

            for dx in range(obs.height):
                for dy in range(obs.width):
                    x, y = obs.cur_row + dx, obs.cur_col + dy
                    if self.current_pos == (x, y):
                        print("Total length: ", coverage_length)
                        self.calculate_coverage_ratio()
                        print("Cr: ", coverage_ratio)
                        print("Rr: ", repetition_rate)
                        raise Exception('Collision with obstacle')

                    self.dynamic_map[x, y] = 3
        ui.set_map(self.dynamic_map)

    def update_probability_map_and_seen_map(self):
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

        # Update with detected obstacles
        detected_obs = self.obs_sensor(vision_range=VISION_SENSOR_RANGE)
        obs_potential_next_move = []
        obs_occupy_list = []

        for obs in detected_obs:
            self.calculateProbabilityMap(obs)
            for row in range(row_count):
                for col in range(col_count):
                    if self.prob_map[row, col] != 0 and self.dynamic_map[row, col] != 1:
                        obs_potential_next_move.append((row, col))
            obs_potential_next_move += self.get_potential_positions(obs)
            obs_occupy_list += obs.get_current_occupy_positions()

        for pos in obs_potential_next_move:
            self.dynamic_map[pos] = 4

        for pos in obs_occupy_list:
            self.dynamic_map[pos] = 3
            self.seen_map[pos] = 3
            self.prob_map[pos] = 100

        # Update threat map for MCTA
        self.threat_map = self.prob_map.copy()

        ui.set_map(self.dynamic_map)

    def run(self):
        global nums_cell_repetition, FPS
        clock = pg.time.Clock()
        run = True
        pause = False
        coverage_finish = False

        loop_count = 0
        print("Starting MCTA Coverage Algorithm")

        while run:
            loop_count += 1
            ui.draw()
            clock.tick(FPS)

            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:  # pause
                        pause = not pause
                    elif event.key == pg.K_LEFT:  # slow down
                        FPS /= 2
                        print(f"Speed: {FPS} FPS")
                    elif event.key == pg.K_RIGHT:  # speed up
                        FPS *= 2
                        print(f"Speed: {FPS} FPS")
                if event.type == pg.QUIT:
                    run = False

            if pause:
                continue

            if coverage_finish:
                continue

            self.task()
            self.update_dynamic_map(loop_count)

            if loop_count % self.velocity != 0:
                continue

            self.update_probability_map_and_seen_map()

            # === MCTA FRAMEWORK MAIN LOOP ===
            # Always use MCTA two-step auction as primary algorithm
            self.logic.set_map(self.seen_map)
            self.logic.set_prob_map(self.prob_map)
            self.logic.set_threat_map(self.threat_map)

            # Check for dynamic obstacles
            has_dynamic_threats = self.detect_dynamic_obs(VISION_SENSOR_RANGE)

            if has_dynamic_threats and MCTA_ENABLE_DYNAMIC_REPLANNING:
                # Use dynamic replanning with auction
                print(f"Dynamic threats detected at {self.current_pos}")
                np.savetxt('array.txt', self.prob_map, fmt='%f', delimiter=' ', newline='\r\n')

                max_bid_value, replan_wp = self.logic.get_replan_wp(
                    self.current_pos,
                    uav_id=self.uav_id,
                    current_energy=self.energy
                )

                if replan_wp is not None:
                    wp = [replan_wp]
                    self.mcta_dynamic_replan_count += 1
                    print(f"Dynamic replan to: {replan_wp} (bid: {max_bid_value:.3f})")
                else:
                    # Fallback to normal MCTA
                    wp = self.logic.get_wp(
                        self.current_pos,
                        uav_id=self.uav_id,
                        current_energy=self.energy
                    )
            else:
                # Normal MCTA two-step auction
                wp = self.logic.get_wp(
                    self.current_pos,
                    uav_id=self.uav_id,
                    current_energy=self.energy
                )

            # Execute movement based on MCTA state
            if self.logic.state == Q.NORMAL:
                if len(wp) > 0:
                    selected_cell = self.select_from_wp(wp)
                    self.move_to(selected_cell)
                    self.mcta_auction_count += 1

            elif self.logic.state == Q.DEADLOCK:
                print(f"MCTA Deadlock handling at {self.current_pos}")
                path = self.logic.escape_deadlock_path(self.current_pos)

                if path == []:
                    # Coverage complete
                    print("MCTA Coverage Complete!")
                    print(f"Auction decisions: {self.mcta_auction_count}")
                    print(f"Fallback decisions: {self.mcta_fallback_count}")
                    print(f"Dynamic replans: {self.mcta_dynamic_replan_count}")
                    pg.image.save(ui.WIN, "C:/Users/Admin/Downloads/mcta_complete.png")
                    coverage_finish = True
                    continue
                else:
                    # Use MCTA for deadlock escape
                    if has_dynamic_threats:
                        _, deadlock_wp = self.logic.get_replan_wp(
                            self.current_pos,
                            uav_id=self.uav_id,
                            current_energy=self.energy
                        )
                        if deadlock_wp is not None and self.prob_map[self.current_pos] < MIN_PROB_THRESHOLD:
                            if self.prob_map[path[0]] > 0:
                                continue
                        target = deadlock_wp if deadlock_wp is not None else path[0]
                    else:
                        target = path[0]

                    self.move_to(target)

            elif self.logic.state == Q.FINISH:
                print("MCTA: Energy exhausted or task complete")
                coverage_finish = True

        return

    # Keep existing utility methods
    def get_border_cells(self, cur_pos):
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

    def obstruct_cell_list(self, pos_from, pos_to, strict=False):
        threshold = 0.3
        start = (pos_from[0] + 0.5, pos_from[1] + 0.5)
        goal = (pos_to[0] + 0.5, pos_to[1] + 0.5)

        vecto = (goal[0] - start[0], goal[1] - start[1])
        angle = -np.arctan2(vecto[0], vecto[1])

        (x, y) = pos_from
        cell_list = [pos_from]

        sx, sy = sign(vecto[0]), sign(vecto[1])
        dx = abs(0.5 / math.sin(angle)) if vecto[0] != 0 else math.inf
        dy = abs(0.5 / math.cos(angle)) if vecto[1] != 0 else math.inf
        sum_x, sum_y = dx, dy

        while (x, y) != pos_to:
            (movx, movy) = (sum_x < sum_y or math.isclose(sum_x, sum_y), sum_y < sum_x or math.isclose(sum_x, sum_y))

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

    def select_from_wp(self, wp):
        return min(wp, key=self.travel_cost)

    def obs_sensor(self, vision_range=VISION_SENSOR_RANGE):
        obs_detected_list = []
        in_sensor_list = []
        border_cells = self.get_border_cells(self.current_pos)

        for pos in border_cells:
            obstruct_cell_list = self.obstruct_cell_list(self.current_pos, pos)
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

    def get_neighbours(self, cur_pos, size):
        cur_x, cur_y = cur_pos
        neighbours = []
        for x in range(cur_x - size, cur_x + size + 1):
            for y in range(cur_y - size, cur_y + size + 1):
                if check_valid_pos((x, y)) == False: continue
                if self.dynamic_map[x, y] == 1 or self.dynamic_map[x, y] == 3:
                    continue
                if (x, y) == cur_pos:
                    continue
                neighbours.append((x, y))
        return neighbours

    def get_potential_positions(self, obs: DynamicObstacle):
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

    def sample(self, z):
        rand = np.random.uniform(-z, z, 12)
        return np.sum(rand) * 1 / 12

    def sampling(self, obs: DynamicObstacle):
        (x, y) = obs.get_pos()
        v_prime = obs.v + self.sample(self.alpha_1 * abs(obs.velocity) + self.alpha_2 * abs(obs.w))
        w_prime = obs.w + self.sample(self.alpha_3 * abs(obs.velocity) + self.alpha_4 * abs(obs.w))
        x_prime = x - v_prime / w_prime * math.sin(obs.theta) + v_prime / w_prime * math.sin(
            obs.theta + self.scan_freq * w_prime)
        y_prime = y + v_prime / w_prime * math.cos(obs.theta) - v_prime / w_prime * math.cos(
            obs.theta + self.scan_freq * w_prime)
        return (round(x_prime), round(y_prime))

    def calculateProbabilityMap(self, obs: DynamicObstacle):
        new_pos_dict = dict()
        for _ in range(NUMS_SAMPLE):
            new_pos = self.sampling(obs)
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

    def detect_dynamic_obs(self, vision_range=VISION_SENSOR_RANGE):
        in_sensor_list = []
        border_cells = self.get_border_cells(self.current_pos)
        for pos in border_cells:
            obstruct_cell_list = self.obstruct_cell_list(self.current_pos, pos)
            for cell in obstruct_cell_list:
                if self.dynamic_map[cell] == 1:
                    break
                if cell not in in_sensor_list:
                    in_sensor_list.append(cell)

        for cell in in_sensor_list:
            if self.dynamic_map[cell] == 3:
                return True
        return False

    def task(self):
        current_pos = self.current_pos
        self.static_map[current_pos] = 2
        self.logic.update_explored(current_pos)
        ui.task(current_pos)

    def move_to(self, pos):
        if self.dynamic_map[pos] == 3:
            raise Exception('Collision with obstacle')

        global total_travel_length, coverage_length, retreat_length, advance_length
        dist = energy = math.dist(self.current_pos, pos)

        if self.move_status in (1, 3):
            energy = 0.5 * energy

        if self.energy < energy:
            raise Exception('Robot run out of battery')
        self.energy -= energy

        self.rotate_to(pos)
        self.current_pos = pos

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
        angle = -np.arctan2(vecto[0], vecto[1])
        return angle % (2 * math.pi)

    def rotate_to(self, pos_to):
        self.angle = self.get_angle(pos_to)

    def calculate_coverage_ratio(self):
        total_coverable_cells, nums_covered_cells = 0, 0
        global coverage_ratio, repetition_rate
        for rows in self.static_map:
            for i in rows:
                if i == 0:
                    total_coverable_cells += 1
                elif i == 2:
                    total_coverable_cells += 1
                    nums_covered_cells += 1
        coverage_ratio = round(nums_covered_cells / total_coverable_cells * 100, 2)
        if nums_covered_cells > 0:
            repetition_rate = round(nums_cell_repetition / nums_covered_cells * 100, 2)


def main():
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.init_static_map(ENVIRONMENT)
    robot.run()
    robot.calculate_coverage_ratio()

    print('\n=== MCTA Coverage Results ===')
    print('Coverage:\t', coverage_length)
    print('Retreat:\t', retreat_length)
    print('Advance:\t', advance_length)
    print('Coverage Ratio:\t', coverage_ratio)
    print('Repetition Rate:\t', repetition_rate)
    print('-' * 8)
    print('Total:', total_travel_length)
    print('\nMCTA Statistics:')
    print(f'Auction decisions: {robot.mcta_auction_count}')
    print(f'Fallback decisions: {robot.mcta_fallback_count}')
    print(f'Dynamic replans: {robot.mcta_dynamic_replan_count}')


if __name__ == "__main__":
    main()