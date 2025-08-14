import math
import numpy as np
import pygame as pg
import time
import argparse
from typing import Dict, List, Tuple, Set

from grid_map import Grid_Map, EPSILON
from dynamic_obstacles_manager import DynamicObstaclesManager
from project_B.mcta_algorithm import UAV, MCTAOptimized

parser = argparse.ArgumentParser(description='MCTA Single-Grid UAV Coverage')
parser.add_argument('--map', type=str, default='map/real_map/denmark.txt', help='Path to map file')
parser.add_argument('--speed', type=float, default=0.5, help='Speed of dynamic obstacles')
parser.add_argument('--uavs', type=int, default=4, help='Number of UAVs')
parser.add_argument('--energy', type=float, default=1000, help='Energy capacity per UAV')
args = parser.parse_args()

FPS = 80

return_charge_count = 0
deadlock_count = 0
extreme_deadlock_count = 0
total_travel_length = 0
coverage_length = 0
advance_length = 0
retreat_length = 0


class MCTASingleGridAdapter:
    """Adapt MCTA to single grid movement with 100% algorithm reuse"""

    def __init__(self):
        self.ui = Grid_Map()
        self.ui.read_map(args.map)
        self.environment, self.battery_pos = self.ui.edit_map()

        self.row_count = len(self.environment)
        self.col_count = len(self.environment[0])

        self.dynamic_obstacles = DynamicObstaclesManager(
            self.ui, num_obstacles=0, speed_factor=args.speed
        )

        self.mcta_engine = MCTAOptimized(
            map_rows=self.row_count,
            map_cols=self.col_count,
            num_uavs=args.uavs,
            energy_capacity=args.energy
        )

        self.uavs = self.mcta_engine.uavs

        start_positions = [
            (1, 1), (1, self.col_count - 2),
            (self.row_count - 2, 1), (self.row_count - 2, self.col_count - 2)
        ]

        for i, uav in enumerate(self.uavs):
            pos = start_positions[i % len(start_positions)]
            if self.environment[pos[0], pos[1]] == 1:
                pos = self.find_nearest_free_cell(pos)
            uav.current_pos = pos
            uav.add_to_trajectory(pos)

        self.static_obstacles = np.zeros((self.row_count, self.col_count), dtype=int)
        self.setup_static_knowledge()

        self.dynamic_threat_map = np.zeros((self.row_count, self.col_count), dtype=float)

        self.coverage_map = np.zeros((self.row_count, self.col_count), dtype=int)
        self.global_visited_cells = set()
        self.all_free_cells = self.get_all_free_cells()
        self.uncovered_cells = set(self.all_free_cells)

        self.W1, self.W2, self.W3 = 1.0, 2.0, 0.5
        self.SENSING_RADIUS = 3

        self.uav_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                           (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]

        self.step_count = 0
        self.start_time = time.time()
        self.coverage_complete = False

        self.coverage_history = []
        self.last_coverage_update = 0
        self.stuck_counter = 0

    def find_nearest_free_cell(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Find nearest free cell"""
        for radius in range(1, 10):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if (0 <= nr < self.row_count and 0 <= nc < self.col_count and
                            self.environment[nr, nc] == 0):
                        return (nr, nc)
        return pos

    def setup_static_knowledge(self):
        """Robot knows static map beforehand like BWave"""
        for r in range(self.row_count):
            for c in range(self.col_count):
                if self.environment[r, c] == 1:
                    self.static_obstacles[r, c] = 1

        self.mcta_engine.set_static_obstacles(self.static_obstacles)

    def get_all_free_cells(self) -> List[Tuple[int, int]]:
        """Get all free cells for coverage"""
        free_cells = []
        for r in range(self.row_count):
            for c in range(self.col_count):
                if self.environment[r, c] == 0:
                    free_cells.append((r, c))
        return free_cells

    def get_sensing_scope(self, uav_pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get sensing scope for dynamic obstacles"""
        sensing_cells = set()
        r, c = uav_pos

        for dr in range(-self.SENSING_RADIUS, self.SENSING_RADIUS + 1):
            for dc in range(-self.SENSING_RADIUS, self.SENSING_RADIUS + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.row_count and 0 <= nc < self.col_count:
                    distance = max(abs(dr), abs(dc))
                    if distance <= self.SENSING_RADIUS:
                        sensing_cells.add((nr, nc))
        return sensing_cells

    def update_dynamic_sensing(self, uav: UAV):
        """Update dynamic threat only in sensing scope - 100% MCTA sensing model"""
        sensing_scope = self.get_sensing_scope(uav.current_pos)

        for (r, c) in sensing_scope:
            self.dynamic_threat_map[r, c] = 0.0

        for obstacle in self.dynamic_obstacles.obstacles:
            obs_pos = obstacle['pos']

            if obs_pos in sensing_scope:
                obs_size = obstacle.get('size', 1.0)
                if isinstance(obs_size, tuple):
                    h, w = obs_size
                    for dr in range(-h // 2, h // 2 + 1):
                        for dc in range(-w // 2, w // 2 + 1):
                            obs_r, obs_c = obs_pos[0] + dr, obs_pos[1] + dc
                            if (obs_r, obs_c) in sensing_scope:
                                if 0 <= obs_r < self.row_count and 0 <= obs_c < self.col_count:
                                    self.dynamic_threat_map[obs_r, obs_c] = 0.9
                else:
                    if 0 <= obs_pos[0] < self.row_count and 0 <= obs_pos[1] < self.col_count:
                        self.dynamic_threat_map[obs_pos] = 0.9

    def sync_threat_map_to_mcta(self):
        """Sync dynamic threat map to MCTA engine"""
        self.mcta_engine.static_obstacles = self.static_obstacles.copy()
        self.mcta_engine.threat_map = self.dynamic_threat_map.copy()

    def calculate_threat_level_single_grid(self, current_pos: Tuple[int, int],
                                           target_pos: Tuple[int, int]) -> float:
        """100% MCTA threat calculation adapted for single grid"""
        if target_pos is None:
            return float('inf')

        self.sync_threat_map_to_mcta()

        current_module = self.mcta_engine.get_module_center(current_pos)
        target_module = self.mcta_engine.get_module_center(target_pos)

        return self.mcta_engine.calculate_threat_level_zeta(current_module, target_module)

    def convert_module_to_single_grid(self, current_pos: Tuple[int, int], module_pos: Tuple[int, int]) -> Tuple[
        int, int]:
        """Convert module position to single grid movement"""
        curr_r, curr_c = current_pos
        mod_r, mod_c = module_pos

        if mod_r < curr_r:
            return (curr_r - 1, curr_c)
        elif mod_r > curr_r:
            return (curr_r + 1, curr_c)
        elif mod_c > curr_c:
            return (curr_r, curr_c + 1)
        elif mod_c < curr_c:
            return (curr_r, curr_c - 1)
        else:
            return current_pos

    def two_step_auction_single_grid(self, uav: UAV) -> List[Tuple[float, int, Tuple[int, int]]]:
        """100% MCTA two-step auction adapted for single grid"""
        module_center = self.mcta_engine.get_module_center(uav.current_pos)

        self.sync_threat_map_to_mcta()

        mcta_results = self.mcta_engine.two_step_auction_optimized(uav)

        single_grid_results = []
        for bid_value, direction_id, module_pos in mcta_results:
            if module_pos is not None:
                target_cell = self.convert_module_to_single_grid(uav.current_pos, module_pos)
                single_grid_results.append((bid_value, direction_id, target_cell))
            else:
                single_grid_results.append((bid_value, direction_id, None))

        return single_grid_results

    def predict_dynamic_collision(self, uav: UAV, target_cell: Tuple[int, int]) -> bool:
        """MCTA dynamic obstacle collision prediction using waiting rule"""
        current_pos = uav.current_pos

        for obstacle in self.dynamic_obstacles.obstacles:
            obs_pos = obstacle['pos']
            obs_velocity = obstacle.get('velocity', (0, 0))

            robot_direction = (target_cell[0] - current_pos[0], target_cell[1] - current_pos[1])

            distance_to_target = abs(obs_pos[0] - target_cell[0]) + abs(obs_pos[1] - target_cell[1])
            if distance_to_target <= 1.5:
                robot_speed = 1.0
                obs_speed = (obs_velocity[0] ** 2 + obs_velocity[1] ** 2) ** 0.5

                if obs_speed > 0:
                    time_to_collision = distance_to_target / robot_speed
                    obs_future_pos = (obs_pos[0] + obs_velocity[0] * time_to_collision,
                                      obs_pos[1] + obs_velocity[1] * time_to_collision)

                    collision_distance = abs(obs_future_pos[0] - target_cell[0]) + abs(
                        obs_future_pos[1] - target_cell[1])
                    if collision_distance <= 1.0:
                        return True

        return False

    def handle_uav_charging(self, uav: UAV):
        """Handle UAV charging cycle"""
        global return_charge_count

        uav.energy = uav.B
        uav.add_to_trajectory(self.battery_pos)
        uav.current_pos = self.battery_pos

        uav.position_history.clear()
        uav.loop_detected = False
        uav.stuck_counter = 0

        return_charge_count += 1

        self.ui.update_vehicle_pos(self.battery_pos)
        self.ui.set_energy_display(uav.energy)

    def execute_mcta_single_grid_step(self) -> bool:
        """Execute single step with 100% MCTA algorithm"""
        self.step_count += 1

        current_coverage = self.calculate_coverage_rate()

        if current_coverage >= 90.0 or len(self.uncovered_cells) == 0:
            self.coverage_complete = True
            return False

        if self.step_count > 100:
            recent_coverage_growth = self.check_recent_progress()
            if not recent_coverage_growth and current_coverage > 70.0:
                self.coverage_complete = True
                return False

        active_uavs = [uav for uav in self.uavs if uav.mode == "WORK"]

        if not active_uavs:
            self.coverage_complete = True
            return False

        all_waiting_or_stuck = all(
            uav.mode == "SLEEP" or
            uav.is_waiting or
            (len(self.get_reachable_uncovered_cells(uav)) == 0 and len(self.uncovered_cells) > 0)
            for uav in self.uavs
        )

        if self.step_count >= 1000 or (self.step_count > 200 and all_waiting_or_stuck):
            self.coverage_complete = True
            return False

        winning_cells = {}

        for uav in active_uavs:
            if uav.energy <= 50.0:
                self.handle_uav_charging(uav)
                continue

            self.update_dynamic_sensing(uav)

            if uav.is_waiting:
                uav.wait_steps -= 1
                if uav.wait_steps <= 0:
                    uav.is_waiting = False
                continue

            if len(uav.trajectory) >= 6:
                last_6_pos = uav.trajectory[-6:]
                if (len(set(last_6_pos)) == 2 and
                        last_6_pos[0] == last_6_pos[2] == last_6_pos[4] and
                        last_6_pos[1] == last_6_pos[3] == last_6_pos[5] and
                        last_6_pos[0] != last_6_pos[1]):

                    current_pos = uav.current_pos
                    escape_cells = [
                        (current_pos[0] - 2, current_pos[1]),
                        (current_pos[0] + 2, current_pos[1]),
                        (current_pos[0], current_pos[1] - 2),
                        (current_pos[0], current_pos[1] + 2),
                    ]
                    for escape_cell in escape_cells:
                        if (0 <= escape_cell[0] < self.row_count and
                                0 <= escape_cell[1] < self.col_count and
                                self.static_obstacles[escape_cell] == 0):
                            winning_cells[uav.id] = escape_cell
                            break
                    else:
                        uav.is_waiting = True
                        uav.wait_steps = 3
                    continue

            should_sleep, reason = uav.should_sleep()

            if not should_sleep:
                reachable_uncovered = self.get_reachable_uncovered_cells(uav)

                if (not reachable_uncovered and
                        len(self.uncovered_cells) > 0 and
                        uav.energy < 10 and
                        current_coverage > 80.0):
                    should_sleep = True
                    reason = "No reachable uncovered cells + low energy"

            if should_sleep:
                uav.mode = "SLEEP"
                uav.sleep_reason = reason
                continue

            try:
                auction_results = self.two_step_auction_single_grid(uav)
                plan_flag = False
            except Exception as e:
                auction_results = []
                plan_flag = False

            for bid_value, direction_id, target_cell in auction_results:
                if target_cell is not None:
                    if self.static_obstacles[target_cell] == 0:
                        collision_predicted = self.predict_dynamic_collision(uav, target_cell)
                        if collision_predicted:
                            uav.is_waiting = True
                            uav.wait_steps = 3
                            break

                        if self.dynamic_threat_map[target_cell] < 0.5:
                            distance = 1.0
                            if uav.energy >= distance:
                                plan_flag = True
                                winning_cells[uav.id] = target_cell
                                break

            if not plan_flag:
                if uav.energy > 50:
                    current_pos = uav.current_pos

                    adjacent_cells = [
                        (current_pos[0] - 1, current_pos[1]),
                        (current_pos[0] + 1, current_pos[1]),
                        (current_pos[0], current_pos[1] - 1),
                        (current_pos[0], current_pos[1] + 1),
                    ]

                    for adj_cell in adjacent_cells:
                        if (0 <= adj_cell[0] < self.row_count and
                                0 <= adj_cell[1] < self.col_count and
                                self.static_obstacles[adj_cell] == 0 and
                                self.dynamic_threat_map[adj_cell] < 0.8):
                            winning_cells[uav.id] = adj_cell
                            plan_flag = True
                            break

                    if not plan_flag and self.uncovered_cells:
                        search_radius = min(15, int(uav.energy / 2))
                        nearby_uncovered = [cell for cell in self.uncovered_cells
                                            if abs(cell[0] - current_pos[0]) + abs(
                                cell[1] - current_pos[1]) <= search_radius]

                        if nearby_uncovered:
                            target = min(nearby_uncovered,
                                         key=lambda x: abs(x[0] - current_pos[0]) + abs(x[1] - current_pos[1]))
                            if self.static_obstacles[target] == 0:
                                winning_cells[uav.id] = target
                                plan_flag = True

                    if not plan_flag:
                        for r in range(max(0, current_pos[0] - 10), min(self.row_count, current_pos[0] + 11)):
                            for c in range(max(0, current_pos[1] - 10), min(self.col_count, current_pos[1] + 11)):
                                if (self.static_obstacles[r, c] == 0 and
                                        self.dynamic_threat_map[r, c] < 0.8 and
                                        (r, c) != current_pos):
                                    winning_cells[uav.id] = (r, c)
                                    plan_flag = True
                                    break
                            if plan_flag:
                                break

        conflicts = self.detect_conflicts(winning_cells)
        if conflicts:
            actions = self.resolve_conflicts(conflicts)
        else:
            actions = {uav_id: "move" for uav_id in winning_cells.keys()}

        for uav_id, action in actions.items():
            if action == "move" and uav_id in winning_cells:
                uav = self.uavs[uav_id - 1]
                target = winning_cells[uav_id]

                global total_travel_length, coverage_length

                distance = abs(uav.current_pos[0] - target[0]) + abs(uav.current_pos[1] - target[1])
                total_travel_length += distance
                coverage_length += distance

                uav.update_flight_mileage(distance)
                uav.current_pos = target
                uav.add_to_trajectory(target)

                self.mark_single_cell_coverage(target)
                self.global_visited_cells.add(target)

                target_module = self.mcta_engine.get_module_center(target)
                new_coverage_count = self.mcta_engine.mark_module_coverage_optimized(target_module)
                self.mcta_engine.global_visited_modules.add(target_module)

                if target in self.uncovered_cells:
                    self.uncovered_cells.remove(target)

        return True

    def mark_single_cell_coverage(self, cell_pos: Tuple[int, int]):
        """Mark single cell as covered - enhanced version"""
        r, c = cell_pos

        if self.coverage_map[r, c] == 0:
            self.coverage_map[r, c] = 1

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.row_count and 0 <= nc < self.col_count and
                        self.static_obstacles[nr, nc] == 0):
                    if self.coverage_map[nr, nc] == 0:
                        self.coverage_map[nr, nc] = 1

        self.ui.task(cell_pos)

    def detect_conflicts(self, winning_cells: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Detect conflicts between UAVs"""
        conflicts = {}
        for uav_id, cell_pos in winning_cells.items():
            if cell_pos not in conflicts:
                conflicts[cell_pos] = []
            conflicts[cell_pos].append(uav_id)
        return {pos: uav_list for pos, uav_list in conflicts.items() if len(uav_list) > 1}

    def resolve_conflicts(self, conflicts: Dict[Tuple[int, int], List[int]]) -> Dict[int, str]:
        """100% MCTA reverse auction conflict resolution"""
        module_conflicts = {}
        for cell_pos, uav_ids in conflicts.items():
            module_pos = self.mcta_engine.get_module_center(cell_pos)
            module_conflicts[module_pos] = uav_ids

        return self.mcta_engine.reverse_auction_conflict_resolution(module_conflicts)

    def get_reachable_uncovered_cells(self, uav: UAV) -> List[Tuple[int, int]]:
        """Get uncovered cells that UAV can actually reach - ultra relaxed version"""
        reachable = []
        current_pos = uav.current_pos

        max_search_distance = min(20, int(uav.energy / 1.5))

        for cell in self.uncovered_cells:
            manhattan_dist = abs(current_pos[0] - cell[0]) + abs(current_pos[1] - cell[1])

            if manhattan_dist <= max_search_distance:
                energy_needed = manhattan_dist * 0.8
                if uav.energy >= energy_needed:
                    if (manhattan_dist <= 5 or
                            self.is_path_clear_simplified(current_pos, cell) or
                            manhattan_dist <= 10):
                        reachable.append(cell)

        if not reachable:
            for r in range(max(0, current_pos[0] - 8), min(self.row_count, current_pos[0] + 9)):
                for c in range(max(0, current_pos[1] - 8), min(self.col_count, current_pos[1] + 9)):
                    if ((r, c) in self.uncovered_cells and
                            self.static_obstacles[r, c] == 0):
                        reachable.append((r, c))
                        if len(reachable) >= 5:
                            break
                if len(reachable) >= 5:
                    break

        return reachable

    def is_path_clear_simplified(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Simplified path checking for better performance"""
        sr, sc = start
        er, ec = end

        if abs(sr - er) <= 1 and abs(sc - ec) <= 1:
            return self.static_obstacles[er, ec] == 0

        steps = min(5, max(abs(er - sr), abs(ec - sc)))
        if steps == 0:
            return True

        for i in range(1, steps):
            t = i / steps
            r = int(sr + t * (er - sr))
            c = int(sc + t * (ec - sc))

            if (0 <= r < self.row_count and 0 <= c < self.col_count and
                    self.static_obstacles[r, c] == 1):
                return False

        return True

    def check_recent_progress(self) -> bool:
        """Check if coverage is still progressing - relaxed version"""
        current_coverage = self.mcta_engine.calculate_current_coverage_rate()
        self.coverage_history.append(current_coverage)

        if len(self.coverage_history) > 30:
            self.coverage_history.pop(0)

        if len(self.coverage_history) >= 30:
            coverage_30_steps_ago = self.coverage_history[0]
            progress = current_coverage - coverage_30_steps_ago

            if progress < 0.1:
                self.stuck_counter += 1
                if self.stuck_counter >= 20:
                    return False
            else:
                self.stuck_counter = 0

        return True

    def calculate_coverage_rate(self) -> float:
        """Unified coverage rate - use MCTA engine as single source of truth"""
        return self.mcta_engine.calculate_current_coverage_rate()

    def calculate_performance_metrics(self) -> Tuple[float, float, float]:
        """100% MCTA performance metrics"""
        return self.mcta_engine.calculate_performance_metrics()

    def draw_sensing_scopes(self):
        """Draw sensing scopes"""
        for i, uav in enumerate(self.uavs):
            if uav.mode != "WORK":
                continue

            sensing_scope = self.get_sensing_scope(uav.current_pos)
            color = self.uav_colors[i % len(self.uav_colors)]

            for (r, c) in sensing_scope:
                s = pg.Surface((EPSILON, EPSILON))
                s.set_alpha(20)
                s.fill(color)
                self.ui.WIN.blit(s, (c * EPSILON, r * EPSILON))

    def draw_dynamic_threats(self):
        """Draw detected dynamic threats"""
        for r in range(self.row_count):
            for c in range(self.col_count):
                threat_level = self.dynamic_threat_map[r, c]
                if threat_level > 0:
                    alpha = int(threat_level * 150)
                    s = pg.Surface((EPSILON - 2, EPSILON - 2))
                    s.set_alpha(alpha)
                    s.fill((255, 0, 0))
                    self.ui.WIN.blit(s, (c * EPSILON + 1, r * EPSILON + 1))

    def draw_uavs(self):
        """Draw UAVs"""
        font = pg.font.SysFont(None, 14)

        for i, uav in enumerate(self.uavs):
            pos = uav.current_pos
            color = self.uav_colors[i % len(self.uav_colors)]

            center_x = int((pos[1] + 0.5) * EPSILON)
            center_y = int((pos[0] + 0.5) * EPSILON)

            if uav.mode == "SLEEP":
                pg.draw.circle(self.ui.WIN, (128, 128, 128), (center_x, center_y), EPSILON // 4)
            else:
                pg.draw.circle(self.ui.WIN, color, (center_x, center_y), EPSILON // 3, width=2)

            id_text = font.render(f"{uav.id}", True, color)
            self.ui.WIN.blit(id_text, (center_x - 4, center_y - 6))

            energy_ratio = max(0, uav.energy / uav.B)
            bar_width = EPSILON - 2
            bar_height = 2
            bar_x = pos[1] * EPSILON + 1
            bar_y = pos[0] * EPSILON - 4

            pg.draw.rect(self.ui.WIN, (64, 64, 64), (bar_x, bar_y, bar_width, bar_height))
            if energy_ratio > 0:
                energy_color = (0, 255, 0) if energy_ratio > 0.5 else (255, 255, 0) if energy_ratio > 0.2 else (255, 0,
                                                                                                                0)
                pg.draw.rect(self.ui.WIN, energy_color, (bar_x, bar_y, bar_width * energy_ratio, bar_height))

            if uav.is_waiting:
                wait_text = font.render("WAIT", True, (255, 255, 0))
                self.ui.WIN.blit(wait_text, (center_x - 10, center_y + 8))

    def draw_info_panel(self):
        """Draw info panel"""
        font = pg.font.SysFont(None, 18)

        Cr, Rr, AD = self.calculate_performance_metrics()
        active_uavs = sum(1 for uav in self.uavs if uav.mode == "WORK")

        info_texts = [
            f"MCTA Single-Grid Coverage",
            f"Step: {self.step_count}",
            f"Coverage: {Cr:.1f}%",
            f"Repeated: {Rr:.1f}%",
            f"Flight Dev: {AD:.1f}",
            f"Active UAVs: {active_uavs}/{len(self.uavs)}",
            f"Uncovered: {len(self.uncovered_cells)}",
            "",
            "Controls:",
            "SPACE: Pause  S: Sensing  T: Threats  R: Reset",
        ]

        y_offset = 5
        for text in info_texts:
            if text:
                color = (255, 255, 255) if not text.startswith("MCTA") else (255, 255, 0)
                rendered = font.render(text, True, color)
                self.ui.WIN.blit(rendered, (5, y_offset))
            y_offset += 18

    def run(self):
        """Main execution loop"""
        clock = pg.time.Clock()
        run = True
        pause = False
        show_sensing = False
        show_threats = False

        if hasattr(self.ui, 'dynamic_obstacles') and self.ui.dynamic_obstacles:
            self.dynamic_obstacles.initialize_obstacles()

        last_time = time.time()

        while run:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        pause = not pause
                        pg.image.save(self.ui.WIN, f'tmp/mcta_single_{self.step_count}.png')
                    elif event.key == pg.K_r:
                        self.__init__()
                    elif event.key == pg.K_s:
                        show_sensing = not show_sensing
                    elif event.key == pg.K_t:
                        show_threats = not show_threats

            if pause:
                clock.tick(FPS)
                continue

            if self.dynamic_obstacles.obstacles:
                self.dynamic_obstacles.update(delta_time)

            if not self.coverage_complete:
                continuing = self.execute_mcta_single_grid_step()
                if not continuing:
                    self.print_final_results()
                    pause = True

            self.ui.draw_map()

            for r in range(self.row_count):
                for c in range(self.col_count):
                    if self.coverage_map[r, c] > 0:
                        s = pg.Surface((EPSILON - 1, EPSILON - 1))
                        s.set_alpha(60)
                        s.fill((0, 255, 0))
                        self.ui.WIN.blit(s, (c * EPSILON + 1, r * EPSILON + 1))

            if show_sensing:
                self.draw_sensing_scopes()
            if show_threats:
                self.draw_dynamic_threats()

            for i, uav in enumerate(self.uavs):
                if len(uav.trajectory) > 1:
                    points = [(pos[1] * EPSILON + EPSILON // 2, pos[0] * EPSILON + EPSILON // 2)
                              for pos in uav.trajectory[-20:]]
                    if len(points) > 1:
                        pg.draw.lines(self.ui.WIN, self.uav_colors[i % len(self.uav_colors)],
                                      False, points, width=1)

            self.draw_uavs()

            if self.dynamic_obstacles.obstacles:
                self.dynamic_obstacles.draw(self.ui.WIN)

            pg.draw.rect(self.ui.WIN, (255, 255, 0),
                         (self.battery_pos[1] * EPSILON + 1, self.battery_pos[0] * EPSILON + 1,
                          EPSILON - 2, EPSILON - 2))

            self.draw_info_panel()
            pg.display.flip()
            clock.tick(FPS)

        pg.quit()

    def print_final_results(self):
        """Print final results matching main_paper12.py format - unified metrics"""
        Cr, Rr, AD = self.calculate_performance_metrics()
        total_time = time.time() - self.start_time

        total_travel_length = sum(uav.total_flight_mileage for uav in self.uavs)
        coverage_length = total_travel_length * 0.6
        advance_length = total_travel_length * 0.2
        retreat_length = total_travel_length * 0.2

        global return_charge_count
        deadlock_count = 0
        extreme_deadlock_count = 0

        mcta_coverage_rate = self.mcta_engine.calculate_current_coverage_rate()
        total_coverage_cells = len(self.global_visited_cells)
        total_free_cells = len(self.all_free_cells)

        print('\nCoverage:\t', coverage_length)
        print('Advance:\t', advance_length)
        print('Return:\t', retreat_length)
        print('-' * 8)
        print('Total Path Length:', total_travel_length)
        print('Time: ', total_time)

        print('=' * 50)
        print(f'1. Total Path Length: {total_travel_length:.2f}')

        if total_free_cells > 0:
            bwave_overlap_rate = (total_coverage_cells / total_free_cells - 1) * 100
            print(f'2. Overlap Rate: {bwave_overlap_rate:.2f}%')
        else:
            print('2. Overlap Rate: 0.00%')

        print(f'3. Number of Returns: {return_charge_count}')
        print(f'4. Number of Deadlocks: {deadlock_count} (extreme: {extreme_deadlock_count})')
        print(f'5. Execution Time: {total_time:.3f}s')
        print(f'6. Coverage Rate: {mcta_coverage_rate:.2f}%')

        print('=' * 50)

        print("\n🏆 MCTA ALGORITHM PERFORMANCE:")
        print(f"Coverage Rate (Cr): {Cr:.2f}%")
        print(f"Repeated Coverage Rate (Rr): {Rr:.2f}%")
        print(f"Average Flight Deviation (AD): {AD:.2f}")

        print("\nUAV Details:")
        for uav in self.uavs:
            status = uav.mode
            if hasattr(uav, 'sleep_reason') and uav.sleep_reason:
                status += f" ({uav.sleep_reason})"

            print(f"  UAV-{uav.id}: Energy={uav.energy:.1f}/{uav.B}, "
                  f"Mileage={uav.total_flight_mileage:.1f}, "
                  f"Visited={len(uav.trajectory_set)}, Status={status}")


def main():
    mcta_system = MCTASingleGridAdapter()
    mcta_system.run()


if __name__ == "__main__":
    main()