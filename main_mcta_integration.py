import math
import numpy as np
import pygame as pg
import time
import argparse
from typing import Dict, List, Tuple, Set

# Import từ project A (giữ nguyên)
from grid_map import Grid_Map, EPSILON
from dynamic_obstacles_manager import DynamicObstaclesManager

# Import và adapt MCTA algorithm
from project_B.mcta_algorithm import MCTAOptimized, UAV

parser = argparse.ArgumentParser(description='MCTA Single-Grid UAV Coverage')
parser.add_argument('--map', type=str, default='map/real_map/denmark.txt', help='Path to map file')
parser.add_argument('--speed', type=float, default=0.5, help='Speed of dynamic obstacles')
parser.add_argument('--uavs', type=int, default=10, help='Number of UAVs')
parser.add_argument('--energy', type=float, default=2000, help='Energy capacity per UAV')
args = parser.parse_args()

FPS = 20


class MCTASingleGridAdapter:
    """Adapt MCTA to single grid movement với static map knowledge"""

    def __init__(self):
        # 1. Initialize Project A Environment
        self.ui = Grid_Map()
        self.ui.read_map(args.map)
        self.environment, self.battery_pos = self.ui.edit_map()

        self.row_count = len(self.environment)
        self.col_count = len(self.environment[0])

        # 2. Initialize Dynamic Obstacles Manager
        self.dynamic_obstacles = DynamicObstaclesManager(
            self.ui, num_obstacles=0, speed_factor=args.speed
        )

        # 3. Setup UAVs manually (không dùng MCTAOptimized constructor)
        self.uavs: List[UAV] = []
        self.setup_uavs()

        # 4. Static map knowledge (robot biết trước như BWave)
        self.static_obstacles = np.zeros((self.row_count, self.col_count), dtype=int)
        self.setup_static_knowledge()

        # 5. Dynamic threat map (chỉ từ sensing)
        self.dynamic_threat_map = np.zeros((self.row_count, self.col_count), dtype=float)

        # 6. Coverage tracking - SINGLE GRID
        self.coverage_map = np.zeros((self.row_count, self.col_count), dtype=int)
        self.global_visited_cells = set()
        self.all_free_cells = self.get_all_free_cells()
        self.uncovered_cells = set(self.all_free_cells)

        # 7. MCTA parameters adapted for single grid
        self.W1, self.W2, self.W3 = 1.0, 2.0, 0.5
        self.SENSING_RADIUS = 3  # Reduced for single grid

        # 8. Visualization
        self.uav_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        # 9. Performance tracking
        self.step_count = 0
        self.start_time = time.time()
        self.coverage_complete = False

        print(f"🚁 MCTA Single-Grid Setup Complete!")
        print(f"Environment: {self.row_count}x{self.col_count}")
        print(f"Free cells: {len(self.all_free_cells)}")
        print(f"UAVs: {len(self.uavs)}")

    def setup_uavs(self):
        """Setup UAVs manually"""
        start_positions = [
            (1, 1), (1, self.col_count - 2),
            (self.row_count - 2, 1), (self.row_count - 2, self.col_count - 2)
        ]

        for i in range(args.uavs):
            pos = start_positions[i % len(start_positions)]
            # Find valid starting position
            if self.environment[pos[0], pos[1]] == 1:
                pos = self.find_nearest_free_cell(pos)

            uav = UAV(i + 1, pos, args.energy)
            uav.add_to_trajectory(pos)
            self.uavs.append(uav)

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
        """Robot biết static map trước như BWave"""
        for r in range(self.row_count):
            for c in range(self.col_count):
                if self.environment[r, c] == 1:
                    self.static_obstacles[r, c] = 1

    def get_all_free_cells(self) -> List[Tuple[int, int]]:
        """Get all free cells for coverage"""
        free_cells = []
        for r in range(self.row_count):
            for c in range(self.col_count):
                if self.environment[r, c] == 0:  # Free cell
                    free_cells.append((r, c))
        return free_cells

    def get_sensing_scope(self, uav_pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get sensing scope cho dynamic obstacles"""
        sensing_cells = set()
        r, c = uav_pos

        for dr in range(-self.SENSING_RADIUS, self.SENSING_RADIUS + 1):
            for dc in range(-self.SENSING_RADIUS, self.SENSING_RADIUS + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.row_count and 0 <= nc < self.col_count:
                    distance = max(abs(dr), abs(dc))  # Chebyshev distance
                    if distance <= self.SENSING_RADIUS:
                        sensing_cells.add((nr, nc))
        return sensing_cells

    def update_dynamic_sensing(self, uav: UAV):
        """Update dynamic threat chỉ trong sensing scope"""
        sensing_scope = self.get_sensing_scope(uav.current_pos)

        # Reset dynamic threats trong sensing scope
        for (r, c) in sensing_scope:
            self.dynamic_threat_map[r, c] = 0.0

        # Detect dynamic obstacles
        for obstacle in self.dynamic_obstacles.obstacles:
            obs_pos = obstacle['pos']
            obs_size = obstacle.get('size', 1.0)

            # Check if obstacle is in sensing scope
            if obs_pos in sensing_scope:
                if isinstance(obs_size, tuple):
                    h, w = obs_size
                    for dr in range(-h // 2, h // 2 + 1):
                        for dc in range(-w // 2, w // 2 + 1):
                            obs_r, obs_c = obs_pos[0] + dr, obs_pos[1] + dc
                            if (obs_r, obs_c) in sensing_scope:
                                self.dynamic_threat_map[obs_r, obs_c] = 0.9
                else:
                    self.dynamic_threat_map[obs_pos] = 0.9

    def get_four_adjacent_cells(self, uav_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4 adjacent cells (UP, RIGHT, DOWN, LEFT)"""
        r, c = uav_pos
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT

        adjacent_cells = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.row_count and 0 <= nc < self.col_count:
                adjacent_cells.append((nr, nc))
            else:
                adjacent_cells.append(None)  # Out of bounds

        return adjacent_cells

    def calculate_threat_level_single_grid(self, current_pos: Tuple[int, int],
                                           target_pos: Tuple[int, int]) -> float:
        """Calculate threat level cho single grid movement"""
        if target_pos is None:
            return float('inf')

        r, c = target_pos

        # Static obstacle - robot biết trước
        if self.static_obstacles[r, c] == 1:
            return float('inf')

        # Dynamic threat - chỉ từ sensing
        dynamic_threat = self.dynamic_threat_map[r, c]

        # Base threat calculation
        threat = dynamic_threat

        # Add surrounding area check (S1, S2, S3 concept adapted)
        surrounding_threat = 0.0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.row_count and 0 <= nc < self.col_count:
                    if abs(dr) + abs(dc) == 1:  # Adjacent cells (S2)
                        surrounding_threat += self.dynamic_threat_map[nr, nc] * self.W2
                    elif abs(dr) + abs(dc) == 2:  # Diagonal cells (S1, S3)
                        surrounding_threat += self.dynamic_threat_map[nr, nc] * self.W1

        return threat + surrounding_threat * 0.1

    def two_step_auction_single_grid(self, uav: UAV) -> List[Tuple[float, int, Tuple[int, int]]]:
        """Adapted two-step auction cho single grid"""
        current_pos = uav.current_pos
        four_cells = self.get_four_adjacent_cells(current_pos)
        bid_results = []

        for i in range(4):
            target_cell = four_cells[i]

            if target_cell is None:
                bid_results.append((0.0, i + 1, None))
                continue

            # Calculate threat level
            threat_i = self.calculate_threat_level_single_grid(current_pos, target_cell)

            # Two-step lookahead
            future_cells = self.get_four_adjacent_cells(target_cell)
            future_threats = []

            for future_cell in future_cells:
                if future_cell is not None and future_cell != current_pos:
                    future_threat = self.calculate_threat_level_single_grid(target_cell, future_cell)
                    if future_threat != float('inf'):
                        future_threats.append(future_threat)

            threat_m = max(future_threats) if future_threats else 0.0

            # Calculate bid value
            if threat_i == float('inf'):
                ci = 0.0
            elif (threat_i + threat_m) > 0:
                ci = 1.0 / (threat_i + threat_m)
            else:
                # Priority factors
                base_value = 10000.0

                # Uncovered cell bonus
                if target_cell in self.uncovered_cells:
                    base_value += 100000.0

                # Not visited bonus
                if target_cell not in uav.trajectory_set:
                    base_value += 50000.0

                # Distance penalty
                distance = abs(current_pos[0] - target_cell[0]) + abs(current_pos[1] - target_cell[1])
                base_value -= distance * 1000.0

                # Recent visit penalty
                recent_trajectory = uav.trajectory[-10:] if len(uav.trajectory) >= 10 else uav.trajectory
                visit_count = recent_trajectory.count(target_cell)
                base_value -= visit_count * 20000.0

                ci = max(base_value, 1.0)

            bid_results.append((ci, i + 1, target_cell))

        # Sort by bid value with direction priority
        priority_map = {1: 4, 2: 3, 3: 2, 4: 1}  # UP, RIGHT, DOWN, LEFT
        bid_results.sort(key=lambda x: (x[0], priority_map[x[1]]), reverse=True)

        return bid_results

    def execute_mcta_single_grid_step(self) -> bool:
        """Execute single step with single grid movement"""
        self.step_count += 1

        # Update coverage calculation
        current_coverage = self.calculate_coverage_rate()

        # Check completion
        if current_coverage >= 90.0 or len(self.uncovered_cells) == 0:
            self.coverage_complete = True
            return False

        active_uavs = [uav for uav in self.uavs if uav.mode == "WORK"]
        if not active_uavs:
            self.coverage_complete = True
            return False

        winning_cells = {}

        for uav in active_uavs:
            # 1. Update dynamic sensing
            self.update_dynamic_sensing(uav)

            # 2. Handle waiting
            if uav.is_waiting:
                uav.wait_steps -= 1
                if uav.wait_steps <= 0:
                    uav.is_waiting = False
                continue

            # 3. Check sleep conditions
            should_sleep, reason = uav.should_sleep()
            if should_sleep:
                uav.mode = "SLEEP"
                uav.sleep_reason = reason
                continue

            # 4. Single-grid auction
            auction_results = self.two_step_auction_single_grid(uav)
            plan_flag = False

            for bid_value, direction_id, target_cell in auction_results:
                if target_cell is not None:
                    # Check if reachable (not static obstacle)
                    if self.static_obstacles[target_cell] == 0:
                        # Check dynamic obstacle collision
                        if self.dynamic_threat_map[target_cell] < 0.5:  # Safe threshold
                            # Check energy
                            distance = 1.0  # Single grid step
                            if uav.energy >= distance * 2:  # Energy buffer
                                plan_flag = True
                                winning_cells[uav.id] = target_cell
                                break

            # If no valid move, try any uncovered cell nearby
            if not plan_flag and self.uncovered_cells:
                current_pos = uav.current_pos
                nearby_uncovered = [cell for cell in self.uncovered_cells
                                    if abs(cell[0] - current_pos[0]) + abs(cell[1] - current_pos[1]) <= 3]
                if nearby_uncovered:
                    target = min(nearby_uncovered,
                                 key=lambda x: abs(x[0] - current_pos[0]) + abs(x[1] - current_pos[1]))
                    if self.static_obstacles[target] == 0 and self.dynamic_threat_map[target] < 0.5:
                        winning_cells[uav.id] = target

        # 5. Conflict resolution
        conflicts = self.detect_conflicts(winning_cells)
        if conflicts:
            actions = self.resolve_conflicts(conflicts)
        else:
            actions = {uav_id: "move" for uav_id in winning_cells.keys()}

        # 6. Execute movements
        for uav_id, action in actions.items():
            if action == "move" and uav_id in winning_cells:
                uav = self.uavs[uav_id - 1]
                target = winning_cells[uav_id]

                # Move UAV
                distance = 1.0  # Single grid
                uav.update_flight_mileage(distance)
                uav.current_pos = target
                uav.add_to_trajectory(target)

                # Mark coverage
                self.coverage_map[target] = 1
                self.global_visited_cells.add(target)
                if target in self.uncovered_cells:
                    self.uncovered_cells.remove(target)

        return True

    def detect_conflicts(self, winning_cells: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Detect conflicts between UAVs"""
        conflicts = {}
        for uav_id, cell_pos in winning_cells.items():
            if cell_pos not in conflicts:
                conflicts[cell_pos] = []
            conflicts[cell_pos].append(uav_id)
        return {pos: uav_list for pos, uav_list in conflicts.items() if len(uav_list) > 1}

    def resolve_conflicts(self, conflicts: Dict[Tuple[int, int], List[int]]) -> Dict[int, str]:
        """Resolve conflicts using reverse auction"""
        uav_actions = {}
        for cell_pos, conflicted_uav_ids in conflicts.items():
            # Select UAV with least flight mileage
            min_mileage = float('inf')
            selected_uav_id = None

            for uav_id in conflicted_uav_ids:
                uav = self.uavs[uav_id - 1]
                if uav.total_flight_mileage < min_mileage:
                    min_mileage = uav.total_flight_mileage
                    selected_uav_id = uav_id

            for uav_id in conflicted_uav_ids:
                if uav_id == selected_uav_id:
                    uav_actions[uav_id] = "move"
                else:
                    uav_actions[uav_id] = "wait"
                    self.uavs[uav_id - 1].is_waiting = True
                    self.uavs[uav_id - 1].wait_steps = 1

        return uav_actions

    def calculate_coverage_rate(self) -> float:
        """Calculate coverage rate"""
        covered_cells = np.sum(self.coverage_map > 0)
        total_free_cells = len(self.all_free_cells)
        return (covered_cells / total_free_cells) * 100.0 if total_free_cells > 0 else 0.0

    def calculate_performance_metrics(self) -> Tuple[float, float, float]:
        """Calculate performance metrics"""
        # Coverage rate
        Cr = self.calculate_coverage_rate()

        # Repeated coverage rate (simplified for single grid)
        total_visits = sum(len(uav.trajectory) for uav in self.uavs)
        unique_visits = len(self.global_visited_cells)
        Rr = ((total_visits - unique_visits) / max(unique_visits, 1)) * 100.0

        # Average flight deviation
        if len(self.uavs) > 0:
            L_bar = sum(uav.total_flight_mileage for uav in self.uavs) / len(self.uavs)
            AD = sum(abs(uav.total_flight_mileage - L_bar) for uav in self.uavs) / len(self.uavs)
        else:
            AD = 0.0

        return Cr, Rr, AD

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

            # UAV ID
            id_text = font.render(f"{uav.id}", True, color)
            self.ui.WIN.blit(id_text, (center_x - 4, center_y - 6))

            # Energy bar
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
            "SPACE: Pause  S: Sensing  R: Reset",
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

        # Initialize dynamic obstacles
        if hasattr(self.ui, 'dynamic_obstacles') and self.ui.dynamic_obstacles:
            self.dynamic_obstacles.initialize_obstacles()

        last_time = time.time()

        while run:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Events
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

            # Update dynamic obstacles
            if self.dynamic_obstacles.obstacles:
                self.dynamic_obstacles.update(delta_time)

            # Execute MCTA algorithm
            if not self.coverage_complete:
                continuing = self.execute_mcta_single_grid_step()
                if not continuing:
                    print(f"\n🎉 MCTA Single-Grid Coverage Complete!")
                    self.print_final_results()
                    pause = True

            # Visualization
            self.ui.draw_map()

            # Draw coverage
            for r in range(self.row_count):
                for c in range(self.col_count):
                    if self.coverage_map[r, c] > 0:
                        s = pg.Surface((EPSILON - 1, EPSILON - 1))
                        s.set_alpha(60)
                        s.fill((0, 255, 0))
                        self.ui.WIN.blit(s, (c * EPSILON + 1, r * EPSILON + 1))

            # Optional visualizations
            if show_sensing:
                self.draw_sensing_scopes()
            if show_threats:
                self.draw_dynamic_threats()

            # Draw trajectories
            for i, uav in enumerate(self.uavs):
                if len(uav.trajectory) > 1:
                    points = [(pos[1] * EPSILON + EPSILON // 2, pos[0] * EPSILON + EPSILON // 2)
                              for pos in uav.trajectory[-20:]]  # Last 20 steps only
                    if len(points) > 1:
                        pg.draw.lines(self.ui.WIN, self.uav_colors[i % len(self.uav_colors)],
                                      False, points, width=1)

            self.draw_uavs()

            # Draw dynamic obstacles
            if self.dynamic_obstacles.obstacles:
                self.dynamic_obstacles.draw(self.ui.WIN)

            # Draw charging station
            pg.draw.rect(self.ui.WIN, (255, 255, 0),
                         (self.battery_pos[1] * EPSILON + 1, self.battery_pos[0] * EPSILON + 1,
                          EPSILON - 2, EPSILON - 2))

            self.draw_info_panel()
            pg.display.flip()
            clock.tick(FPS)

        pg.quit()

    def print_final_results(self):
        """Print final results"""
        Cr, Rr, AD = self.calculate_performance_metrics()
        total_time = time.time() - self.start_time

        print("=" * 60)
        print("🏆 MCTA SINGLE-GRID COVERAGE RESULTS")
        print("=" * 60)
        print(f"Coverage Rate (Cr): {Cr:.2f}%")
        print(f"Repeated Coverage Rate (Rr): {Rr:.2f}%")
        print(f"Average Flight Deviation (AD): {AD:.2f}")
        print(f"Total Steps: {self.step_count}")
        print(f"Execution Time: {total_time:.2f}s")

        total_mileage = sum(uav.total_flight_mileage for uav in self.uavs)
        print(f"Total Path Length: {total_mileage:.2f}")

        print("\nUAV Details:")
        for uav in self.uavs:
            print(f"  UAV-{uav.id}: Energy={uav.energy:.1f}/{uav.B}, "
                  f"Mileage={uav.total_flight_mileage:.1f}, "
                  f"Visited={len(uav.trajectory_set)}, Status={uav.mode}")


def main():
    mcta_system = MCTASingleGridAdapter()
    mcta_system.run()


if __name__ == "__main__":
    main()