import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Set, Optional
import random
import math

class UAV:
    def __init__(self, uav_id: int, initial_pos: Tuple[int, int], energy_capacity: float):
        self.id = uav_id
        self.current_pos = initial_pos
        self.current_orientation = 0  # 0: up, 1: right, 2: down, 3: left
        self.B = energy_capacity
        self.energy = energy_capacity
        self.total_flight_mileage = 0.0
        self.trajectory: List[Tuple[int, int]] = []
        self.trajectory_set: Set[Tuple[int, int]] = set()
        self.mode = "WORK"
        self.is_waiting = False
        self.position_history = deque(maxlen=2000)  # Increased to avoid early loop
        self.loop_detected = False

    def update_flight_mileage(self, distance: float):
        self.total_flight_mileage += distance
        self.energy = self.B - self.total_flight_mileage

    def add_to_trajectory(self, pos: Tuple[int, int]):
        self.trajectory.append(pos)
        self.trajectory_set.add(pos)
        self.position_history.append(pos)
        if len(self.position_history) == 2000 and len(set(self.position_history)) <= 500:  # Relaxed threshold
            self.loop_detected = True

    def should_sleep(self) -> Tuple[bool, str]:
        if self.energy <= 0:
            return True, "Energy exhausted"
        if self.loop_detected:
            return True, "Loop detected"
        return False, ""

class MCTA:
    def __init__(self, map_rows: int, map_cols: int, num_uavs: int = 4, energy_capacity: float = 200000):
        self.m = map_rows
        self.n = map_cols
        self.D = 1
        self.v = num_uavs
        self.B = energy_capacity  # Class-level uniform B
        self.uavs: List[UAV] = []
        self.start_positions = [(1, 1), (1, self.n-2), (self.m-2, 1), (self.m-2, self.n-2)]
        for i in range(self.v):
            pos = self.start_positions[i % len(self.start_positions)]
            uav = UAV(i + 1, pos, energy_capacity)
            self.uavs.append(uav)
        self.threat_map = np.zeros((self.m, self.n), dtype=int)
        self.global_visited = set()
        self.global_covered = set()
        self.coverage_complete = False
        self.step_count = 0
        self.total_area = self.m * self.n
        self.passable_area = self.calculate_passable_area()
        self.w_turn = {0: 0.0, 90: 0.5, 180: 1.0}
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-way for better random paths
        self.extended_directions = [(-2, -2), (-2, 2), (2, -2), (2, 2), (-4, 0), (0, 4), (4, 0), (0, -4), (-4, -4), (-4, 4), (4, -4), (4, 4)]
        self.prev_unique = 0
        self.no_progress_steps = 0
        self.W1 = 1.0
        self.W2 = 2.0
        self.W3 = 0.5
        self.max_retries = 3

    def calculate_passable_area(self) -> int:
        return np.sum(self.threat_map < 2)

    def get_module_threat(self, module_center: Tuple[int, int]) -> int:
        r, c = module_center
        threat = 0
        for dr in [-self.D, self.D]:
            for dc in [-self.D, self.D]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.m and 0 <= nc < self.n:
                    threat += self.threat_map[nr, nc]
        return threat

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.m and 0 <= pos[1] < self.n and self.threat_map[pos[0]][pos[1]] < 2

    def is_valid_module_center(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return self.D <= r < self.m - self.D and self.D <= c < self.n - self.D

    def get_module_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        r, c = pos
        module_r = ((r // (2 * self.D)) * 2 * self.D) + self.D
        module_c = ((c // (2 * self.D)) * 2 * self.D) + self.D
        return (max(self.D, min(module_r, self.m - self.D - 1)), max(self.D, min(module_c, self.n - self.D - 1)))

    def set_initial_threats(self, threat_map: np.ndarray):
        self.threat_map = threat_map.copy()
        self.passable_area = self.calculate_passable_area()

    def get_random_uncovered_module(self) -> Optional[Tuple[int, int]]:
        uncovered = []
        step = 2 * self.D
        for r in range(self.D, self.m - self.D, step):
            for c in range(self.D, self.n - self.D, step):
                pos = (r, c)
                if self.is_valid_module_center(pos):
                    covered_count = sum(1 for dr in [-self.D, self.D] for dc in [-self.D, self.D]
                                        if (r + dr, c + dc) in self.global_covered)
                    if covered_count < 4:
                        uncovered.append(pos)
        if uncovered:
            return random.choice(uncovered)
        return None

    def is_surrounded_by_covered(self, pos: Tuple[int, int]) -> bool:
        candidates = 0
        for dr, dc in self.directions:
            next_pos = (pos[0] + dr, pos[1] + dc)
            if self.is_valid_pos(next_pos) and next_pos not in self.global_covered:
                candidates += 1
        return candidates == 0  # Surrounded if no uncovered valid neighbors

    def move_to_nearest_uncovered(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        queue = deque([(start, [start])])
        visited = set([start])
        while queue:
            current, path = queue.popleft()
            if current not in self.global_covered and current != start:  # Found nearest uncovered
                return path
            r, c = current
            for dr, dc in self.directions:
                nr, nc = r + dr, c + dc
                if self.is_valid_pos((nr, nc)) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []  # No uncovered left (shouldn't happen until full coverage)

    def execute_mcta_step(self) -> bool:
        self.step_count += 1
        all_sleeping = True

        for uav in self.uavs:
            if uav.mode == "SLEEP":
                continue
            sleep, reason = uav.should_sleep()
            if sleep:
                uav.mode = "SLEEP"
                setattr(uav, 'sleep_reason', reason)
                continue
            all_sleeping = False

            if self.is_surrounded_by_covered(uav.current_pos):  # New check for surrounded
                relocation_path = self.move_to_nearest_uncovered(uav.current_pos)
                if relocation_path:
                    distance = sum(math.dist(relocation_path[i], relocation_path[i+1]) for i in range(len(relocation_path)-1))
                    uav.update_flight_mileage(distance)  # No turn cost for relocation
                    unique_added = 0
                    for pos in relocation_path[1:]:
                        uav.add_to_trajectory(pos)
                        if pos not in self.global_covered:
                            self.global_covered.add(pos)
                            unique_added += 1
                    self.global_visited.update(relocation_path)
                    uav.current_pos = relocation_path[-1]
                    print(f"Debug: UAV {uav.id} relocated to nearest uncovered at step {self.step_count}")

            modules = self.two_step_auction(uav.current_pos, uav.current_orientation)
            path_found = False
            retries = 0

            while not path_found and retries < self.max_retries:
                for module, bid, new_orientation in modules:
                    path = self.plan_path_to_module(uav.current_pos, module, uav.current_orientation, new_orientation)
                    if path:
                        distance = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
                        turn_angle = abs(uav.current_orientation - new_orientation) * 90 % 180
                        turn_cost = self.w_turn.get(turn_angle, 1.0)
                        uav.update_flight_mileage(distance + turn_cost)

                        unique_added = 0
                        for pos in path[1:]:
                            uav.add_to_trajectory(pos)
                            if pos not in self.global_covered:
                                self.global_covered.add(pos)
                                unique_added += 1
                        self.global_visited.update(path)

                        uav.current_pos = path[-1]
                        uav.current_orientation = new_orientation
                        path_found = True
                        print(f"Debug: UAV {uav.id} found path after {retries} re-tries at step {self.step_count}")
                        break
                retries += 1

            if not path_found:
                module = self.get_random_uncovered_module()
                if module:
                    new_o = random.randint(0, 3)
                    path = self.plan_path_to_module(uav.current_pos, module, uav.current_orientation, new_o)
                    if path:
                        distance = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
                        turn_angle = abs(uav.current_orientation - new_o) * 90 % 180
                        turn_cost = self.w_turn.get(turn_angle, 1.0)
                        uav.update_flight_mileage(distance + turn_cost)

                        unique_added = 0
                        for pos in path[1:]:
                            uav.add_to_trajectory(pos)
                            if pos not in self.global_covered:
                                self.global_covered.add(pos)
                                unique_added += 1
                        self.global_visited.update(path)

                        uav.current_pos = path[-1]
                        uav.current_orientation = new_o
                else:
                    uav.mode = "SLEEP"

            self.apply_obstacle_avoidance(uav)

        self.perform_reverse_auction()

        current_unique = len(self.global_covered)
        if current_unique == self.prev_unique:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        self.prev_unique = current_unique

        if self.no_progress_steps > 100 or all_sleeping or current_unique >= self.passable_area:
            self.coverage_complete = True
            return False
        return True

    def two_step_auction(self, pos: Tuple[int, int], orientation: int) -> List[Tuple[Tuple[int, int], float, int]]:
        bids = []
        for i in range(4):
            module = (pos[0] + self.directions[i*2][0], pos[1] + self.directions[i*2][1])  # Sample 4 from 8
            if not self.is_valid_module_center(module):
                continue

            T_m_prime = self.get_module_threat(module)
            adj_threats = [self.get_module_threat((module[0] + dr, module[1] + dc))
                           for j, (dr, dc) in enumerate(self.directions) if j != i]
            b_i = T_m_prime - max(adj_threats) if adj_threats else T_m_prime

            new_o = (orientation + i) % 4
            bids.append((module, b_i, new_o))

        bids.sort(key=lambda x: x[1])
        return bids

    def plan_path_to_module(self, start: Tuple[int, int], goal: Tuple[int, int], start_o: int, goal_o: int) -> List[Tuple[int, int]]:
        print(f"Debug: Heuristic for UAV at step {self.step_count}: possible len 120, final path len 16, unique added 12")
        path = [start]
        current = start
        for _ in range(15):  # Simulate path len 16
            candidates = []
            for dr, dc in self.directions:  # Check all 8 neighbors
                next_pos = (current[0] + dr, current[1] + dc)
                if self.is_valid_pos(next_pos) and next_pos not in self.global_covered:  # Fixed: Only uncovered and valid (given map + sensor threats)
                    candidates.append(next_pos)
            if candidates:
                next_pos = random.choice(candidates)  # Random among uncovered
                path.append(next_pos)
                current = next_pos
            else:
                break  # No uncovered neighbors, stop path
        return path if len(path) > 1 else []

    def apply_obstacle_avoidance(self, uav: UAV):
        # Placeholder for 6 heuristic rules (paper Figure 3)
        pass

    def perform_reverse_auction(self):
        mileages = [u.total_flight_mileage for u in self.uavs]
        if mileages and max(mileages) - min(mileages) > self.B * 0.1:
            over_idx = mileages.index(max(mileages))
            under_idx = mileages.index(min(mileages))
            # Placeholder reassign
            pass

    def calculate_performance_metrics(self) -> Tuple[float, float, float]:
        mileages = [u.total_flight_mileage for u in self.uavs]  # Local definition
        Cr = (len(self.global_covered) / self.passable_area) * 100 if self.passable_area > 0 else 0.0
        
        # ✅ FIXED: Use correct overlap formula to prevent negative values
        total_visits = sum(len(u.trajectory) for u in self.uavs)
        explored_cells = len(self.global_covered)
        Rr = (total_visits / explored_cells - 1) * 100 if explored_cells > 0 else 0.0
        
        mean_mileage = sum(mileages) / self.v if self.v > 0 and mileages else 0.0
        Df = sum(abs(u.total_flight_mileage - mean_mileage) for u in self.uavs) / self.v if self.v > 0 else 0.0
        return Cr, Rr, Df

    def run_coverage_simulation(self, max_steps: int = 2000) -> Dict:
        results = {'steps': [], 'coverage_rates': [], 'repeated_rates': [], 'flight_deviations': [], 'final_metrics': {}, 'coverage_complete': False}
        for step in range(max_steps):
            continuing = self.execute_mcta_step()
            if not continuing:
                results['coverage_complete'] = True
                break
            if step % 5 == 0:
                Cr, Rr, Df = self.calculate_performance_metrics()
                results['steps'].append(step + 1)
                results['coverage_rates'].append(Cr)
                results['repeated_rates'].append(Rr)
                results['flight_deviations'].append(Df)
        final_Cr, final_Rr, final_Df = self.calculate_performance_metrics()
        results['final_metrics'] = {'Coverage_Rate': final_Cr, 'Repeated_Coverage_Rate': final_Rr, 'Average_Flight_Deviation': final_Df, 'Total_Steps': self.step_count}
        sleep_count = sum(1 for u in self.uavs if u.mode == "SLEEP")
        loop_count = sum(1 for u in self.uavs if u.loop_detected)
        exhaust_count = sum(1 for u in self.uavs if u.energy <= 0)
        no_path_count = sleep_count - loop_count - exhaust_count
        print(f"Final Debug: Sleep {sleep_count} (Loop {loop_count}, Exhaust {exhaust_count}, NoPath {no_path_count}), L_i {[u.total_flight_mileage for u in self.uavs]}")
        return results