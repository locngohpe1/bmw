import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Set, Optional
import random

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
        self.directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        self.extended_directions = [(-2, -2), (-2, 2), (2, -2), (2, 2), (-4, 0), (0, 4), (4, 0), (0, -4), (-4, -4), (-4, 4), (4, -4), (4, 4)]
        self.prev_unique = 0
        self.no_progress_steps = 0
        self.W1 = 1.0
        self.W2 = 2.0
        self.W3 = 0.5

    def calculate_passable_area(self) -> int:
        return np.sum(self.threat_map < 2)

    def get_module_threat(self, module_center: Tuple[int, int]) -> int:
        r, c = module_center
        threat = 0
        for dr in [-self.D, 0]:
            for dc in [-self.D, 0]:
                nr, nc = r + dr, c + dc
                if self.is_valid_pos((nr, nc)):
                    threat += self.threat_map[nr, nc]
        return threat

    def define_areas_s1_s2_s3(self, current_module: Tuple[int, int], adjacent_module: Tuple[int, int]) -> Tuple[List, List, List]:
        curr_r, curr_c = current_module
        adj_r, adj_c = adjacent_module
        direction = (adj_r - curr_r, adj_c - curr_c)

        s1_positions = []
        s2_positions = []
        s3_positions = []

        if direction == (-2, 0):  # UP
            s1_positions = [(curr_r - 1, curr_c - 1), (curr_r - 1, curr_c)]
            s2_positions = [(adj_r + 1, adj_c - 1), (adj_r + 1, adj_c)]
            s3_positions = [(adj_r, adj_c - 1), (adj_r, adj_c)]
        elif direction == (0, 2):  # RIGHT
            s1_positions = [(curr_r - 1, curr_c + 1), (curr_r, curr_c + 1)]
            s2_positions = [(adj_r - 1, adj_c), (adj_r, adj_c)]
            s3_positions = [(adj_r - 1, adj_c + 1), (adj_r, adj_c + 1)]
        elif direction == (2, 0):  # DOWN
            s1_positions = [(curr_r + 1, curr_c - 1), (curr_r + 1, curr_c)]
            s2_positions = [(adj_r - 1, adj_c - 1), (adj_r - 1, adj_c)]
            s3_positions = [(adj_r, adj_c - 1), (adj_r, adj_c)]
        elif direction == (0, -2):  # LEFT
            s1_positions = [(curr_r - 1, curr_c - 1), (curr_r, curr_c - 1)]
            s2_positions = [(adj_r - 1, adj_c), (adj_r, adj_c)]
            s3_positions = [(adj_r - 1, adj_c - 1), (adj_r, adj_c - 1)]

        return s1_positions, s2_positions, s3_positions

    def calculate_threat_level_zeta(self, current_module: Tuple[int, int], adjacent_module: Tuple[int, int]) -> float:
        if adjacent_module is None:
            return float('inf')

        zeta = 0.0
        s1_positions, s2_positions, s3_positions = self.define_areas_s1_s2_s3(current_module, adjacent_module)

        for pos in s1_positions:
            if self.is_valid_pos(pos):
                eta = self.threat_map[pos[0], pos[1]]  # Use threat_map directly, as t_u
                zeta += eta * self.W1

        for pos in s2_positions:
            if self.is_valid_pos(pos):
                eta = self.threat_map[pos[0], pos[1]]
                zeta += eta * self.W2

        for pos in s3_positions:
            if self.is_valid_pos(pos):
                eta = self.threat_map[pos[0], pos[1]]
                zeta += eta * self.W3

        return zeta

    def get_four_adjacent_modules(self, current_module: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = current_module
        modules = []
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if self.is_valid_module_center((nr, nc)):
                modules.append((nr, nc))
        return modules

    def calculate_bid(self, current_module: Tuple[int, int], adj_module: Tuple[int, int], current_orientation: int) -> float:
        T_mj = self.calculate_threat_level_zeta(current_module, adj_module)
        dir_idx = self.directions.index((adj_module[0] - current_module[0], adj_module[1] - current_module[1]))
        turn_angle = abs(dir_idx - current_orientation) * 90 % 360
        if turn_angle > 180: turn_angle = 360 - turn_angle
        w_turn_val = self.w_turn.get(turn_angle, 0.0)
        next_modules = self.get_four_adjacent_modules(adj_module)
        next_threats = [self.calculate_threat_level_zeta(adj_module, m) for m in next_modules if m != current_module]
        max_next = max(next_threats) if next_threats else 0
        bid = -T_mj - w_turn_val * max_next
        if adj_module not in self.global_visited:
            bid += 200  # Increased bonus for unvisited
        else:
            bid -= 2000  # Stronger penalty for visited
        return bid

    def two_step_auction(self, uav: UAV) -> List[Tuple[float, Tuple[int, int], int]]:
        current_module = self.get_module_center(uav.current_pos)
        current_orientation = uav.current_orientation
        adj_modules = self.get_four_adjacent_modules(current_module)
        bids = []
        for adj in adj_modules:
            b = self.calculate_bid(current_module, adj, current_orientation)
            dir_idx = self.directions.index((adj[0] - current_module[0], adj[1] - current_module[1]))
            bids.append((b, adj, dir_idx))
        bids.sort(reverse=True)
        return bids

    def is_reachable(self, from_pos: Tuple[int, int], to_module: Tuple[int, int]) -> bool:
        return self.calculate_threat_level_zeta(from_pos, to_module) < 10  # Relaxed to avoid early no path

    def obstacle_avoidance_heuristic(self, uav: UAV, target_module: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = target_module
        path = []
        possible = [(r+dr, c+dc) for dr in range(-5,6) for dc in range(-5,6) if (dr,dc) != (0,0)]  # Increased range
        random.shuffle(possible)
        for p in possible:
            if self.is_valid_pos(p) and self.threat_map[p[0], p[1]] < 2 and p not in uav.trajectory[-64:]:  # Relaxed to -64, removed trajectory_set for early steps
                path.append(p)
            if len(path) == 16: break
        if len(path) < 4:
            print(f"Debug: Fallback zigzag for UAV {uav.id} at module {target_module}, len(path) = {len(path)}")
            zigzag = [(r-1, c), (r, c-1), (r+1, c), (r, c+1), (r-2, c-1), (r-1, c-2), (r+1, c+2), (r+2, c+1), (r-3, c), (r, c-3), (r+3, c), (r, c+3), (r-2, c+2), (r+2, c-2)]
            path = [p for p in zigzag if self.is_valid_pos(p) and self.threat_map[p[0], p[1]] < 2 and p not in uav.trajectory[-64:]]
            if len(path) < 4:
                path = [ (r, c) for _ in range(4) ]  # Minimal dummy path if all fails (avoid None)
        print(f"Debug: Heuristic for UAV {uav.id} at step {self.step_count}: possible len {len(possible)}, final path len {len(path)}, unique added {len(set(path) - uav.trajectory_set)}")
        return path

    def detect_conflicts(self, winning_modules: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        conflicts = {}
        for uav_id, module_pos in winning_modules.items():
            if module_pos not in conflicts:
                conflicts[module_pos] = []
            conflicts[module_pos].append(uav_id)
        return {pos: uav_list for pos, uav_list in conflicts.items() if len(uav_list) > 1}

    def resolve_conflicts(self, conflicts: Dict[Tuple[int, int], List[int]]) -> Dict[int, bool]:
        winners = {}
        L_bar = sum(u.total_flight_mileage for u in self.uavs) / self.v if self.v > 0 else 0
        for module, uav_list in conflicts.items():
            preferred = [uid for uid in uav_list if self.uavs[uid-1].total_flight_mileage < L_bar - 10]
            min_mileage_uav = min(preferred or uav_list, key=lambda uid: self.uavs[uid-1].total_flight_mileage)
            for uid in uav_list:
                winners[uid] = (uid == min_mileage_uav)
        return winners

    def update_dynamic_obstacles(self):
        for r in range(self.m):
            for c in range(self.n):
                if self.threat_map[r, c] > 0:
                    move_prob = 0.2 / (self.threat_map[r, c] + 1)
                    if random.random() < move_prob:
                        dr, dc = random.choice(self.directions + self.extended_directions + [(0,0)])
                        nr, nc = r + dr // 2, c + dc // 2
                        if self.is_valid_pos((nr, nc)):
                            self.threat_map[nr, nc] = self.threat_map[r, c]
                            self.threat_map[r, c] = 0

    def balance_uavs(self):
        L_list = [u.total_flight_mileage for u in self.uavs]
        if max(L_list) - min(L_list) > 5 * self.v:
            max_uav = self.uavs[np.argmax(L_list)]
            min_uav = self.uavs[np.argmin(L_list)]
            max_uav.current_pos, min_uav.current_pos = min_uav.current_pos, max_uav.current_pos
            max_uav.current_orientation, min_uav.current_orientation = min_uav.current_orientation, max_uav.current_orientation

    def execute_mcta_step(self) -> bool:
        self.update_dynamic_obstacles()
        self.balance_uavs()
        winning_modules = {}
        re_try_count = {u.id: 0 for u in self.uavs}
        for uav in self.uavs:
            if uav.mode == "SLEEP": continue
            auction_results = self.two_step_auction(uav)
            found = False
            re_try = 0
            for b, module, dir_idx in auction_results:
                re_try += 1
                if self.is_reachable(uav.current_pos, module):
                    path = self.obstacle_avoidance_heuristic(uav, module)
                    if path:
                        winning_modules[uav.id] = module
                        uav.current_orientation = dir_idx
                        self.global_visited.add(module)
                        found = True
                        break
            re_try_count[uav.id] = re_try
            if found:
                print(f"Debug: UAV {uav.id} found path after {re_try} re-tries at step {self.step_count}")
            else:
                module = self.get_random_uncovered_module()
                path = self.obstacle_avoidance_heuristic(uav, module)
                if path:
                    winning_modules[uav.id] = module
                    dir_idx = random.randint(0,3)
                    uav.current_orientation = dir_idx
                    self.global_visited.add(module)
                    found = True
                    print(f"Debug: UAV {uav.id} redirected to uncovered module at step {self.step_count}")
            if not found:
                uav.mode = "SLEEP"
                print(f"Debug: UAV {uav.id} sleep due to no path after {re_try} re-tries at step {self.step_count}")
        conflicts = self.detect_conflicts(winning_modules)
        resolutions = self.resolve_conflicts(conflicts)
        for uav in self.uavs:
            if uav.mode == "SLEEP": continue
            if uav.id not in winning_modules:
                uav.mode = "SLEEP"
                continue
            module = winning_modules[uav.id]
            conflict_win = resolutions.get(uav.id, True)
            if conflict_win:
                should_sleep, reason = uav.should_sleep()
                if not should_sleep:
                    path = self.obstacle_avoidance_heuristic(uav, module)  # Re-call if needed
                    if path:
                        dist = len(path)
                        uav.update_flight_mileage(dist)
                        for pos in path:
                            uav.add_to_trajectory(pos)
                            self.global_covered.add(pos)
                        uav.current_pos = path[-1] if path else module
                    else:
                        uav.mode = "SLEEP"
                        print(f"Debug: UAV {uav.id} sleep due to no path in conflict win at step {self.step_count}")
                else:
                    uav.mode = "SLEEP"
                    print(f"Debug: UAV {uav.id} sleep due to {reason} at step {self.step_count}")
            else:
                uav.is_waiting = True
        self.step_count += 1
        Cr, Rr, Df = self.calculate_performance_metrics()
        unique_now = len(self.global_covered)
        progress = unique_now - self.prev_unique > 0
        self.prev_unique = unique_now
        if self.step_count % 50 == 0:
            sleep_count = sum(1 for u in self.uavs if u.mode == "SLEEP")
            loop_count = sum(1 for u in self.uavs if u.loop_detected)
            exhaust_count = sum(1 for u in self.uavs if u.energy <= 0)
            no_path_count = sleep_count - loop_count - exhaust_count
            L_i = [u.total_flight_mileage for u in self.uavs]
            print(f"Debug Step {self.step_count}: C_r {Cr:.2f}%, R_r {Rr:.2f}%, D_f {Df:.2f}, Unique {unique_now}, Sleep {sleep_count} (Loop {loop_count}, Exhaust {exhaust_count}, NoPath {no_path_count}), L_i {L_i}, Re-try counts {re_try_count}")
        if not progress:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        return (any(uav.mode == "WORK" for uav in self.uavs) or self.step_count < 400 or progress) and Cr < 95.0 and self.no_progress_steps < 50

    def calculate_performance_metrics(self) -> Tuple[float, float, float]:
        union_covered = set()
        total_mileage = 0.0
        for uav in self.uavs:
            union_covered.update(uav.trajectory_set)
            total_mileage += uav.total_flight_mileage
        Cr = (len(union_covered) / self.total_area) * 100.0 if self.total_area > 0 else 0.0
        Rr = (total_mileage - len(union_covered)) / total_mileage * 100.0 if total_mileage > 0 else 0.0
        if self.v > 0:
            L_bar = total_mileage / self.v
            Df = sum(abs(uav.total_flight_mileage - L_bar) for uav in self.uavs) / self.v
        else:
            Df = 0.0
        return Cr, Rr, Df

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.m and 0 <= pos[1] < self.n

    def is_valid_module_center(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 < r < self.m - 1 and 0 < c < self.n - 1

    def get_module_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        r, c = pos
        module_r = ((r // (2 * self.D)) * 2 * self.D) + self.D
        module_c = ((c // (2 * self.D)) * 2 * self.D) + self.D
        return (max(1, min(module_r, self.m - 2)), max(1, min(module_c, self.n - 2)))

    def set_initial_threats(self, threat_map: np.ndarray):
        self.threat_map = threat_map.copy()
        self.passable_area = self.calculate_passable_area()

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

# Test with seed for reproducibility
if __name__ == "__main__":
    random.seed(42)
    mcta = MCTA(map_rows=20, map_cols=20, num_uavs=4, energy_capacity=200000)
    threat_map = np.zeros((20, 20), dtype=int)
    threat_map[5:6, 5:6] = 1
    threat_map[15:16, 15:16] = 2
    mcta.set_initial_threats(threat_map)
    results = mcta.run_coverage_simulation(max_steps=2000)
    print(f"Coverage Complete: {results['coverage_complete']}")
    print(f"Total Steps: {results['final_metrics']['Total_Steps']}")
    print(f"Coverage Rate: {results['final_metrics']['Coverage_Rate']:.2f}%")
    print(f"Repeated Rate: {results['final_metrics']['Repeated_Coverage_Rate']:.2f}%")
    print(f"Flight Deviation: {results['final_metrics']['Average_Flight_Deviation']:.2f}")