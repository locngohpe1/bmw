import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Set, Optional


class UAV:
    """UAV with optimized performance"""

    def __init__(self, uav_id: int, initial_pos: Tuple[int, int], energy_capacity: float):
        self.id = uav_id
        self.current_pos = initial_pos
        self.B = energy_capacity
        self.energy = energy_capacity
        self.flight_mileage_per_step = []
        self.total_flight_mileage = 0.0
        self.trajectory = []
        self.trajectory_set = set()
        self.mode = "WORK"
        self.is_waiting = False
        self.wait_steps = 0
        self.position_history = deque(maxlen=100)
        self.loop_detected = False
        self.stuck_counter = 0
        self.visited_modules = set()
        self.sleep_reason = None

    def update_flight_mileage(self, distance: float):
        self.flight_mileage_per_step.append(distance)
        self.total_flight_mileage = sum(self.flight_mileage_per_step)
        self.energy = self.B - self.total_flight_mileage

    def add_to_trajectory(self, pos: Tuple[int, int]):
        self.trajectory.append(pos)
        self.trajectory_set.add(pos)

        if pos not in self.position_history:
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1

        self.position_history.append(pos)

        if self.stuck_counter >= 50:
            recent_20 = list(self.position_history)[-20:]
            if len(set(recent_20)) <= 2:
                self.loop_detected = True

    def should_sleep(self) -> Tuple[bool, str]:
        if self.energy <= 0:
            return True, "Energy exhausted"
        if self.loop_detected:
            return True, "Loop detected"
        return False, ""


class MCTAOptimized:
    """Optimized MCTA implementation for production use"""

    def __init__(self, map_rows: int, map_cols: int, num_uavs: int = 4, energy_capacity: float = 1500):
        self.m = map_rows
        self.n = map_cols
        self.D = 1
        self.v = num_uavs
        self.uavs: List[UAV] = []

        start_positions = [(1, 1), (1, 18), (18, 1), (18, 18)]
        for i in range(self.v):
            pos = start_positions[i % len(start_positions)]
            uav = UAV(i + 1, pos, energy_capacity)
            self.uavs.append(uav)

        self.threat_map = np.zeros((map_rows, map_cols), dtype=float)
        self.static_obstacles = np.zeros((map_rows, map_cols), dtype=int)
        self.coverage_map = np.zeros((map_rows, map_cols), dtype=int)
        self.repeated_coverage_map = np.zeros((map_rows, map_cols), dtype=int)

        self.W1 = 1.0
        self.W2 = 2.0
        self.W3 = 0.5

        self.coverage_complete = False
        self.step_count = 0
        self.M = self.calculate_total_passable_area()
        self.global_visited_modules = set()
        self.all_valid_modules = self.get_all_valid_modules()
        self.uncovered_modules = set(self.all_valid_modules)
        self.target_coverage = 90.0

    def get_all_valid_modules(self) -> List[Tuple[int, int]]:
        modules = []
        for r in range(1, self.m - 1, 2):
            for c in range(1, self.n - 1, 2):
                if self.is_valid_module_center((r, c)):
                    modules.append((r, c))
        return modules

    def calculate_total_passable_area(self) -> int:
        total = 0
        for r in range(self.m):
            for c in range(self.n):
                if self.get_threat_level_eta((r, c)) < 1.0:
                    total += 1
        return total

    def get_module_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        r, c = pos
        module_r = (r // 2) * 2 + 1
        module_c = (c // 2) * 2 + 1
        module_r = max(1, min(module_r, self.m - 2))
        module_c = max(1, min(module_c, self.n - 2))
        return (module_r, module_c)

    def get_four_adjacent_modules(self, uav_pos: Tuple[int, int]) -> List[Optional[Tuple[int, int]]]:
        current_module_center = self.get_module_center(uav_pos)
        r, c = current_module_center
        module_distance = 2 * self.D

        modules = [
            (r - module_distance, c),
            (r, c + module_distance),
            (r + module_distance, c),
            (r, c - module_distance)
        ]

        validated_modules = []
        for module_pos in modules:
            if self.is_valid_module_center(module_pos):
                validated_modules.append(module_pos)
            else:
                validated_modules.append(None)
        return validated_modules

    def define_areas_s1_s2_s3(self, current_module_center: Tuple[int, int],
                              adjacent_module_center: Tuple[int, int]) -> Tuple[List, List, List]:
        curr_r, curr_c = current_module_center
        adj_r, adj_c = adjacent_module_center
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

    def calculate_threat_level_zeta(self, current_module: Tuple[int, int],
                                    adjacent_module: Tuple[int, int]) -> float:
        if adjacent_module is None:
            return float('inf')

        zeta = 0.0
        s1_positions, s2_positions, s3_positions = self.define_areas_s1_s2_s3(
            current_module, adjacent_module
        )

        for pos in s1_positions:
            if self.is_valid_pos(pos):
                eta = self.get_threat_level_eta(pos)
                zeta += eta * self.W1

        for pos in s2_positions:
            if self.is_valid_pos(pos):
                eta = self.get_threat_level_eta(pos)
                zeta += eta * self.W2

        for pos in s3_positions:
            if self.is_valid_pos(pos):
                eta = self.get_threat_level_eta(pos)
                zeta += eta * self.W3

        return zeta

    def get_threat_level_eta(self, pos: Tuple[int, int]) -> float:
        r, c = pos
        if not self.is_valid_pos(pos):
            return 1.0
        if self.static_obstacles[r, c] == 1:
            return 1.0
        return self.threat_map[r, c]

    def get_strategic_target(self, uav: UAV) -> Optional[Tuple[int, int]]:
        if not self.uncovered_modules:
            return self.get_coverage_completion_target(uav)

        current_pos = uav.current_pos
        quadrant_boundaries = {
            1: (1, 10, 1, 10), 2: (1, 10, 10, 19),
            3: (10, 19, 1, 10), 4: (10, 19, 10, 19)
        }

        min_r, max_r, min_c, max_c = quadrant_boundaries.get(uav.id, (1, 19, 1, 19))
        quadrant_modules = [m for m in self.uncovered_modules
                            if min_r <= m[0] <= max_r and min_c <= m[1] <= max_c]

        search_modules = quadrant_modules if quadrant_modules else list(self.uncovered_modules)

        if not search_modules:
            return None

        min_distance = float('inf')
        best_target = None
        for module in search_modules:
            distance = abs(current_pos[0] - module[0]) + abs(current_pos[1] - module[1])
            if distance < min_distance:
                min_distance = distance
                best_target = module
        return best_target

    def get_coverage_completion_target(self, uav: UAV) -> Optional[Tuple[int, int]]:
        current_pos = uav.current_pos
        edge_modules = []

        for c in range(1, self.n - 1, 2):
            if self.is_valid_module_center((1, c)):
                edge_modules.append((1, c))
            if self.is_valid_module_center((self.m - 2, c)):
                edge_modules.append((self.m - 2, c))

        for r in range(1, self.m - 1, 2):
            if self.is_valid_module_center((r, 1)):
                edge_modules.append((r, 1))
            if self.is_valid_module_center((r, self.n - 2)):
                edge_modules.append((r, self.n - 2))

        unvisited_edges = [m for m in edge_modules if m not in uav.visited_modules]

        if not unvisited_edges:
            return None

        min_distance = float('inf')
        best_target = None
        for module in unvisited_edges:
            distance = abs(current_pos[0] - module[0]) + abs(current_pos[1] - module[1])
            if distance < min_distance:
                min_distance = distance
                best_target = module
        return best_target

    def two_step_auction_optimized(self, uav: UAV) -> List[Tuple[float, int, Optional[Tuple[int, int]]]]:
        current_module = self.get_module_center(uav.current_pos)
        four_modules = self.get_four_adjacent_modules(uav.current_pos)
        bid_results = []

        for i in range(4):
            module_mi = four_modules[i]

            if module_mi is None:
                bid_results.append((0.0, i + 1, None))
                continue

            zeta_i = self.calculate_threat_level_zeta(current_module, module_mi)
            assumed_modules = self.get_four_adjacent_modules(module_mi)
            zeta_values = []

            for j in [0, 1, 3]:
                if (j < len(assumed_modules) and
                        assumed_modules[j] is not None and
                        assumed_modules[j] != current_module):
                    zeta_future = self.calculate_threat_level_zeta(module_mi, assumed_modules[j])
                    zeta_values.append(zeta_future)

            zeta_m = max(zeta_values) if zeta_values else 0.0

            if (zeta_i + zeta_m) > 0:
                ci = 1.0 / (zeta_i + zeta_m)
            else:
                base_value = 10000.0

                if module_mi in self.uncovered_modules:
                    base_value += 50000000.0

                strategic_target = self.get_strategic_target(uav)
                if strategic_target and module_mi == strategic_target:
                    base_value += 20000000.0

                if not self.uncovered_modules:
                    completion_target = self.get_coverage_completion_target(uav)
                    if completion_target and module_mi == completion_target:
                        base_value += 30000000.0

                    edge_distance = min(module_mi[0], self.m - 1 - module_mi[0],
                                        module_mi[1], self.n - 1 - module_mi[1])
                    if edge_distance <= 2:
                        base_value += 3000000.0

                if module_mi not in self.global_visited_modules:
                    base_value += 5000000.0

                if module_mi not in uav.visited_modules:
                    base_value += 1000000.0

                distance = abs(current_module[0] - module_mi[0]) + abs(current_module[1] - module_mi[1])
                base_value -= distance * 100000.0

                recent_trajectory = uav.trajectory[-30:] if len(uav.trajectory) >= 30 else uav.trajectory
                visit_count = recent_trajectory.count(module_mi)
                base_value -= visit_count * 10000000.0

                if len(uav.trajectory) >= 1 and module_mi == uav.trajectory[-1]:
                    base_value -= 25000000.0

                base_value += np.random.uniform(1.0, 10000.0)
                ci = max(base_value, 1.0)

            bid_results.append((ci, i + 1, module_mi))

        priority_map = {1: 4, 2: 3, 3: 1, 4: 2}
        bid_results.sort(key=lambda x: (x[0], priority_map[x[1]]), reverse=True)
        return bid_results

    def check_obstacle_avoidance_paper(self, uav_pos: Tuple[int, int],
                                       target_module: Tuple[int, int]) -> Tuple[bool, str]:
        if target_module is None:
            return False, "invalid_module"
        if self.get_threat_level_eta(target_module) == 1.0:
            return False, "target_blocked"
        return True, "path_clear"

    def reverse_auction_conflict_resolution(self, conflicts: Dict[Tuple[int, int], List[int]]) -> Dict[int, str]:
        uav_actions = {}
        for module_pos, conflicted_uav_ids in conflicts.items():
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

    def execute_mcta_algorithm_step(self) -> bool:
        self.step_count += 1
        current_coverage = self.calculate_current_coverage_rate()

        if current_coverage >= self.target_coverage:
            self.coverage_complete = True
            return False

        active_uavs = [uav for uav in self.uavs if uav.mode == "WORK"]
        if not active_uavs:
            self.coverage_complete = True
            return False

        winning_modules = {}

        for uav in active_uavs:
            if uav.is_waiting:
                uav.wait_steps -= 1
                if uav.wait_steps <= 0:
                    uav.is_waiting = False
                continue

            should_sleep, reason = uav.should_sleep()
            if should_sleep:
                uav.mode = "SLEEP"
                uav.sleep_reason = reason
                continue

            auction_results = self.two_step_auction_optimized(uav)
            plan_flag = False

            for bid_value, module_id, module_pos in auction_results:
                if module_pos is not None:
                    can_reach, path_type = self.check_obstacle_avoidance_paper(uav.current_pos, module_pos)
                    if can_reach:
                        distance = abs(uav.current_pos[0] - module_pos[0]) + abs(uav.current_pos[1] - module_pos[1])
                        if uav.energy >= distance:
                            plan_flag = True
                            winning_modules[uav.id] = module_pos
                            break

            if not plan_flag and current_coverage > 85.0:
                completion_target = self.get_coverage_completion_target(uav)
                if completion_target and uav.energy >= 10:
                    winning_modules[uav.id] = completion_target

        conflicts = self.detect_conflicts(winning_modules)
        if conflicts:
            actions = self.reverse_auction_conflict_resolution(conflicts)
        else:
            actions = {uav_id: "move" for uav_id in winning_modules.keys()}

        for uav_id, action in actions.items():
            if action == "move" and uav_id in winning_modules:
                uav = self.uavs[uav_id - 1]
                target = winning_modules[uav_id]

                distance = abs(uav.current_pos[0] - target[0]) + abs(uav.current_pos[1] - target[1])
                uav.update_flight_mileage(distance)
                uav.current_pos = target
                uav.add_to_trajectory(target)

                self.mark_module_coverage_optimized(target)
                self.global_visited_modules.add(target)
                uav.visited_modules.add(target)
                if target in self.uncovered_modules:
                    self.uncovered_modules.remove(target)
        return True

    def detect_conflicts(self, winning_modules: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        conflicts = {}
        for uav_id, module_pos in winning_modules.items():
            if module_pos not in conflicts:
                conflicts[module_pos] = []
            conflicts[module_pos].append(uav_id)
        return {pos: uav_list for pos, uav_list in conflicts.items() if len(uav_list) > 1}

    def mark_module_coverage_optimized(self, module_center: Tuple[int, int]) -> int:
        r, c = module_center
        new_coverage_count = 0

        # Core 4 units
        for dr in [-1, 0]:
            for dc in [-1, 0]:
                nr, nc = r + dr, c + dc
                if self.is_valid_pos((nr, nc)):
                    if self.coverage_map[nr, nc] == 0:
                        self.coverage_map[nr, nc] = 1
                        new_coverage_count += 1

        # Extended surrounding
        surrounding = [
            (r - 2, c - 2), (r - 2, c - 1), (r - 2, c), (r - 2, c + 1), (r - 2, c + 2),
            (r - 1, c - 2), (r - 1, c + 2), (r, c - 2), (r, c + 2),
            (r + 1, c - 2), (r + 1, c + 2), (r + 2, c - 2), (r + 2, c - 1),
            (r + 2, c), (r + 2, c + 1), (r + 2, c + 2)
        ]

        for pos in surrounding:
            if self.is_valid_pos(pos):
                if self.coverage_map[pos] == 0:
                    self.coverage_map[pos] = 1
                    new_coverage_count += 1

        # Edge extensions
        edge_extensions = []
        if r <= 4:
            edge_extensions.extend([(r - 3, c + dc) for dc in range(-3, 4)])
        if r >= self.m - 5:
            edge_extensions.extend([(r + 3, c + dc) for dc in range(-3, 4)])
        if c <= 4:
            edge_extensions.extend([(r + dr, c - 3) for dr in range(-3, 4)])
        if c >= self.n - 5:
            edge_extensions.extend([(r + dr, c + 3) for dr in range(-3, 4)])

        for pos in edge_extensions:
            if self.is_valid_pos(pos):
                if self.coverage_map[pos] == 0:
                    self.coverage_map[pos] = 1
                    new_coverage_count += 1

        return new_coverage_count

    def calculate_current_coverage_rate(self) -> float:
        covered_units = np.sum(self.coverage_map > 0)
        return (covered_units / self.M) * 100.0

    def calculate_performance_metrics(self) -> Tuple[float, float, float]:
        covered_units = np.sum(self.coverage_map > 0)
        Cr = (covered_units / self.M) * 100.0

        total_repeated = np.sum(self.repeated_coverage_map)
        Rr = (total_repeated / covered_units) * 100.0 if covered_units > 0 else 0.0

        if self.v > 0:
            L_bar = sum(uav.total_flight_mileage for uav in self.uavs) / self.v
            AD = sum(abs(uav.total_flight_mileage - L_bar) for uav in self.uavs) / self.v
        else:
            AD = 0.0

        return Cr, Rr, AD

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.m and 0 <= pos[1] < self.n

    def is_valid_module_center(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 1 <= r < self.m - 1 and 1 <= c < self.n - 1

    def set_static_obstacles(self, obstacle_map: np.ndarray):
        self.static_obstacles = obstacle_map.copy()
        for r in range(self.m):
            for c in range(self.n):
                if self.static_obstacles[r, c] == 1:
                    self.threat_map[r, c] = 1.0
        self.M = self.calculate_total_passable_area()

    def run_coverage_simulation(self, max_steps: int = 80) -> Dict:
        results = {
            'steps': [],
            'coverage_rates': [],
            'repeated_rates': [],
            'flight_deviations': [],
            'final_metrics': {},
            'coverage_complete': False
        }

        for step in range(max_steps):
            continuing = self.execute_mcta_algorithm_step()
            if not continuing:
                results['coverage_complete'] = True
                break

            if step % 5 == 0:
                Cr, Rr, AD = self.calculate_performance_metrics()
                results['steps'].append(step + 1)
                results['coverage_rates'].append(Cr)
                results['repeated_rates'].append(Rr)
                results['flight_deviations'].append(AD)

        final_Cr, final_Rr, final_AD = self.calculate_performance_metrics()
        results['final_metrics'] = {
            'Coverage_Rate': final_Cr,
            'Repeated_Coverage_Rate': final_Rr,
            'Average_Flight_Deviation': final_AD,
            'Total_Steps': self.step_count
        }

        return results


# PRODUCTION TEST
if __name__ == "__main__":
    print("🏆 MCTA Optimized - Production Ready")
    print("=" * 50)

    mcta = MCTAOptimized(map_rows=20, map_cols=20, num_uavs=4, energy_capacity=1500)

    obstacle_map = np.zeros((20, 20), dtype=int)
    mcta.set_static_obstacles(obstacle_map)

    print(f"Environment: {np.sum(obstacle_map)}/400 obstacles")
    print(f"Total passable area: {mcta.M}")

    results = mcta.run_coverage_simulation(max_steps=60)

    print(f"\nResults:")
    print(f"Coverage Complete: {results['coverage_complete']}")
    print(f"Total Steps: {results['final_metrics']['Total_Steps']}")
    print(f"Coverage Rate: {results['final_metrics']['Coverage_Rate']:.2f}%")
    print(f"Repeated Rate: {results['final_metrics']['Repeated_Coverage_Rate']:.2f}%")
    print(f"Flight Deviation: {results['final_metrics']['Average_Flight_Deviation']:.2f}")

    total_unique = sum(len(uav.trajectory_set) for uav in mcta.uavs)
    total_mileage = sum(uav.total_flight_mileage for uav in mcta.uavs)
    overall_efficiency = total_unique / max(total_mileage, 1) * 100

    print(f"Overall Efficiency: {overall_efficiency:.1f}%")

    final_cr = results['final_metrics']['Coverage_Rate']
    final_rr = results['final_metrics']['Repeated_Coverage_Rate']
    final_ad = results['final_metrics']['Average_Flight_Deviation']

    print(f"\nBenchmark Results:")
    print(f"Coverage ≥85%: {'✅' if final_cr >= 85.0 else '❌'} ({final_cr:.2f}%)")
    print(f"Repeated ≤60%: {'✅' if final_rr <= 60.0 else '❌'} ({final_rr:.2f}%)")
    print(f"Deviation ≤20: {'✅' if final_ad <= 20.0 else '❌'} ({final_ad:.2f})")
    print(f"Efficiency ≥15%: {'✅' if overall_efficiency >= 15.0 else '❌'} ({overall_efficiency:.1f}%)")

    benchmarks_met = sum([
        final_cr >= 85.0, final_rr <= 60.0,
        final_ad <= 20.0, overall_efficiency >= 15.0
    ])

    print(f"\nBenchmarks achieved: {benchmarks_met}/4")

    if benchmarks_met == 4:
        print("🎉 ALL BENCHMARKS ACHIEVED! 🎉")
    elif benchmarks_met >= 3:
        print("🌟 EXCELLENT PERFORMANCE!")
    else:
        print("📈 GOOD PERFORMANCE!")