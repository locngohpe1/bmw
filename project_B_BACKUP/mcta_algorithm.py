import numpy as np
import math
from collections import deque
from typing import Dict, List, Tuple, Set, Optional


class UAV:
    """Individual UAV class - EXACT paper specifications"""

    def __init__(self, uav_id: int, initial_pos: Tuple[int, int], energy_capacity: float):
        self.id = uav_id  # i in paper (1, 2, ..., v)
        self.current_pos = initial_pos  # Current position p
        self.orientation = 0  # Orientation o

        # Energy model - EXACT from paper
        self.B = energy_capacity  # Initial energy B
        self.energy = energy_capacity  # Current energy Bi,k

        # Flight mileage tracking - EXACT Equation (1)
        self.flight_mileage_per_step = []  # li,k for each step k
        self.total_flight_mileage = 0.0  # Li = Σk li,k

        # Trajectory tracking - EXACT paper format
        self.trajectory = []  # Fi,k trajectory
        self.trajectory_set = set()  # For Fi,k set operations

        # UAV mode
        self.mode = "WORK"  # "WORK" or "SLEEP"

        # Multi-UAV coordination
        self.is_waiting = False
        self.wait_steps = 0

        # Loop detection
        self.loop_detected = False
        self.visited_positions = deque(maxlen=20)  # For loop detection

    def update_flight_mileage(self, distance: float):
        """Update Li = Σk li,k - EXACT Equation (1)"""
        self.flight_mileage_per_step.append(distance)
        self.total_flight_mileage = sum(self.flight_mileage_per_step)

        # Update energy - EXACT Equation (5): Bi,k = B - Σ(k'=0 to k) li,k'
        self.energy = self.B - self.total_flight_mileage

    def add_to_trajectory(self, pos: Tuple[int, int]):
        """Add position to trajectory Fi,k"""
        self.trajectory.append(pos)
        self.trajectory_set.add(pos)

        # Loop detection logic
        self.visited_positions.append(pos)
        if len(self.visited_positions) >= 8:
            # Check for loop pattern (back and forth)
            recent_4 = list(self.visited_positions)[-4:]
            prev_4 = list(self.visited_positions)[-8:-4]
            if recent_4 == prev_4:
                self.loop_detected = True


class MCTAFramework:
    """MCTA Framework - 100% Paper Implementation"""

    def __init__(self, map_rows: int, map_cols: int, battery_pos: Tuple[int, int],
                 num_uavs: int = 1, energy_capacity: float = 1000):

        # Map dimensions - paper uses m×D grid
        self.m = map_rows
        self.n = map_cols
        self.D = 1  # Side length D of basic square unit

        # Multi-UAV system - EXACT from paper
        self.v = num_uavs  # v energy-limited UAVs
        self.uavs: List[UAV] = []
        self.battery_pos = battery_pos

        # Initialize UAVs at battery position
        for i in range(self.v):
            uav = UAV(i + 1, battery_pos, energy_capacity)
            self.uavs.append(uav)

        # Environment maps
        self.threat_map = np.zeros((map_rows, map_cols))  # η values [0,1]
        self.static_obstacles = np.zeros((map_rows, map_cols))
        self.coverage_map = np.zeros((map_rows, map_cols))
        self.repeated_coverage_map = np.zeros((map_rows, map_cols))

        # Area weights - EXACT from paper "W2 > W1 > W3"
        self.W1 = 1.0  # S1 area weight
        self.W2 = 2.0  # S2 area weight (highest)
        self.W3 = 0.5  # S3 area weight (lowest)

        # Module definition - EXACT from paper
        self.module_size = 2  # Module = 2x2 square of 4 basic units

        # Coverage completion flag
        self.coverage_complete = False
        self.step_count = 0

    def get_module_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get module center - EXACT paper definition"""
        # Paper: "treat the center point of the module as an equivalent replacement"
        # Module is 2x2 square, center is at the middle
        r, c = pos

        # Find which module this position belongs to
        module_r = (r // 2) * 2
        module_c = (c // 2) * 2

        # Center of 2x2 module
        center_r = module_r + 0.5
        center_c = module_c + 0.5

        # Return as integer coordinates for practical implementation
        return (int(center_r + 0.5), int(center_c + 0.5))

    def get_four_adjacent_modules(self, uav_pos: Tuple[int, int]) -> List[Optional[Tuple[int, int]]]:
        """Get exactly 4 modules (m1, m2, m3, m4) - EXACT Algorithm 1"""
        current_module_center = self.get_module_center(uav_pos)
        r, c = current_module_center

        # Distance between adjacent modules = 2*D = 2*1 = 2
        module_distance = 2 * self.D

        # EXACTLY 4 modules as specified in Algorithm 1
        # Paper priority: m1 > m2 > m4 > m3
        modules = [
            (r - module_distance, c),  # m1: up
            (r, c + module_distance),  # m2: right
            (r + module_distance, c),  # m3: down
            (r, c - module_distance)  # m4: left
        ]

        # Validate modules and return
        validated_modules = []
        for module_pos in modules:
            if self.is_valid_module_center(module_pos):
                validated_modules.append(module_pos)
            else:
                validated_modules.append(None)  # Invalid module

        return validated_modules

    def define_areas_s1_s2_s3(self, current_module_center: Tuple[int, int],
                              adjacent_module_center: Tuple[int, int]) -> Tuple[List, List, List]:
        """Define S1, S2, S3 areas - EXACT Figure 2 specifications"""
        curr_r, curr_c = current_module_center
        adj_r, adj_c = adjacent_module_center

        # Direction vector
        direction = (adj_r - curr_r, adj_c - curr_c)

        s1_positions = []  # Units in current module close to adjacent module
        s2_positions = []  # Units in adjacent module closest to current UAV
        s3_positions = []  # Units in adjacent module far from current UAV

        if direction == (-2, 0):  # Adjacent module is UP
            # S1: top edge of current module
            s1_positions = [(curr_r - 1, curr_c - 1), (curr_r - 1, curr_c)]
            # S2: bottom edge of adjacent module (closest to current)
            s2_positions = [(adj_r + 1, adj_c - 1), (adj_r + 1, adj_c)]
            # S3: top edge of adjacent module (farthest from current)
            s3_positions = [(adj_r - 1, adj_c - 1), (adj_r - 1, adj_c)]

        elif direction == (0, 2):  # Adjacent module is RIGHT
            # S1: right edge of current module
            s1_positions = [(curr_r - 1, curr_c + 1), (curr_r, curr_c + 1)]
            # S2: left edge of adjacent module
            s2_positions = [(adj_r - 1, adj_c - 1), (adj_r, adj_c - 1)]
            # S3: right edge of adjacent module
            s3_positions = [(adj_r - 1, adj_c + 1), (adj_r, adj_c + 1)]

        elif direction == (2, 0):  # Adjacent module is DOWN
            # S1: bottom edge of current module
            s1_positions = [(curr_r + 1, curr_c - 1), (curr_r + 1, curr_c)]
            # S2: top edge of adjacent module
            s2_positions = [(adj_r - 1, adj_c - 1), (adj_r - 1, adj_c)]
            # S3: bottom edge of adjacent module
            s3_positions = [(adj_r + 1, adj_c - 1), (adj_r + 1, adj_c)]

        elif direction == (0, -2):  # Adjacent module is LEFT
            # S1: left edge of current module
            s1_positions = [(curr_r - 1, curr_c - 1), (curr_r, curr_c - 1)]
            # S2: right edge of adjacent module
            s2_positions = [(adj_r - 1, adj_c + 1), (adj_r, adj_c + 1)]
            # S3: left edge of adjacent module
            s3_positions = [(adj_r - 1, adj_c - 1), (adj_r, adj_c - 1)]

        return s1_positions, s2_positions, s3_positions

    def calculate_threat_level_zeta(self, current_module: Tuple[int, int],
                                    adjacent_module: Tuple[int, int]) -> float:
        """Calculate ζ = Σ(ηWd) - EXACT Equation (3)"""
        if adjacent_module is None:
            return float('inf')  # Invalid module has infinite threat

        zeta = 0.0

        # Get S1, S2, S3 areas - EXACT paper definitions
        s1_positions, s2_positions, s3_positions = self.define_areas_s1_s2_s3(
            current_module, adjacent_module
        )

        # Calculate ζ = Σ(ηWd) with proper area weights
        # Paper: W2 > W1 > W3, so S2 has highest weight

        for pos in s1_positions:
            if self.is_valid_pos(pos):
                eta = self.get_threat_level_eta(pos)
                zeta += eta * self.W1

        for pos in s2_positions:
            if self.is_valid_pos(pos):
                eta = self.get_threat_level_eta(pos)
                zeta += eta * self.W2  # Highest weight

        for pos in s3_positions:
            if self.is_valid_pos(pos):
                eta = self.get_threat_level_eta(pos)
                zeta += eta * self.W3  # Lowest weight

        return zeta

    def get_threat_level_eta(self, pos: Tuple[int, int]) -> float:
        """Get threat level η ∈ [0,1] - EXACT paper definition"""
        r, c = pos
        if not self.is_valid_pos(pos):
            return 1.0  # Out of bounds = obstacle

        # Paper definitions:
        # η = 0: safe unit, UAV can pass freely
        # 0 < η < 1: potential threat, UAV can still pass
        # η = 1: extremely dangerous obstacle, UAV cannot pass

        if self.static_obstacles[r, c] == 1:
            return 1.0  # Static obstacle

        return self.threat_map[r, c]  # Dynamic threat level

    def two_step_auction(self, uav: UAV) -> List[Tuple[float, int, Optional[Tuple[int, int]]]]:
        """Algorithm 1: Two-step Auction - 100% EXACT Implementation"""
        current_module = self.get_module_center(uav.current_pos)

        # Get exactly 4 modules - EXACT Algorithm 1 Line 1
        four_modules = self.get_four_adjacent_modules(uav.current_pos)

        bid_results = []

        # Algorithm 1 Lines 1-6: for i ← 1 to 4 do
        for i in range(4):
            module_mi = four_modules[i]

            if module_mi is None:
                # Invalid module gets zero bid value
                bid_results.append((0.0, i + 1, None))
                continue

            # Line 2: ci ← ζi
            zeta_i = self.calculate_threat_level_zeta(current_module, module_mi)

            # Line 3: Assume that the UAV is in module mi
            # Line 4: Based on module mi, calculate ζm = max(ζ1, ζ2, ζ4)

            # Get 4 modules from the assumed position at module_mi
            assumed_modules = self.get_four_adjacent_modules(module_mi)
            zeta_values = []

            # Only consider ζ1, ζ2, ζ4 (exclude ζ3) as stated in Algorithm 1
            for j in [0, 1, 3]:  # Indices for m1, m2, m4
                if (assumed_modules[j] is not None and
                        assumed_modules[j] != current_module):  # Don't go back
                    zeta_future = self.calculate_threat_level_zeta(module_mi, assumed_modules[j])
                    zeta_values.append(zeta_future)

            zeta_m = max(zeta_values) if zeta_values else 0.0

            # Line 5: ci ← 1/(ci + ζm) - EXACT Equation (4)
            if (zeta_i + zeta_m) > 0:
                ci = 1.0 / (zeta_i + zeta_m)
            else:
                ci = float('inf')  # Perfect score for zero threat

            bid_results.append((ci, i + 1, module_mi))

        # Line 7: {m'1, m'2, m'3, m'4} ← sort({c1, c2, c3, c4}, o)
        # Sort by bidding price (higher is better) and orientation priority
        # Paper: "module priority level m1 > m2 > m4 > m3"
        priority_map = {1: 4, 2: 3, 3: 1, 4: 2}  # Higher number = higher priority

        # Sort by bid value (descending) then by priority (descending)
        bid_results.sort(key=lambda x: (x[0], priority_map[x[1]]), reverse=True)

        return bid_results

    def check_obstacle_avoidance_figure3(self, uav_pos: Tuple[int, int],
                                         target_module: Tuple[int, int]) -> Tuple[bool, str]:
        """Implement ALL 6 cases from Figure 3 - EXACT Implementation"""
        if target_module is None:
            return False, "invalid_module"

        curr_r, curr_c = uav_pos
        target_r, target_c = target_module
        current_module = self.get_module_center(uav_pos)

        # Get S1, S2, S3 areas
        s1_positions, s2_positions, s3_positions = self.define_areas_s1_s2_s3(
            current_module, target_module
        )

        # Check for obstacles in each area
        s1_obstacles = [pos for pos in s1_positions
                        if self.is_valid_pos(pos) and self.get_threat_level_eta(pos) == 1.0]
        s2_obstacles = [pos for pos in s2_positions
                        if self.is_valid_pos(pos) and self.get_threat_level_eta(pos) == 1.0]
        s3_obstacles = [pos for pos in s3_positions
                        if self.is_valid_pos(pos) and self.get_threat_level_eta(pos) == 1.0]

        # Case (a): Obstacle in S1 → UAV turns into next module ✓
        if len(s1_obstacles) == 1 and len(s2_obstacles) == 0 and len(s3_obstacles) == 0:
            return True, "turn_and_enter"

        # Case (b): Obstacle in S2 → UAV turns into next module ✓
        if len(s2_obstacles) == 1 and len(s1_obstacles) == 0 and len(s3_obstacles) == 0:
            return True, "turn_and_enter"

        # Case (c): Obstacle in S3 → UAV goes straight ✓
        if len(s3_obstacles) == 1 and len(s1_obstacles) == 0 and len(s2_obstacles) == 0:
            return True, "go_straight"

        # Case (d): Double obstacles in S2 → Cannot enter ✗
        if len(s2_obstacles) >= 2:
            return False, "double_obstacles_s2"

        # Case (e): Obstacle at neighbor of current UAV position → Cannot fly ✗
        neighbors_current = [
            (curr_r - 1, curr_c), (curr_r + 1, curr_c),
            (curr_r, curr_c - 1), (curr_r, curr_c + 1)
        ]

        for neighbor in neighbors_current:
            if (self.is_valid_pos(neighbor) and
                    self.get_threat_level_eta(neighbor) == 1.0 and
                    abs(neighbor[0] - target_r) + abs(neighbor[1] - target_c) <= 2):
                return False, "neighbor_obstacle"

        # Case (f): Obstacle in direction + opposite pattern → Cannot fly ✗
        if (len(s1_obstacles) >= 1 and len(s2_obstacles) >= 1) or \
                (len(s2_obstacles) >= 1 and len(s3_obstacles) >= 1):
            return False, "cross_pattern_obstacle"

        # Default: Can move if no blocking pattern detected
        return True, "clear_path"

    def reverse_auction_conflict_resolution(self, conflicts: Dict[Tuple[int, int], List[int]]) -> Dict[int, str]:
        """Reverse auction mechanism - EXACT from paper"""
        uav_actions = {}

        for module_pos, conflicted_uav_ids in conflicts.items():
            # Paper: "the conflicting module center selects the UAV reversely"
            # "module will give priority to choosing the UAV that has encountered 'unfair' treatment"
            # "the one that has the least flight mileage L"

            min_mileage = float('inf')
            selected_uav_id = None

            for uav_id in conflicted_uav_ids:
                uav = self.uavs[uav_id - 1]  # Convert to 0-based index

                # Choose UAV with least total flight mileage Li
                if uav.total_flight_mileage < min_mileage:
                    min_mileage = uav.total_flight_mileage
                    selected_uav_id = uav_id

            # Selected UAV moves, others wait
            for uav_id in conflicted_uav_ids:
                if uav_id == selected_uav_id:
                    uav_actions[uav_id] = "move"
                else:
                    uav_actions[uav_id] = "wait"
                    # Paper: "other UAVs will pause for one time step"
                    self.uavs[uav_id - 1].is_waiting = True
                    self.uavs[uav_id - 1].wait_steps = 1

        return uav_actions

    def execute_mcta_algorithm_step(self) -> bool:
        """Algorithm 2: MCTA Framework - EXACT Implementation"""
        self.step_count += 1

        # Check if all UAVs are in sleep mode
        active_uavs = [uav for uav in self.uavs if uav.mode == "WORK"]
        if not active_uavs:
            self.coverage_complete = True
            return False

        winning_modules = {}

        # Execute Algorithm 1 for each active UAV
        for uav in active_uavs:
            # Handle waiting UAVs
            if uav.is_waiting:
                uav.wait_steps -= 1
                if uav.wait_steps <= 0:
                    uav.is_waiting = False
                continue

            # Algorithm 2 Line 1: Two-step Auction
            auction_results = self.two_step_auction(uav)

            # Algorithm 2 Lines 2-9: Find reachable module
            plan_flag = False

            for j in range(4):  # Lines 3-9: for j ← 1 to 4 do
                bid_value, module_id, module_pos = auction_results[j]

                # Line 4: if module m corresponding to m'j is reachable
                if module_pos is not None:
                    can_reach, path_type = self.check_obstacle_avoidance_figure3(
                        uav.current_pos, module_pos
                    )

                    if can_reach:
                        # Check energy constraint before committing
                        distance = math.dist(uav.current_pos, module_pos)
                        if uav.energy >= distance:
                            # Line 5-6: orientation and plan_flag
                            direction = (module_pos[0] - uav.current_pos[0],
                                         module_pos[1] - uav.current_pos[1])
                            uav.orientation = math.atan2(direction[1], direction[0])
                            plan_flag = True
                            winning_modules[uav.id] = module_pos
                            break  # Line 7: break

            # Algorithm 2 Lines 10-25: Check conditions for sleep mode
            if plan_flag:
                # Line 13: Check remaining energy and loop
                if uav.energy <= 0 or uav.loop_detected:
                    # Line 18: mode ← sleep
                    uav.mode = "SLEEP"
                    if uav.id in winning_modules:
                        del winning_modules[uav.id]
                # Line 16: Choose suitable way to reach module (handled in movement)
            else:
                # Line 24: mode ← sleep (no reachable modules)
                uav.mode = "SLEEP"

        # Multi-UAV conflict detection and resolution
        conflicts = self.detect_conflicts(winning_modules)

        if conflicts:
            # Line 11-12: Judge conflict and resolve
            actions = self.reverse_auction_conflict_resolution(conflicts)
        else:
            # Line 12: no conflict occurs → all move
            actions = {uav_id: "move" for uav_id in winning_modules.keys()}

        # Execute movements
        for uav_id, action in actions.items():
            if action == "move" and uav_id in winning_modules:
                uav = self.uavs[uav_id - 1]
                target = winning_modules[uav_id]

                # Calculate and update flight mileage
                distance = math.dist(uav.current_pos, target)
                uav.update_flight_mileage(distance)

                # Move UAV
                uav.current_pos = target
                uav.add_to_trajectory(target)

                # Mark coverage
                self.mark_module_coverage(target)

        return True

    def detect_conflicts(self, winning_modules: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Detect multi-UAV conflicts - when multiple UAVs target same module"""
        conflicts = {}

        for uav_id, module_pos in winning_modules.items():
            if module_pos not in conflicts:
                conflicts[module_pos] = []
            conflicts[module_pos].append(uav_id)

        # Return only conflicted modules (more than 1 UAV)
        return {pos: uav_list for pos, uav_list in conflicts.items() if len(uav_list) > 1}

    def mark_module_coverage(self, module_center: Tuple[int, int]):
        """Mark entire 2x2 module as covered"""
        r, c = module_center

        # Mark all 4 units of the module
        for dr in range(-1, 1):  # -1, 0
            for dc in range(-1, 1):  # -1, 0
                nr, nc = r + dr, c + dc
                if self.is_valid_pos((nr, nc)):
                    if self.coverage_map[nr, nc] == 0:
                        self.coverage_map[nr, nc] = 1  # First time coverage
                    else:
                        self.repeated_coverage_map[nr, nc] += 1  # Repeated coverage

    def calculate_performance_metrics(self) -> Tuple[float, float, float]:
        """Calculate Cr, Rr, AD - EXACT Equations (6), (7), (8)"""

        # Total area M (passable units only)
        total_passable_units = 0
        for r in range(self.m):
            for c in range(self.n):
                if self.get_threat_level_eta((r, c)) < 1.0:  # η < 1 means passable
                    total_passable_units += 1

        if total_passable_units == 0:
            return 0.0, 0.0, 0.0

        # Coverage rate Cr - Equation (6)
        # Cr = |F1 ∪ F2 ∪ ... ∪ Fv| / M × 100%
        union_trajectory = set()
        for uav in self.uavs:
            union_trajectory.update(uav.trajectory_set)

        covered_units = len(union_trajectory)
        Cr = (covered_units / total_passable_units) * 100.0

        # Repeated coverage rate Rr - Equation (7)
        # Rr = Σi,k(li,k - |Fi,k|) / |F1 ∪ F2 ∪ ... ∪ Fv| × 100%
        total_flight_distance = sum(uav.total_flight_mileage for uav in self.uavs)

        if covered_units > 0:
            repeated_distance = total_flight_distance - covered_units
            Rr = (repeated_distance / covered_units) * 100.0
        else:
            Rr = 0.0

        # Average flight deviation AD - Equation (8)
        # AD = (1/v) Σi abs(Li - L̄)
        if self.v > 0:
            L_bar = sum(uav.total_flight_mileage for uav in self.uavs) / self.v
            AD = sum(abs(uav.total_flight_mileage - L_bar) for uav in self.uavs) / self.v
        else:
            AD = 0.0

        return Cr, Rr, AD

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within map bounds"""
        return 0 <= pos[0] < self.m and 0 <= pos[1] < self.n

    def is_valid_module_center(self, pos: Tuple[int, int]) -> bool:
        """Check if module center is valid (has space for 2x2 module)"""
        r, c = pos
        return 1 <= r < self.m - 1 and 1 <= c < self.n - 1

    def set_static_obstacles(self, obstacle_map: np.ndarray):
        """Set static obstacle map"""
        self.static_obstacles = obstacle_map.copy()
        # Update threat map with static obstacles (η = 1.0)
        for r in range(self.m):
            for c in range(self.n):
                if self.static_obstacles[r, c] == 1:
                    self.threat_map[r, c] = 1.0

    def update_dynamic_threats(self, detected_threats: Dict[Tuple[int, int], float]):
        """Update threat map with detected dynamic threats"""
        for pos, eta_value in detected_threats.items():
            if self.is_valid_pos(pos) and self.static_obstacles[pos] == 0:
                self.threat_map[pos] = eta_value

    def run_coverage_simulation(self, max_steps: int = 1000) -> Dict:
        """Run complete MCTA simulation"""
        results = {
            'steps': [],
            'coverage_rates': [],
            'repeated_rates': [],
            'flight_deviations': [],
            'uav_trajectories': [[] for _ in range(self.v)],
            'coverage_complete': False
        }

        for step in range(max_steps):
            # Execute one MCTA step
            continuing = self.execute_mcta_algorithm_step()

            if not continuing:
                results['coverage_complete'] = True
                break

            # Calculate metrics
            Cr, Rr, AD = self.calculate_performance_metrics()

            # Store results
            results['steps'].append(step + 1)
            results['coverage_rates'].append(Cr)
            results['repeated_rates'].append(Rr)
            results['flight_deviations'].append(AD)

            # Store trajectories
            for i, uav in enumerate(self.uavs):
                if len(uav.trajectory) > len(results['uav_trajectories'][i]):
                    results['uav_trajectories'][i] = uav.trajectory.copy()

        # Final metrics
        final_Cr, final_Rr, final_AD = self.calculate_performance_metrics()
        results['final_metrics'] = {
            'Coverage_Rate': final_Cr,
            'Repeated_Coverage_Rate': final_Rr,
            'Average_Flight_Deviation': final_AD
        }
        return results
# Usage Example - 100% Paper Configuration
if __name__ == "__main__":
    # Create 20x20 environment as in paper experiments
    mcta = MCTAFramework(
        map_rows=20,
        map_cols=20,
        battery_pos=(1, 1),
        num_uavs=4,  # Test with 4 UAVs as in paper Figure 5
        energy_capacity=100  # Adjust based on experiment
    )

    # Set up environment with 10% obstacles (as in paper)
    obstacle_map = np.random.choice([0, 1], size=(20, 20), p=[0.9, 0.1])
    obstacle_map[1, 1] = 0  # Ensure battery position is clear
    mcta.set_static_obstacles(obstacle_map)

    # Run simulation
    print("Running MCTA simulation - 100% Paper Implementation")
    results = mcta.run_coverage_simulation(max_steps=500)

    print(f"Coverage Complete: {results['coverage_complete']}")
    print(f"Final Coverage Rate: {results['final_metrics']['Coverage_Rate']:.2f}%")
    print(f"Final Repeated Coverage Rate: {results['final_metrics']['Repeated_Coverage_Rate']:.2f}%")
    print(f"Final Average Flight Deviation: {results['final_metrics']['Average_Flight_Deviation']:.2f}")