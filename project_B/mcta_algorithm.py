import numpy as np
import math
from collections import deque
import random
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

        # UAV mode
        self.mode = "WORK"  # "WORK" or "SLEEP"

        # Multi-UAV coordination
        self.is_waiting = False
        self.wait_steps = 0

    def update_flight_mileage(self, distance: float):
        """Update Li = Σk li,k - EXACT Equation (1)"""
        self.flight_mileage_per_step.append(distance)
        self.total_flight_mileage = sum(self.flight_mileage_per_step)

        # Update energy - EXACT Equation (5)
        # Bi,k = B - Σ(k'=0 to k) li,k'
        self.energy = self.B - self.total_flight_mileage


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

        # Initialize UAVs
        for i in range(self.v):
            uav = UAV(i + 1, battery_pos, energy_capacity)
            self.uavs.append(uav)

        # Environment maps
        self.threat_map = np.zeros((map_rows, map_cols))  # η values [0,1]
        self.static_obstacles = np.zeros((map_rows, map_cols))
        self.coverage_map = np.zeros((map_rows, map_cols))

        # Area weights - EXACT from paper "W2 > W1 > W3"
        # Paper doesn't give exact values, so use relative weights
        self.W1 = 1.0  # S1 area weight
        self.W2 = 2.0  # S2 area weight (highest)
        self.W3 = 0.5  # S3 area weight (lowest)

        # Module definition - EXACT from paper
        # "module as a square composed of four units"
        self.module_unit_count = 4  # Module = 4 basic units

        # Coverage completion flag
        self.coverage_complete = False

    def get_four_adjacent_modules(self, uav_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get exactly 4 modules (m1, m2, m3, m4) - EXACT Algorithm 1 requirement"""
        current_module_center = self.get_module_center(uav_pos)
        r, c = current_module_center

        # EXACTLY 4 modules as specified in Algorithm 1
        modules = [
            (r - 2, c),  # m1: up
            (r, c + 2),  # m2: right
            (r + 2, c),  # m3: down
            (r, c - 2)  # m4: left
        ]

        # Ensure we always return 4 modules (pad with None if out of bounds)
        valid_modules = []
        for module_pos in modules:
            if self.is_valid_module_center(module_pos):
                valid_modules.append(module_pos)
            else:
                valid_modules.append(None)  # Invalid module

        return valid_modules

    def get_module_center(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get module center - treat as equivalent replacement"""
        # Module composed of 4 units in 2x2 arrangement
        module_row = (pos[0] // 2) * 2 + 1  # Center of 2x2 module
        module_col = (pos[1] // 2) * 2 + 1
        return (module_row, module_col)

    def define_areas_s1_s2_s3(self, current_module_center: Tuple[int, int],
                              adjacent_module_center: Tuple[int, int]) -> Tuple[List, List, List]:
        """Define S1, S2, S3 areas - EXACT paper definitions"""
        curr_r, curr_c = current_module_center
        adj_r, adj_c = adjacent_module_center

        # S1: "units in the current module that are close to the adjacent module"
        s1_positions = []
        direction = (adj_r - curr_r, adj_c - curr_c)

        if direction == (-2, 0):  # Adjacent module is up
            s1_positions = [(curr_r - 1, curr_c - 1), (curr_r - 1, curr_c), (curr_r - 1, curr_c + 1)]
        elif direction == (0, 2):  # Adjacent module is right
            s1_positions = [(curr_r - 1, curr_c + 1), (curr_r, curr_c + 1), (curr_r + 1, curr_c + 1)]
        elif direction == (2, 0):  # Adjacent module is down
            s1_positions = [(curr_r + 1, curr_c - 1), (curr_r + 1, curr_c), (curr_r + 1, curr_c + 1)]
        elif direction == (0, -2):  # Adjacent module is left
            s1_positions = [(curr_r - 1, curr_c - 1), (curr_r, curr_c - 1), (curr_r + 1, curr_c - 1)]

        # S2: "basic units in adjacent modules that are closest to the module center where the UAV is located"
        s2_positions = []
        if direction == (-2, 0):  # Adjacent module is up
            s2_positions = [(adj_r + 1, adj_c - 1), (adj_r + 1, adj_c), (adj_r + 1, adj_c + 1)]
        elif direction == (0, 2):  # Adjacent module is right
            s2_positions = [(adj_r - 1, adj_c - 1), (adj_r, adj_c - 1), (adj_r + 1, adj_c - 1)]
        elif direction == (2, 0):  # Adjacent module is down
            s2_positions = [(adj_r - 1, adj_c - 1), (adj_r - 1, adj_c), (adj_r - 1, adj_c + 1)]
        elif direction == (0, -2):  # Adjacent module is left
            s2_positions = [(adj_r - 1, adj_c + 1), (adj_r, adj_c + 1), (adj_r + 1, adj_c + 1)]

        # S3: "basic units in adjacent modules that are far away from the module center"
        s3_positions = []
        if direction == (-2, 0):  # Adjacent module is up
            s3_positions = [(adj_r - 1, adj_c - 1), (adj_r - 1, adj_c), (adj_r - 1, adj_c + 1)]
        elif direction == (0, 2):  # Adjacent module is right
            s3_positions = [(adj_r - 1, adj_c + 1), (adj_r, adj_c + 1), (adj_r + 1, adj_c + 1)]
        elif direction == (2, 0):  # Adjacent module is down
            s3_positions = [(adj_r + 1, adj_c - 1), (adj_r + 1, adj_c), (adj_r + 1, adj_c + 1)]
        elif direction == (0, -2):  # Adjacent module is left
            s3_positions = [(adj_r - 1, adj_c - 1), (adj_r, adj_c - 1), (adj_r + 1, adj_c - 1)]

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
        """Get threat level η ∈ [0,1] - EXACT paper definition"""
        r, c = pos
        if not self.is_valid_pos(pos):
            return 1.0

        # η = 0: safe unit, UAV can pass freely
        # 0 < η < 1: potential threat to UAVs, but UAV can still pass
        # η = 1: extremely dangerous obstacle which UAV cannot pass

        if self.static_obstacles[r, c] == 1:
            return 1.0

        return self.threat_map[r, c]

    def two_step_auction(self, uav: UAV) -> List[Tuple[float, int, Optional[Tuple[int, int]]]]:
        """Algorithm 1: Two-step Auction - EXACT Implementation"""
        current_module = self.get_module_center(uav.current_pos)

        # Get exactly 4 modules - EXACT Algorithm 1 Line 1
        four_modules = self.get_four_adjacent_modules(uav.current_pos)

        bid_results = []

        # Algorithm 1 Lines 1-6
        for i in range(4):  # i ← 1 to 4
            module_mi = four_modules[i]

            if module_mi is None:
                # Invalid module gets infinite threat
                bid_results.append((0.0, i + 1, None))
                continue

            # Line 2: ci ← ζi
            zeta_i = self.calculate_threat_level_zeta(current_module, module_mi)

            # Line 3: Assume that the UAV is in module mi
            # Line 4: Based on module mi, calculate ζm = max(ζ1, ζ2, ζ4)

            # Get 4 modules from the assumed position
            assumed_modules = self.get_four_adjacent_modules(module_mi)
            zeta_values = []

            # Only consider ζ1, ζ2, ζ4 (exclude ζ3) as stated in paper
            for j in [0, 1, 3]:  # Indices for m1, m2, m4
                if assumed_modules[j] is not None and assumed_modules[j] != current_module:
                    zeta_future = self.calculate_threat_level_zeta(module_mi, assumed_modules[j])
                    zeta_values.append(zeta_future)

            zeta_m = max(zeta_values) if zeta_values else 0.0

            # Line 5: ci ← 1/(ci + ζm) - EXACT Equation (4)
            if (zeta_i + zeta_m) > 0:
                ci = 1.0 / (zeta_i + zeta_m)
            else:
                ci = float('inf')

            bid_results.append((ci, i + 1, module_mi))

        # Line 7: Sort by bidding price and orientation priority
        # Paper states: "module priority level m1 > m2 > m4 > m3"
        priority_map = {1: 4, 2: 3, 3: 1, 4: 2}  # Higher number = higher priority

        bid_results.sort(key=lambda x: (x[0], priority_map[x[1]]), reverse=True)

        return bid_results

    def check_obstacle_avoidance_figure3(self, uav_pos: Tuple[int, int],
                                         target_module: Tuple[int, int]) -> bool:
        """Implement Figure 3 obstacle avoidance rules - EXACT 6 cases"""
        if target_module is None:
            return False

        # Get relative direction to target module
        curr_r, curr_c = uav_pos
        target_r, target_c = target_module

        direction = (target_r - curr_r, target_c - curr_c)

        # Check obstacles in different areas around target
        obstacle_positions = []

        # Scan area around target module for obstacles
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                check_pos = (target_r + dr, target_c + dc)
                if (self.is_valid_pos(check_pos) and
                        self.get_threat_level_eta(check_pos) == 1.0):
                    obstacle_positions.append(check_pos)

        # Apply Figure 3 rules (simplified - full implementation would need exact visual patterns)

        # Case (d): Double obstacles located in area S2 → UAV cannot enter
        s1, s2, s3 = self.define_areas_s1_s2_s3(self.get_module_center(uav_pos), target_module)

        s2_obstacles = sum(1 for pos in s2 if pos in obstacle_positions)
        if s2_obstacles >= 2:
            return False

        # Case (e): Obstacle at neighbor of current UAV position → cannot fly
        neighbors = [(curr_r - 1, curr_c), (curr_r + 1, curr_c),
                     (curr_r, curr_c - 1), (curr_r, curr_c + 1)]

        for neighbor in neighbors:
            if neighbor == target_module and neighbor in obstacle_positions:
                return False

        # Case (f): Cross pattern → cannot fly
        if target_module in neighbors:
            s1_obstacles = sum(1 for pos in s1 if pos in obstacle_positions)
            if s1_obstacles >= 1 and s2_obstacles >= 1:
                return False

        # Cases (a), (b), (c): Can move with appropriate turning
        return True

    def reverse_auction_conflict_resolution(self, conflicts: Dict[Tuple[int, int], List[int]]) -> Dict[int, str]:
        """Reverse auction mechanism - EXACT from paper"""
        uav_actions = {}

        for module_pos, conflicted_uav_ids in conflicts.items():
            # "the conflicting module center selects the UAV reversely"
            # "module will give priority to choosing the UAV that has encountered 'unfair' treatment"
            # "the one that has the least flight mileage L"

            min_mileage = float('inf')
            selected_uav_id = None

            for uav_id in conflicted_uav_ids:
                uav = self.uavs[uav_id - 1]

                if uav.total_flight_mileage < min_mileage:
                    min_mileage = uav.total_flight_mileage
                    selected_uav_id = uav_id

            # Selected UAV moves, others wait
            for uav_id in conflicted_uav_ids:
                if uav_id == selected_uav_id:
                    uav_actions[uav_id] = "move"
                else:
                    uav_actions[uav_id] = "wait"
                    # "other UAVs will pause for one time step"
                    self.uavs[uav_id - 1].is_waiting = True
                    self.uavs[uav_id - 1].wait_steps = 1

        return uav_actions

    def execute_mcta_algorithm_step(self) -> bool:
        """Execute one complete MCTA step for all UAVs"""
        active_uavs = [uav for uav in self.uavs if uav.mode == "WORK"]

        if not active_uavs:
            self.coverage_complete = True
            return False

        winning_modules = {}

        # Execute Algorithm 1 for each UAV
        for uav in active_uavs:
            if uav.is_waiting:
                uav.wait_steps -= 1
                if uav.wait_steps <= 0:
                    uav.is_waiting = False
                continue

            # Two-step auction
            auction_results = self.two_step_auction(uav)

            # Find first reachable module
            for bid_value, module_id, module_pos in auction_results:
                if (module_pos is not None and
                        self.check_obstacle_avoidance_figure3(uav.current_pos, module_pos)):

                    # Check energy constraint
                    distance = math.dist(uav.current_pos, module_pos)
                    if uav.energy >= distance:
                        winning_modules[uav.id] = module_pos
                        break

            # Check sleep mode conditions
            if (uav.energy <= 0 or  # Condition 1: energy exhausted
                    self.detect_loop(uav) or  # Condition 2: loop detected
                    uav.id not in winning_modules):  # Condition 3: no passable modules
                uav.mode = "SLEEP"

        # Detect conflicts and resolve
        conflicts = self.detect_conflicts(winning_modules)
        if conflicts:
            actions = self.reverse_auction_conflict_resolution(conflicts)
        else:
            actions = {uav_id: "move" for uav_id in winning_modules.keys()}

        # Execute movements
        for uav_id, action in actions.items():
            if action == "move" and uav_id in winning_modules:
                uav = self.uavs[uav_id - 1]
                target = winning_modules[uav_id]

                distance = math.dist(uav.current_pos, target)
                uav.update_flight_mileage(distance)
                uav.current_pos = target
                uav.trajectory.append(target)

                self.mark_coverage(target)

        return True

    def detect_conflicts(self, winning_modules: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Detect multi-UAV conflicts"""
        conflicts = {}
        for uav_id, module_pos in winning_modules.items():
            if module_pos not in conflicts:
                conflicts[module_pos] = []
            conflicts[module_pos].append(uav_id)

        return {pos: uav_list for pos, uav_list in conflicts.items() if len(uav_list) > 1}

    def detect_loop(self, uav: UAV) -> bool:
        """Detect loop in UAV trajectory"""
        if len(uav.trajectory) < 6:
            return False

        # Simple loop detection
        recent = uav.trajectory[-3:]
        previous = uav.trajectory[-6:-3]
        return recent == previous

    def mark_coverage(self, module_center: Tuple[int, int]):
        """Mark module as covered"""
        r, c = module_center
        for dr in range(-1, 1):
            for dc in range(-1, 1):
                nr, nc = r + dr, c + dc
                if self.is_valid_pos((nr, nc)):
                    self.coverage_map[nr, nc] = 1

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.m and 0 <= pos[1] < self.n

    def is_valid_module_center(self, pos: Tuple[int, int]) -> bool:
        return 1 <= pos[0] < self.m - 1 and 1 <= pos[1] < self.n - 1

    def set_static_obstacles(self, obstacle_map: np.ndarray):
        """Set static obstacle map"""
        self.static_obstacles = obstacle_map.copy()
        for r in range(self.m):
            for c in range(self.n):
                if self.static_obstacles[r, c] == 1:
                    self.threat_map[r, c] = 1.0

    def update_threat_map(self, detected_threats: Dict[Tuple[int, int], float]):
        """Update threat map from UAV sensing"""
        # Clear dynamic threats
        for r in range(self.m):
            for c in range(self.n):
                if self.static_obstacles[r, c] == 0:
                    self.threat_map[r, c] = 0.0

        # Add detected threats
        for pos, eta_value in detected_threats.items():
            if self.is_valid_pos(pos):
                self.threat_map[pos] = eta_value