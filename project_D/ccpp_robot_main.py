import torch
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GridState(Enum):
    UNVISITED = 0
    VISITED = 1
    OBSTACLE = 2
    DEADLOCK = 3
    UNKNOWN = 4


@dataclass
class Position:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class CCPPRobot:
    def __init__(self, width: int, height: int, sensor_range: int = 2):
        self.width = width
        self.height = height
        self.sensor_range = sensor_range

        # Neural network parameters from paper
        self.A = 1.0
        self.B = 100.0  # Back to reasonable value
        self.D = 100.0  # Back to reasonable value
        self.E = 1000.0
        self.m = 10.0  # Back to paper value
        self.r0 = 2.0  # Back to paper value

        # Initialize grids on GPU
        self.grid_state = torch.zeros((height, width), dtype=torch.int, device=device)
        self.neural_activity = torch.zeros((height, width), dtype=torch.float32, device=device)
        self.external_input = torch.zeros((height, width), dtype=torch.float32, device=device)

        # Robot state
        self.position = Position(0, 0)
        self.backtrack_list = []
        self.path = [self.position]
        
        # ✅ FIXED: Add overlap tracking for correct metrics
        self.count_cell_go_through = 1  # Starting position counts as first coverage move

        # 8-directional movement as in paper
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonal
        ]

        self.initialize_environment()

    def initialize_environment(self):
        """Initialize the environment with unvisited cells"""
        self.grid_state.fill_(GridState.UNVISITED.value)
        self.external_input.fill_(self.E)  # All unvisited initially
        self.neural_activity.fill_(0.0)

        # Mark starting position as visited
        self.grid_state[self.position.y, self.position.x] = GridState.VISITED.value
        self.external_input[self.position.y, self.position.x] = 0.0

    def add_obstacles(self, obstacles: List[Tuple[int, int]]):
        """Add static obstacles to the environment"""
        for x, y in obstacles:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid_state[y, x] = GridState.OBSTACLE.value
                self.external_input[y, x] = -self.E

    def add_dynamic_obstacle(self, x: int, y: int):
        """Add a dynamic obstacle detected by sensors"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid_state[y, x] = GridState.OBSTACLE.value
            self.external_input[y, x] = -self.E

    def remove_dynamic_obstacle(self, x: int, y: int):
        """Remove a dynamic obstacle that moved away"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid_state[y, x] == GridState.OBSTACLE.value:
                self.grid_state[y, x] = GridState.UNVISITED.value
                self.external_input[y, x] = self.E

    def get_neighbors(self, pos: Position) -> List[Position]:
        """Get valid neighboring positions"""
        neighbors = []
        for dx, dy in self.directions:
            new_x, new_y = pos.x + dx, pos.y + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                neighbors.append(Position(new_x, new_y))
        return neighbors

    def calculate_connection_weight(self, pos1: Position, pos2: Position) -> float:
        """Calculate connection weight - Equation (3) from paper: f(a) = m/a"""
        # Use Euclidean distance as specified in paper
        distance = np.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)

        # Equation (3) from paper: f(a) = m/a if 0 < a ≤ r0, 0 if a > r0
        if 0 < distance <= self.r0:
            return self.m / distance
        return 0.0

    def update_neural_activity(self):
        """Update neural activities using shunting short-memory model - Equation (1) from paper"""
        dt = 0.01  # Much smaller time step for numerical stability

        # Create new activity tensor
        new_activity = torch.zeros_like(self.neural_activity)

        for y in range(self.height):
            for x in range(self.width):
                current_pos = Position(x, y)
                current_activity = self.neural_activity[y, x].item()
                external_input = self.external_input[y, x].item()

                # Calculate neighbor influence - Σ(vij[xj]+) exactly as in paper
                neighbor_excitation = 0.0
                neighbors = self.get_neighbors(current_pos)

                for neighbor in neighbors:
                    neighbor_activity = self.neural_activity[neighbor.y, neighbor.x].item()
                    weight = self.calculate_connection_weight(current_pos, neighbor)
                    # Only positive activities contribute ([xj]+)
                    if neighbor_activity > 0:
                        neighbor_excitation += weight * neighbor_activity

                # Equation (1) EXACTLY from paper: dxi/dt = -Axi + (B-xi)[Ii+ + Σvij*xj+] - (D+xi)[Ii-]
                Ii_positive = max(0, external_input)  # [Ii]+
                Ii_negative = max(0, -external_input)  # [Ii]-

                # CRITICAL FIX: Paper equation implementation
                excitatory_term = (self.B - current_activity) * (Ii_positive + neighbor_excitation)
                inhibitory_term = (self.D + current_activity) * Ii_negative
                passive_decay = self.A * current_activity
                dxi_dt = -passive_decay + excitatory_term - inhibitory_term

                # Stability check - clip derivative if too large
                if abs(dxi_dt) > 1000:
                    dxi_dt = 1000 if dxi_dt > 0 else -1000
                # Update with proper integration and bounds checking
                new_val = current_activity + dt * dxi_dt

                # Prevent overflow and maintain paper bounds
                if new_val > 1e6:  # Overflow protection
                    new_val = 1e6
                elif new_val < 0:
                    new_val = 0.0

                new_activity[y, x] = new_val

        self.neural_activity = new_activity
    def select_next_position_with_priority(self) -> Optional[Position]:
        """Select next position using priority template from paper Section 3.1.2"""
        current = self.position
        neighbors = self.get_neighbors(current)

        # Get valid unvisited neighbors with their activities
        candidates = []
        for neighbor in neighbors:
            if self.grid_state[neighbor.y, neighbor.x] == GridState.UNVISITED.value:
                activity = self.neural_activity[neighbor.y, neighbor.x].item()
                candidates.append((neighbor, activity))

        if not candidates:
            return None

        # Find maximum activity
        max_activity = max(candidates, key=lambda x: x[1])[1]
        tolerance = 1e-6  # Small tolerance for floating point comparison

        # Find all candidates with maximum activity (rank one class)
        rank_one_candidates = [pos for pos, act in candidates
                               if abs(act - max_activity) <= tolerance]

        # Apply priority template if more than one candidate in rank one class
        if len(rank_one_candidates) > 1:
            # Priority template from paper Section 3.1.2
            # "The regularity in our prior template is the up and down"
            current = self.position

            # First priority: UP direction
            for candidate in rank_one_candidates:
                if candidate.x == current.x and candidate.y == current.y - 1:  # UP
                    return candidate

            # Second priority: DOWN direction
            for candidate in rank_one_candidates:
                if candidate.x == current.x and candidate.y == current.y + 1:  # DOWN
                    return candidate

            # Third priority: LEFT/RIGHT
            for candidate in rank_one_candidates:
                if candidate.x == current.x - 1 and candidate.y == current.y:  # LEFT
                    return candidate

            for candidate in rank_one_candidates:
                if candidate.x == current.x + 1 and candidate.y == current.y:  # RIGHT
                    return candidate

            # Last priority: Diagonals
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                for candidate in rank_one_candidates:
                    if (candidate.x == current.x + dx and
                            candidate.y == current.y + dy):
                        return candidate
        # Return the candidate with highest activity
        return max(candidates, key=lambda x: x[1])[0]

    def is_deadlock(self) -> bool:
        """Algorithm 2: Deadlock detection exactly as in paper"""
        neighbors = self.get_neighbors(self.position)
        current_activity = self.neural_activity[self.position.y, self.position.x].item()

        # Check if all neighbors are visited or obstacles
        has_unvisited_neighbor = False
        for neighbor in neighbors:
            state = self.grid_state[neighbor.y, neighbor.x]
            if state == GridState.UNVISITED.value:
                has_unvisited_neighbor = True
                break

        if has_unvisited_neighbor:
            return False  # Not deadlock if unvisited neighbors exist

        # If no unvisited neighbors, check activity condition
        for neighbor in neighbors:
            state = self.grid_state[neighbor.y, neighbor.x]
            neighbor_activity = self.neural_activity[neighbor.y, neighbor.x].item()

            if state in [GridState.VISITED.value, GridState.OBSTACLE.value]:
                if neighbor_activity >= current_activity:
                    return False  # Not deadlock if neighbor has higher/equal activity

        return True  # All conditions satisfied - deadlock detected

    def update_backtrack_list(self):
        """Algorithm 1: Updating backtracking List - exactly as in paper"""
        neighbors = self.get_neighbors(self.position)
        unvisited_neighbors = 0

        # Count unvisited neighbors
        for neighbor in neighbors:
            if self.grid_state[neighbor.y, neighbor.x] == GridState.UNVISITED.value:
                unvisited_neighbors += 1

        # Algorithm 1 logic: if current position has unvisited neighbors, add to backtrack list
        if unvisited_neighbors > 0:
            if self.position not in self.backtrack_list:
                self.backtrack_list.append(self.position)

        # Remove positions from backtrack list that no longer have unvisited neighbors
        positions_to_remove = []
        for pos in self.backtrack_list:
            pos_neighbors = self.get_neighbors(pos)
            has_unvisited = False

            for neighbor in pos_neighbors:
                if self.grid_state[neighbor.y, neighbor.x] == GridState.UNVISITED.value:
                    has_unvisited = True
                    break

            if not has_unvisited:
                positions_to_remove.append(pos)

        # Remove invalid positions
        for pos in positions_to_remove:
            self.backtrack_list.remove(pos)

    def dynamic_a_star(self, start: Position, goal: Position) -> List[Position]:
        """Dynamic A* pathfinding algorithm allowing movement through visited cells"""
        def heuristic(pos: Position) -> float:
            return abs(pos.x - goal.x) + abs(pos.y - goal.y)

        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                # Only skip obstacles, allow movement through visited cells for backtracking
                state = self.grid_state[neighbor.y, neighbor.x]
                # ✅ CRITICAL FIX: Allow movement through visited cells for backtracking
                if state == GridState.OBSTACLE.value:
                    continue  # Only block obstacles, allow visited cells

                # ✅ DYNAMIC OBSTACLE CHECK: Skip cells with dynamic obstacles
                if hasattr(self, 'dynamic_obstacle_positions'):
                    if (neighbor.x, neighbor.y) in self.dynamic_obstacle_positions:
                        continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        return []  # No path found

    def simulate_sensor_detection(self, dynamic_obstacles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Simulate sensor detection of dynamic obstacles"""
        detected = []
        for obs_x, obs_y in dynamic_obstacles:
            distance = np.sqrt((self.position.x - obs_x) ** 2 + (self.position.y - obs_y) ** 2)
            if distance <= self.sensor_range:
                detected.append((obs_x, obs_y))
        return detected

    def select_best_backtrack_point(self) -> Optional[Position]:
        """Select most recent valid backtrack point as in paper"""
        if not self.backtrack_list:
            return None

        # Paper uses "newest point in the backtracking list as the goal"
        # Check from most recent to oldest
        for candidate in reversed(self.backtrack_list):
            # Verify candidate still has unvisited neighbors
            neighbors = self.get_neighbors(candidate)
            has_unvisited = any(self.grid_state[n.y, n.x] == GridState.UNVISITED.value
                                for n in neighbors)
            if has_unvisited:
                return candidate

        return None

    def run_coverage(self, max_steps: int = 1000, dynamic_obstacles: List[Tuple[int, int]] = None) -> Dict:
        """Main coverage algorithm following paper logic"""
        if dynamic_obstacles is None:
            dynamic_obstacles = []
        step = 0
        deadlock_count = 0
        coverage_history = []

        while step < max_steps:
            # 1. Update neural activities
            self.update_neural_activity()

            # 2. Update backtrack list
            self.update_backtrack_list()

            # 3. Detect dynamic obstacles
            detected_obstacles = self.simulate_sensor_detection(dynamic_obstacles)
            for obs_x, obs_y in detected_obstacles:
                self.add_dynamic_obstacle(obs_x, obs_y)

            # 4. Try normal movement first
            next_pos = self.select_next_position_with_priority()

            if next_pos is not None:
                # Normal movement
                self.position = next_pos
                self.path.append(next_pos)
                self.grid_state[next_pos.y, next_pos.x] = GridState.VISITED.value
                self.external_input[next_pos.y, next_pos.x] = 0.0
                self.count_cell_go_through += 1  # ✅ FIXED: Track coverage moves
            elif self.is_deadlock():
                # Deadlock situation - use backtracking
                deadlock_count += 1
                backtrack_point = self.select_best_backtrack_point()

                if backtrack_point is None:
                    break

                # Plan path to backtrack point using Dynamic A*
                path = self.dynamic_a_star(self.position, backtrack_point)
                if path and len(path) > 1:
                    # Move along path
                    for pos in path[1:]:
                        self.position = pos
                        self.path.append(pos)
                        # Mark backtrack path cells as visited if they were unvisited
                        if self.grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                            self.grid_state[pos.y, pos.x] = GridState.VISITED.value
                            self.external_input[pos.y, pos.x] = 0.0
                            self.count_cell_go_through += 1  # ✅ FIXED: Track coverage moves during backtrack
                else:
                    # Cannot reach backtrack point, remove it
                    if backtrack_point in self.backtrack_list:
                        self.backtrack_list.remove(backtrack_point)
            else:
                # No valid moves - check for backtracking opportunity
                total_unvisited = torch.sum(self.grid_state == GridState.UNVISITED.value).item()
                if total_unvisited > 0 and len(self.backtrack_list) > 0:
                    # Force backtracking when stuck but unvisited cells remain
                    deadlock_count += 1
                    backtrack_point = self.select_best_backtrack_point()
                    if backtrack_point:
                        path = self.dynamic_a_star(self.position, backtrack_point)
                        if path and len(path) > 1:
                            for pos in path[1:]:
                                self.position = pos
                                self.path.append(pos)
                                if self.grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                                    self.grid_state[pos.y, pos.x] = GridState.VISITED.value
                                    self.external_input[pos.y, pos.x] = 0.0
                                    self.count_cell_go_through += 1  # ✅ FIXED: Track coverage moves in forced backtrack
                            continue  # Continue coverage
                break

            step += 1

            # Calculate and record coverage rate
            total_cells = self.width * self.height
            obstacle_cells = torch.sum(self.grid_state == GridState.OBSTACLE.value).item()
            visited_cells = torch.sum(self.grid_state == GridState.VISITED.value).item()
            accessible_cells = total_cells - obstacle_cells
            current_coverage = visited_cells / accessible_cells if accessible_cells > 0 else 0
            coverage_history.append(current_coverage)

            # Check if all accessible cells are visited
            total_unvisited = torch.sum(self.grid_state == GridState.UNVISITED.value).item()
            if total_unvisited == 0:
                break

        # Calculate final coverage rate
        final_coverage_rate = coverage_history[-1] if coverage_history else 0
        
        # ✅ FIXED: Calculate overlap rate using correct formula
        explored_cells = torch.sum(self.grid_state == GridState.VISITED.value).item()
        overlap_rate = (self.count_cell_go_through / explored_cells - 1) * 100 if explored_cells > 0 else 0.0

        return {
            'steps': step,
            'coverage_rate': final_coverage_rate,
            'path_length': len(self.path),
            'deadlock_count': deadlock_count,  # Use local variable, not self.deadlock_count
            'coverage_history': coverage_history,
            'overlap_rate': overlap_rate,  # ✅ FIXED: Add overlap rate to results
            'count_cell_go_through': self.count_cell_go_through,  # For debugging
            'explored_cells': explored_cells  # For debugging
        }