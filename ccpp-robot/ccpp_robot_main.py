import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import heapq
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

# Ensure GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
        self.B = 100.0
        self.D = 100.0
        self.E = 1000.0
        self.m = 10.0
        self.r0 = 2.0

        # Initialize grids on GPU
        self.grid_state = torch.zeros((height, width), dtype=torch.int, device=device)
        self.neural_activity = torch.zeros((height, width), dtype=torch.float32, device=device)
        self.external_input = torch.zeros((height, width), dtype=torch.float32, device=device)

        # Robot state
        self.position = Position(0, 0)
        self.backtrack_list = []
        self.path = [self.position]

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
        """Calculate connection weight between two positions"""
        distance = max(abs(pos1.x - pos2.x), abs(pos1.y - pos2.y))  # Chebyshev distance for 8-connectivity
        if 0 < distance <= self.r0:
            return self.m / distance
        return 0.0

    def update_neural_activity(self):
        """Update neural activities using shunting short-memory model"""
        dt = 0.01  # Smaller time step to prevent overflow

        # Create new activity tensor
        new_activity = torch.zeros_like(self.neural_activity)

        for y in range(self.height):
            for x in range(self.width):
                current_pos = Position(x, y)
                current_activity = self.neural_activity[y, x]
                external_input = self.external_input[y, x]

                # Prevent inf values
                if torch.isinf(current_activity) or torch.isnan(current_activity):
                    current_activity = 0.0

                # Calculate neighbor influence
                neighbor_influence = 0.0
                neighbors = self.get_neighbors(current_pos)

                for neighbor in neighbors:
                    neighbor_activity = self.neural_activity[neighbor.y, neighbor.x]
                    if torch.isinf(neighbor_activity) or torch.isnan(neighbor_activity):
                        neighbor_activity = 0.0

                    weight = self.calculate_connection_weight(current_pos, neighbor)
                    if neighbor_activity > 0:
                        neighbor_influence += weight * neighbor_activity

                # Clamp neighbor influence to prevent explosion
                neighbor_influence = min(neighbor_influence, self.B)

                # Shunting equation with clamping
                if external_input >= 0:  # Positive input (unvisited)
                    excitation = min(external_input + neighbor_influence, self.B)
                    delta = (-self.A * current_activity +
                            (self.B - current_activity) * excitation)
                else:  # Negative input (obstacle)
                    inhibition = min(abs(external_input), self.D)
                    delta = (-self.A * current_activity -
                            (self.D + current_activity) * inhibition)

                # Update with clamping
                new_val = current_activity + dt * delta
                new_activity[y, x] = torch.clamp(new_val, 0.0, self.B)

        self.neural_activity = new_activity

    def select_next_position_with_priority(self) -> Optional[Position]:
        """Select next position using neural activity and priority template"""
        current = self.position
        neighbors = self.get_neighbors(current)

        # Get all valid unvisited neighbors with their activities
        candidates = []
        for neighbor in neighbors:
            if self.grid_state[neighbor.y, neighbor.x] == GridState.UNVISITED.value:
                activity = self.neural_activity[neighbor.y, neighbor.x]
                candidates.append((neighbor, activity))

        if not candidates:
            return None

        # Sort by activity (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Apply priority template for ties
        max_activity = candidates[0][1]
        tolerance = 0.001  # Small tolerance for floating point

        # Get all candidates with maximum activity
        top_candidates = [pos for pos, act in candidates if abs(act - max_activity) <= tolerance]

        if len(top_candidates) > 1:
            # Apply priority template: up, down, left, right, then diagonals
            priority_directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dx, dy in priority_directions:
                target = Position(current.x + dx, current.y + dy)
                for candidate in top_candidates:
                    if candidate.x == target.x and candidate.y == target.y:
                        return candidate

        return candidates[0][0]  # Return highest activity

    def is_deadlock(self) -> bool:
        """Check if robot is in deadlock situation according to paper"""
        neighbors = self.get_neighbors(self.position)

        # Check if all neighbors are visited or obstacles
        for neighbor in neighbors:
            state = self.grid_state[neighbor.y, neighbor.x]
            if state == GridState.UNVISITED.value:
                return False

        return True  # All neighbors are visited or obstacles

    def update_backtrack_list(self):
        """Algorithm 1: Update backtracking list exactly as in paper"""
        neighbors = self.get_neighbors(self.position)
        unvisited_count = 0
        all_visited_or_obstacle = True

        # Check neighbor states
        for neighbor in neighbors:
            state = self.grid_state[neighbor.y, neighbor.x]
            if state == GridState.UNVISITED.value:
                unvisited_count += 1
                all_visited_or_obstacle = False
            elif state != GridState.OBSTACLE.value and state != GridState.VISITED.value:
                all_visited_or_obstacle = False
        # Algorithm 1 logic from paper
        if unvisited_count > 0:
            # Add current point to backtracking list if it has unvisited neighbors
            if self.position not in self.backtrack_list:
                self.backtrack_list.append(self.position)

        if all_visited_or_obstacle:
            # Remove current point from backtracking list
            if self.position in self.backtrack_list:
                self.backtrack_list.remove(self.position)

    def select_best_backtrack_point(self) -> Optional[Position]:
        """Select the best backtrack point with accessible unvisited neighbors"""
        if not self.backtrack_list:
            return None

        # Remove invalid backtrack points and find best one
        valid_backtrack = []
        for pos in self.backtrack_list:
            neighbors = self.get_neighbors(pos)
            unvisited_count = 0

            for neighbor in neighbors:
                if self.grid_state[neighbor.y, neighbor.x] == GridState.UNVISITED.value:
                    unvisited_count += 1

            if unvisited_count > 0:
                # Check if we can actually reach this backtrack point
                path = self.dynamic_a_star(self.position, pos)
                if path:
                    valid_backtrack.append((pos, unvisited_count, len(path)))

        if not valid_backtrack:
            self.backtrack_list.clear()
            return None

        # Update backtrack list to only valid points
        self.backtrack_list = [pos for pos, _, _ in valid_backtrack]

        # Select best backtrack point: most unvisited neighbors, shortest path as tiebreaker
        best_pos = max(valid_backtrack, key=lambda x: (x[1], -x[2]))[0]
        return best_pos

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
                if state == GridState.OBSTACLE.value:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        return []  # No path found

    def escape_deadlock(self) -> bool:
        """Algorithm 2: Deadlock detection and escaping"""
        if not self.is_deadlock():
            return True

        self.deadlock_count += 1
        self.grid_state[self.position.y, self.position.x] = GridState.DEADLOCK.value

        # Select best backtrack point
        backtrack_point = self.select_best_backtrack_point()
        if not backtrack_point:
            return False  # No more areas to explore

        # Plan path to backtrack point
        path = self.dynamic_a_star(self.position, backtrack_point)
        if not path:
            return False

        # Move along the path
        for pos in path[1:]:  # Skip current position
            self.position = pos
            self.path.append(pos)
            if self.grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                self.grid_state[pos.y, pos.x] = GridState.VISITED.value
                self.external_input[pos.y, pos.x] = 0.0

        return True

    def simulate_sensor_detection(self, dynamic_obstacles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Simulate sensor detection of dynamic obstacles"""
        detected = []
        for obs_x, obs_y in dynamic_obstacles:
            distance = np.sqrt((self.position.x - obs_x) ** 2 + (self.position.y - obs_y) ** 2)
            if distance <= self.sensor_range:
                detected.append((obs_x, obs_y))
        return detected

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
            elif self.is_deadlock():
                # Deadlock situation - use backtracking
                deadlock_count += 1
                print(f"Deadlock {deadlock_count} at {self.position.x},{self.position.y} (step {step})")

                backtrack_point = self.select_best_backtrack_point()

                if backtrack_point is None:
                    print("No valid backtrack points - coverage complete or stuck")
                    break

                print(f"Backtracking to {backtrack_point.x},{backtrack_point.y}")

                # Plan path to backtrack point
                path = self.dynamic_a_star(self.position, backtrack_point)
                if path and len(path) > 1:
                    print(f"Backtrack path length: {len(path)}")
                    # Move along path
                    for pos in path[1:]:
                        self.position = pos
                        self.path.append(pos)
                        # Don't mark backtrack path as visited unless it was unvisited
                        if self.grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                            self.grid_state[pos.y, pos.x] = GridState.VISITED.value
                            self.external_input[pos.y, pos.x] = 0.0
                else:
                    print(f"Cannot reach backtrack point {backtrack_point.x},{backtrack_point.y}")
                    # Remove unreachable backtrack point
                    if backtrack_point in self.backtrack_list:
                        self.backtrack_list.remove(backtrack_point)
            else:
                # No valid moves and not in deadlock - coverage complete
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

        return {
            'steps': step,
            'coverage_rate': final_coverage_rate,
            'path_length': len(self.path),
            'deadlock_count': deadlock_count,
            'coverage_history': coverage_history
        }

    def visualize(self, save_path: str = None):
        """Visualize the current state and path"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Grid state and path
        grid_np = self.grid_state.cpu().numpy()

        # Create color map
        colors = np.zeros((self.height, self.width, 3))
        colors[grid_np == GridState.UNVISITED.value] = [1, 1, 1]  # White
        colors[grid_np == GridState.VISITED.value] = [0.7, 0.7, 0.7]  # Gray
        colors[grid_np == GridState.OBSTACLE.value] = [0, 0, 0]  # Black
        colors[grid_np == GridState.DEADLOCK.value] = [1, 0, 0]  # Red

        ax1.imshow(colors, origin='lower')

        # Plot path
        if len(self.path) > 1:
            path_x = [p.x for p in self.path]
            path_y = [p.y for p in self.path]
            ax1.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Path')

        # Mark start and current position
        ax1.plot(self.path[0].x, self.path[0].y, 'go', markersize=10, label='Start')
        ax1.plot(self.position.x, self.position.y, 'ro', markersize=10, label='Current')

        # Mark backtrack points
        for bp in self.backtrack_list:
            ax1.plot(bp.x, bp.y, 'bo', markersize=8, label='Backtrack' if bp == self.backtrack_list[0] else '')

        ax1.set_title('Grid State and Robot Path')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Neural activity landscape
        activity_np = self.neural_activity.cpu().numpy()
        im = ax2.imshow(activity_np, origin='lower', cmap='viridis')
        ax2.set_title('Neural Activity Landscape')
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create robot instance
    robot = CCPPRobot(width=30, height=30, sensor_range=2)

    # Add some static obstacles
    obstacles = [
        (5, 5), (5, 6), (5, 7), (5, 8),
        (10, 10), (11, 10), (12, 10),
        (20, 15), (20, 16), (21, 15), (21, 16),
        (15, 5), (16, 5), (17, 5)
    ]
    robot.add_obstacles(obstacles)

    # Add some dynamic obstacles that will appear during coverage
    dynamic_obstacles = [(12, 12), (13, 13)]

    print("Starting coverage...")
    print(f"Environment: {robot.width}x{robot.height}")
    print(f"Total cells: {robot.width * robot.height}")
    print(f"Obstacle cells: {len(obstacles)}")
    print(f"Accessible cells: {robot.width * robot.height - len(obstacles)}")

    start_time = time.time()

    # Run coverage
    results = robot.run_coverage(max_steps=2000, dynamic_obstacles=dynamic_obstacles)

    end_time = time.time()

    # Print detailed results
    print(f"\nCoverage Results:")
    print(f"Steps taken: {results['steps']}")
    print(f"Coverage rate: {results['coverage_rate']:.2%}")
    print(f"Path length: {results['path_length']}")
    print(f"Deadlock count: {results['deadlock_count']}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # Print final grid state analysis
    total_cells = robot.width * robot.height
    visited_cells = torch.sum(robot.grid_state == GridState.VISITED.value).item()
    unvisited_cells = torch.sum(robot.grid_state == GridState.UNVISITED.value).item()
    obstacle_cells = torch.sum(robot.grid_state == GridState.OBSTACLE.value).item()

    print(f"\nGrid Analysis:")
    print(f"Visited cells: {visited_cells}")
    print(f"Unvisited cells: {unvisited_cells}")
    print(f"Obstacle cells: {obstacle_cells}")
    print(f"Total cells: {total_cells}")
    print(f"Backtrack list length: {len(robot.backtrack_list)}")

    # Check if robot can see any unvisited cells
    current_neighbors = robot.get_neighbors(robot.position)
    unvisited_neighbors = [n for n in current_neighbors
                           if robot.grid_state[n.y, n.x] == GridState.UNVISITED.value]
    print(f"Unvisited neighbors from current position: {len(unvisited_neighbors)}")

    # Visualize results
    robot.visualize()

    # Plot coverage progress
    if 'coverage_history' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['coverage_history'])
        plt.title('Coverage Progress Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Coverage Rate')
        plt.grid(True)
        plt.show()

    # Debug: Print some neural activities
    print(f"\nNeural Activity Debug:")
    print(f"Max activity: {torch.max(robot.neural_activity).item():.6f}")
    print(f"Min activity: {torch.min(robot.neural_activity).item():.6f}")
    print(f"Mean activity: {torch.mean(robot.neural_activity).item():.6f}")