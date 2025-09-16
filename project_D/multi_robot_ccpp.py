import torch
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

# Import from main robot implementation
from ccpp_robot_main import CCPPRobot, GridState, Position

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiRobotCCPP:
    def __init__(self, width: int, height: int, num_robots: int, sensor_range: int = 2):
        self.width = width
        self.height = height
        self.num_robots = num_robots
        self.sensor_range = sensor_range

        # Shared grid state for all robots (per paper assumption)
        self.shared_grid_state = torch.zeros((height, width), dtype=torch.int, device=device)
        self.shared_external_input = torch.zeros((height, width), dtype=torch.float32, device=device)

        # Initialize robots
        self.robots = []
        self.robot_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        # Create robots with different starting positions
        start_positions = self._generate_start_positions(num_robots)
        for i in range(num_robots):
            robot = CCPPRobot(width, height, sensor_range)
            robot.robot_id = i
            robot.position = start_positions[i]
            robot.path = [robot.position]
            robot.color = self.robot_colors[i % len(self.robot_colors)]

            # CRITICAL FIX: Share both grid state AND external input (paper assumption)
            robot.grid_state = self.shared_grid_state
            robot.external_input = self.shared_external_input
            # Each robot maintains individual neural_activity (paper design)
            
            # ✅ FIXED: Initialize overlap tracking for each robot
            robot.count_cell_go_through = 1  # Starting position counts

            self.robots.append(robot)

        self.initialize_shared_environment()

    def _generate_start_positions(self, num_robots: int) -> List[Position]:
        """Generate well-distributed starting positions for robots"""
        positions = []

        if num_robots == 1:
            positions.append(Position(0, 0))
        elif num_robots == 2:
            positions.extend([Position(0, 0), Position(self.width - 1, self.height - 1)])
        elif num_robots == 4:
            positions.extend([
                Position(0, 0),
                Position(self.width - 1, 0),
                Position(0, self.height - 1),
                Position(self.width - 1, self.height - 1)
            ])
        else:
            # Distribute robots around the perimeter
            for i in range(num_robots):
                if i < num_robots // 2:
                    x = i * (self.width // (num_robots // 2 + 1))
                    y = 0
                else:
                    x = (i - num_robots // 2) * (self.width // (num_robots // 2 + 1))
                    y = self.height - 1
                positions.append(Position(min(x, self.width - 1), y))

        return positions

    def initialize_shared_environment(self):
        """Initialize shared environment for all robots"""
        self.shared_grid_state.fill_(GridState.UNVISITED.value)
        self.shared_external_input.fill_(1000.0)  # E value for unvisited

        # Mark robot starting positions as visited
        for robot in self.robots:
            self.shared_grid_state[robot.position.y, robot.position.x] = GridState.VISITED.value
            self.shared_external_input[robot.position.y, robot.position.x] = 0.0

    def add_shared_obstacles(self, obstacles: List[Tuple[int, int]]):
        """Add obstacles to shared environment"""
        for x, y in obstacles:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.shared_grid_state[y, x] = GridState.OBSTACLE.value
                self.shared_external_input[y, x] = -1000.0

    def treat_other_robots_as_obstacles(self, current_robot_id: int):
        """Paper assumption: treat other robots as obstacles"""
        # Mark other robots as temporary obstacles for current robot's pathfinding
        for robot_id, robot in enumerate(self.robots):
            if robot_id != current_robot_id:
                pos = robot.position
                # Store original state before marking as obstacle
                if hasattr(self, '_original_states'):
                    self._original_states[pos] = self.shared_grid_state[pos.y, pos.x].item()
                else:
                    self._original_states = {pos: self.shared_grid_state[pos.y, pos.x].item()}

                # Temporarily mark as obstacle only if not permanent obstacle
                if self.shared_external_input[pos.y, pos.x] != -1000.0:
                    self.shared_grid_state[pos.y, pos.x] = GridState.OBSTACLE.value

    def restore_robot_positions(self, current_robot_id: int):
        """Restore robot positions after pathfinding"""
        if hasattr(self, '_original_states'):
            for pos, original_state in self._original_states.items():
                # Restore original state
                self.shared_grid_state[pos.y, pos.x] = original_state
            # Clear stored states
            self._original_states = {}

    def market_based_bidding(self, deadlock_robot_id: int) -> Optional[Position]:
        """Algorithm 3: Market-based bidding process EXACTLY as in paper"""
        deadlock_robot = self.robots[deadlock_robot_id]

        if not deadlock_robot.backtrack_list:
            return None

        # Paper Algorithm 3: Test each candidate point p starting from most recent
        for candidate_point in reversed(deadlock_robot.backtrack_list):
            # Verify candidate still has unvisited neighbors
            neighbors = deadlock_robot.get_neighbors(candidate_point)
            has_unvisited = any(self.shared_grid_state[n.y, n.x] == GridState.UNVISITED.value
                                for n in neighbors)

            if not has_unvisited:
                continue

            # Every robot computes its Euclidean distance to p as tender price
            tender_prices = {}
            for robot_id, robot in enumerate(self.robots):
                distance = np.sqrt((robot.position.x - candidate_point.x) ** 2 +
                                   (robot.position.y - candidate_point.y) ** 2)
                tender_prices[robot_id] = distance

            # Find minimum tender price among all robots
            min_price = min(tender_prices.values())
            deadlock_price = tender_prices[deadlock_robot_id]

            # Paper condition: "If the tender price of the bidder Rdl is lower than any other robots"
            if deadlock_price <= min_price:
                # Additional paper condition: "Gbt should be away from the other robots so
                # that it will not be covered by the others in a short time"
                min_distance_to_others = float('inf')
                for robot_id, robot in enumerate(self.robots):
                    if robot_id != deadlock_robot_id:
                        distance = tender_prices[robot_id]
                        min_distance_to_others = min(min_distance_to_others, distance)

                # Use a reasonable threshold for "short time" coverage
                short_time_threshold = 5.0  # Euclidean distance

                if min_distance_to_others > short_time_threshold:
                    return candidate_point

        # Paper: "if all the points in the BTlist have already been considered and
        # none of them satisfies the above condition, the most recent point in the BTlist is chosen as Gbt"
        if deadlock_robot.backtrack_list:
            return deadlock_robot.backtrack_list[-1]

        return None

    def run_multi_robot_coverage(self, max_steps: int = 2000,
                                 dynamic_obstacles: List[Tuple[int, int]] = None) -> Dict:
        """Main multi-robot coverage algorithm following paper EXACTLY"""
        if dynamic_obstacles is None:
            dynamic_obstacles = []

        step = 0
        coverage_history = []
        robot_statistics = {i: {'steps': 0, 'deadlocks': 0, 'path_length': 0}
                            for i in range(self.num_robots)}

        while step < max_steps:
            robots_moved = False

            # Process each robot independently (paper approach)
            for robot_id, robot in enumerate(self.robots):

                # 1. Treat other robots as obstacles (paper assumption)
                self.treat_other_robots_as_obstacles(robot_id)

                # 2. Update robot's individual neural activity
                robot.update_neural_activity()

                # 3. Update backtrack list (Algorithm 1)
                robot.update_backtrack_list()

                # 4. Detect dynamic obstacles
                detected_obstacles = robot.simulate_sensor_detection(dynamic_obstacles)
                for obs_x, obs_y in detected_obstacles:
                    self.shared_grid_state[obs_y, obs_x] = GridState.OBSTACLE.value
                    self.shared_external_input[obs_y, obs_x] = -1000.0

                # 5. Try normal movement first (using priority template)
                next_pos = robot.select_next_position_with_priority()

                if next_pos is not None:
                    # Normal movement
                    robot.position = next_pos
                    robot.path.append(next_pos)
                    self.shared_grid_state[next_pos.y, next_pos.x] = GridState.VISITED.value
                    self.shared_external_input[next_pos.y, next_pos.x] = 0.0
                    robot.count_cell_go_through += 1  # ✅ FIXED: Track coverage moves
                    robots_moved = True
                    robot_statistics[robot_id]['path_length'] += 1

                elif robot.is_deadlock():
                    # 6. Deadlock situation - use market-based bidding (Algorithm 3)
                    robot_statistics[robot_id]['deadlocks'] += 1

                    # CRITICAL: Market-based bidding exactly as in paper
                    backtrack_point = self.market_based_bidding(robot_id)

                    if backtrack_point:
                        # Plan path to backtrack point using Dynamic A*
                        path = robot.dynamic_a_star(robot.position, backtrack_point)

                        if path and len(path) > 1:
                            # Move along path
                            for pos in path[1:]:  # Skip current position
                                robot.position = pos
                                robot.path.append(pos)

                                # Mark path cells as visited if they were unvisited
                                if self.shared_grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                                    self.shared_grid_state[pos.y, pos.x] = GridState.VISITED.value
                                    self.shared_external_input[pos.y, pos.x] = 0.0

                            robots_moved = True
                            robot_statistics[robot_id]['path_length'] += len(path) - 1
                        else:
                            # Cannot reach backtrack point, remove it
                            if backtrack_point in robot.backtrack_list:
                                robot.backtrack_list.remove(backtrack_point)
                else:
                    # No valid moves and not deadlock - check for remaining work
                    total_unvisited = torch.sum(self.shared_grid_state == GridState.UNVISITED.value).item()
                    if total_unvisited > 0 and robot.backtrack_list:
                        # Try backtracking anyway
                        robot_statistics[robot_id]['deadlocks'] += 1
                        backtrack_point = robot.select_best_backtrack_point()

                        if backtrack_point:
                            path = robot.dynamic_a_star(robot.position, backtrack_point)
                            if path and len(path) > 1:
                                for pos in path[1:]:
                                    robot.position = pos
                                    robot.path.append(pos)
                                    if self.shared_grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                                        self.shared_grid_state[pos.y, pos.x] = GridState.VISITED.value
                                        self.shared_external_input[pos.y, pos.x] = 0.0
                                        robot.count_cell_go_through += 1  # ✅ FIXED: Track coverage moves during backtrack
                                robots_moved = True
                                robot_statistics[robot_id]['path_length'] += len(path) - 1

                # 7. Restore robot positions for next robot
                self.restore_robot_positions(robot_id)

                robot_statistics[robot_id]['steps'] += 1

            # Calculate coverage rate
            total_cells = self.width * self.height
            obstacle_cells = torch.sum(self.shared_grid_state == GridState.OBSTACLE.value).item()
            visited_cells = torch.sum(self.shared_grid_state == GridState.VISITED.value).item()
            accessible_cells = total_cells - obstacle_cells
            coverage_rate = visited_cells / accessible_cells if accessible_cells > 0 else 0

            coverage_history.append(coverage_rate)

            # Check termination conditions - paper doesn't specify early termination at 98%
            total_unvisited = torch.sum(self.shared_grid_state == GridState.UNVISITED.value).item()
            if total_unvisited == 0 or not robots_moved:
                break

            step += 1

        # ✅ FIXED: Calculate overlap rate for multi-robot system
        total_coverage_moves = sum(robot.count_cell_go_through for robot in self.robots)
        explored_cells = torch.sum(self.shared_grid_state == GridState.VISITED.value).item()
        overlap_rate = (total_coverage_moves / explored_cells - 1) * 100 if explored_cells > 0 else 0.0

        return {
            'total_steps': step,
            'coverage_rate': coverage_history[-1] if coverage_history else 0,
            'coverage_history': coverage_history,
            'robot_statistics': robot_statistics,
            'total_path_length': sum(stats['path_length'] for stats in robot_statistics.values()),
            'total_deadlocks': sum(stats['deadlocks'] for stats in robot_statistics.values()),
            'overlap_rate': overlap_rate,  # ✅ FIXED: Add overlap rate for multi-robot
            'total_coverage_moves': total_coverage_moves,  # For debugging
            'explored_cells': explored_cells  # For debugging
        }

    def visualize_multi_robot(self, save_path: str = None):
        """Visualize multi-robot coverage state"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Grid state with robot paths
        grid_np = self.shared_grid_state.cpu().numpy()

        # Create base color map
        colors = np.zeros((self.height, self.width, 3))
        colors[grid_np == GridState.UNVISITED.value] = [1, 1, 1]  # White
        colors[grid_np == GridState.VISITED.value] = [0.9, 0.9, 0.9]  # Light gray
        colors[grid_np == GridState.OBSTACLE.value] = [0, 0, 0]  # Black
        colors[grid_np == GridState.DEADLOCK.value] = [0.8, 0.2, 0.2]  # Dark red

        ax1.imshow(colors, origin='lower')

        # Plot robot paths
        color_map = {'red': 'r', 'blue': 'b', 'green': 'g', 'orange': 'orange',
                     'purple': 'purple', 'brown': 'brown', 'pink': 'pink', 'gray': 'gray'}

        for i, robot in enumerate(self.robots):
            if len(robot.path) > 1:
                path_x = [p.x for p in robot.path]
                path_y = [p.y for p in robot.path]
                color = color_map.get(robot.color, 'black')
                ax1.plot(path_x, path_y, color=color, linewidth=2, alpha=0.7,
                         label=f'Robot {i + 1}')

            # Mark current position
            ax1.plot(robot.position.x, robot.position.y, 'o',
                     color=color_map.get(robot.color, 'black'), markersize=12,
                     markeredgecolor='black', markeredgewidth=2)

            # Mark backtrack points
            for bp in robot.backtrack_list:
                ax1.plot(bp.x, bp.y, 's', color=color_map.get(robot.color, 'black'),
                         markersize=6, alpha=0.6)

        ax1.set_title('Multi-Robot Coverage Paths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Coverage rate comparison
        ax2.bar(range(len(self.robots)),
                [len(robot.path) for robot in self.robots],
                color=[color_map.get(robot.color, 'black') for robot in self.robots],
                alpha=0.7)
        ax2.set_xlabel('Robot ID')
        ax2.set_ylabel('Path Length')
        ax2.set_title('Robot Workload Distribution')
        ax2.set_xticks(range(len(self.robots)))
        ax2.set_xticklabels([f'R{i + 1}' for i in range(len(self.robots))])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()