import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import heapq
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import copy

# Import from main robot implementation
from ccpp_robot_main import CCPPRobot, GridState, Position

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiRobotCCPP:
    def __init__(self, width: int, height: int, num_robots: int, sensor_range: int = 2):
        sself.width = width
        self.height = height
        self.num_robots = num_robots
        self.sensor_range = sensor_range

        # Shared grid state for all robots
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

            # Each robot has individual neural activity but shares grid state
            # This follows paper assumption: robots share position info and treat others as obstacles
            robot.grid_state = self.shared_grid_state
            robot.external_input = self.shared_external_input
            # robot.neural_activity remains individual (not shared)

            self.robots.append(robot)

        self.initialize_shared_environment()
        self.communication_range = 8  # Communication range for robot coordination

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

    def get_nearby_robots(self, robot_id: int) -> List[int]:
        """Get robots within communication range"""
        current_robot = self.robots[robot_id]
        nearby = []

        for i, other_robot in enumerate(self.robots):
            if i != robot_id:
                distance = np.sqrt((current_robot.position.x - other_robot.position.x) ** 2 +
                                   (current_robot.position.y - other_robot.position.y) ** 2)
                if distance <= self.communication_range:
                    nearby.append(i)

        return nearby

    def market_based_bidding(self, deadlock_robot_id: int) -> Optional[Position]:
        """Algorithm 3: Market-based bidding process exactly as in paper"""
        deadlock_robot = self.robots[deadlock_robot_id]

        if not deadlock_robot.backtrack_list:
            return None

        # Test each candidate point starting from most recent (as in paper)
        for candidate_point in reversed(deadlock_robot.backtrack_list):

            # Compute Euclidean distances (tender prices) for all robots
            tender_prices = {}
            for i, robot in enumerate(self.robots):
                distance = np.sqrt((robot.position.x - candidate_point.x) ** 2 +
                                   (robot.position.y - candidate_point.y) ** 2)
                tender_prices[i] = distance

            # Find minimum tender price
            min_price = min(tender_prices.values())
            deadlock_price = tender_prices[deadlock_robot_id]

            # Check two conditions from paper:
            # (i) close to the deadlock robot - robot wins bid if has minimum price
            # (ii) far away from other robots - ensure no conflict
            if deadlock_price == min_price:
                # Additional check: candidate should be far from other robots
                conflict_free = True
                for i, robot in enumerate(self.robots):
                    if i != deadlock_robot_id:
                        if tender_prices[i] < self.communication_range:  # Too close to other robot
                            conflict_free = False
                            break

                if conflict_free or len([p for p in tender_prices.values() if p == min_price]) == 1:
                    return candidate_point

        # If no point satisfies all conditions, return most recent point
        return deadlock_robot.backtrack_list[-1] if deadlock_robot.backtrack_list else None

    def avoid_robot_collision(self, robot_id: int, next_pos: Position) -> bool:
        """Check if next position conflicts with other robots"""
        for i, other_robot in enumerate(self.robots):
            if i != robot_id:
                # Check current position conflict
                if next_pos == other_robot.position:
                    return False

                # Check if robots are swapping positions
                if (next_pos == other_robot.position and
                        self.robots[robot_id].position == other_robot.position):
                    return False

        return True

    def coordinate_robot_movement(self, robot_id: int) -> Optional[Position]:
        """Coordinate movement to avoid conflicts between robots"""
        robot = self.robots[robot_id]

        # Get candidate positions from robot's neural activity
        neighbors = robot.get_neighbors(robot.position)
        candidates = []

        for neighbor in neighbors:
            if self.shared_grid_state[neighbor.y, neighbor.x] == GridState.UNVISITED.value:
                # Check collision with other robots
                collision_free = True
                for other_id, other_robot in enumerate(self.robots):
                    if other_id != robot_id:
                        # Avoid current positions and predicted next positions
                        if (neighbor.x == other_robot.position.x and
                            neighbor.y == other_robot.position.y):
                            collision_free = False
                            break

                        # Avoid swapping positions
                        if (neighbor == other_robot.position and
                            robot.position == other_robot.position):
                            collision_free = False
                            break

                if collision_free:
                    activity = robot.neural_activity[neighbor.y, neighbor.x]
                    candidates.append((neighbor, activity))

        if not candidates:
            return None

        # Apply priority template to collision-free candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        max_activity = candidates[0][1]
        tolerance = 0.001

        # Get top activity candidates
        top_candidates = [pos for pos, act in candidates if abs(act - max_activity) <= tolerance]

        if len(top_candidates) > 1:
            # Apply priority template
            current = robot.position
            priority_directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dx, dy in priority_directions:
                target = Position(current.x + dx, current.y + dy)
                for candidate in top_candidates:
                    if candidate.x == target.x and candidate.y == target.y:
                        return candidate

        return candidates[0][0]  # Return highest activity collision-free position

    def run_multi_robot_coverage(self, max_steps: int = 2000,
                                 dynamic_obstacles: List[Tuple[int, int]] = None) -> Dict:
        """Main multi-robot coverage algorithm following paper"""
        if dynamic_obstacles is None:
            dynamic_obstacles = []

        step = 0
        coverage_history = []
        robot_statistics = {i: {'steps': 0, 'deadlocks': 0, 'path_length': 0}
                            for i in range(self.num_robots)}

        while step < max_steps:
            robots_moved = False

            # Process each robot
            for robot_id, robot in enumerate(self.robots):

                # 1. Update robot's individual neural activity
                robot.update_neural_activity()

                # 2. Communicate with nearby robots and treat them as obstacles
                nearby_robots = self.get_nearby_robots(robot_id)
                for other_id in nearby_robots:
                    other_robot = self.robots[other_id]
                    # Nearby robots are considered in collision avoidance
                    pass

                # 3. Update backtrack list
                robot.update_backtrack_list()

                # 4. Detect dynamic obstacles
                detected_obstacles = robot.simulate_sensor_detection(dynamic_obstacles)
                for obs_x, obs_y in detected_obstacles:
                    self.shared_grid_state[obs_y, obs_x] = GridState.OBSTACLE.value
                    self.shared_external_input[obs_y, obs_x] = -1000.0

                # 5. Try normal movement first (avoiding other robots)
                next_pos = self.coordinate_robot_movement(robot_id)

                if next_pos is not None:
                    # Normal movement
                    robot.position = next_pos
                    robot.path.append(next_pos)
                    self.shared_grid_state[next_pos.y, next_pos.x] = GridState.VISITED.value
                    self.shared_external_input[next_pos.y, next_pos.x] = 0.0
                    robots_moved = True
                    robot_statistics[robot_id]['path_length'] += 1

                elif robot.is_deadlock():
                    # 6. Deadlock situation - use market-based bidding
                    robot_statistics[robot_id]['deadlocks'] += 1

                    # Market-based bidding for backtrack point selection
                    backtrack_point = self.market_based_bidding(robot_id)

                    if backtrack_point:
                        # Plan collision-free path to backtrack point
                        path = self.plan_collision_free_path(robot_id, backtrack_point)

                        if path and len(path) > 1:
                            # Move along path
                            for pos in path[1:]:  # Skip current position
                                robot.position = pos
                                robot.path.append(pos)
                                if self.shared_grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                                    self.shared_grid_state[pos.y, pos.x] = GridState.VISITED.value
                                    self.shared_external_input[pos.y, pos.x] = 0.0

                            robots_moved = True
                            robot_statistics[robot_id]['path_length'] += len(path) - 1

                robot_statistics[robot_id]['steps'] += 1

            # Calculate coverage rate
            total_cells = self.width * self.height
            obstacle_cells = torch.sum(self.shared_grid_state == GridState.OBSTACLE.value).item()
            visited_cells = torch.sum(self.shared_grid_state == GridState.VISITED.value).item()
            accessible_cells = total_cells - obstacle_cells
            coverage_rate = visited_cells / accessible_cells if accessible_cells > 0 else 0

            coverage_history.append(coverage_rate)

            # Check termination conditions
            if coverage_rate >= 0.98 or not robots_moved:
                break

            step += 1

        return {
            'total_steps': step,
            'coverage_rate': coverage_history[-1] if coverage_history else 0,
            'coverage_history': coverage_history,
            'robot_statistics': robot_statistics,
            'total_path_length': sum(stats['path_length'] for stats in robot_statistics.values()),
            'total_deadlocks': sum(stats['deadlocks'] for stats in robot_statistics.values())
        }

    def plan_collision_free_path(self, robot_id: int, goal: Position) -> List[Position]:
        """Plan path avoiding other robots using modified A*"""
        robot = self.robots[robot_id]

        def heuristic(pos: Position) -> float:
            return abs(pos.x - goal.x) + abs(pos.y - goal.y)

        def is_occupied_by_robot(pos: Position) -> bool:
            for i, other_robot in enumerate(self.robots):
                if i != robot_id and other_robot.position == pos:
                    return True
            return False

        # Use counter to ensure unique ordering for heapq
        counter = 0
        open_set = [(0, counter, robot.position)]
        came_from = {}
        g_score = {robot.position: 0}

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(robot.position)
                return path[::-1]

            for neighbor in robot.get_neighbors(current):
                # Skip obstacles
                if self.shared_grid_state[neighbor.y, neighbor.x] == GridState.OBSTACLE.value:
                    continue

                # Skip positions occupied by other robots (with some tolerance)
                if is_occupied_by_robot(neighbor) and neighbor != goal:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return []  # No path found

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

        # Plot 2: Neural activity landscape
        activity_np = self.robots[0].neural_activity.cpu().numpy()
        im = ax2.imshow(activity_np, origin='lower', cmap='viridis')
        ax2.set_title('Shared Neural Activity Landscape')
        plt.colorbar(im, ax=ax2)

        # Mark robot positions on activity plot
        for i, robot in enumerate(self.robots):
            color = color_map.get(robot.color, 'black')
            ax2.plot(robot.position.x, robot.position.y, 'o',
                     color=color, markersize=10, markeredgecolor='white', markeredgewidth=1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create multi-robot system
    multi_robot = MultiRobotCCPP(width=25, height=25, num_robots=4, sensor_range=2)

    # Add obstacles
    obstacles = [
        (5, 5), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8),
        (12, 10), (12, 11), (12, 12), (13, 12), (14, 12),
        (18, 15), (18, 16), (19, 15), (19, 16), (20, 15), (20, 16),
        (8, 20), (9, 20), (10, 20), (11, 20)
    ]
    multi_robot.add_shared_obstacles(obstacles)

    # Dynamic obstacles
    dynamic_obstacles = [(15, 8), (16, 9)]

    print("Starting multi-robot coverage...")
    start_time = time.time()

    # Run multi-robot coverage
    results = multi_robot.run_multi_robot_coverage(max_steps=1500,
                                                   dynamic_obstacles=dynamic_obstacles)

    end_time = time.time()

    # Print results
    print(f"\nMulti-Robot Coverage Results:")
    print(f"Total steps: {results['total_steps']}")
    print(f"Coverage rate: {results['coverage_rate']:.2%}")
    print(f"Total path length: {results['total_path_length']}")
    print(f"Total deadlocks: {results['total_deadlocks']}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    print(f"\nIndividual Robot Statistics:")
    for robot_id, stats in results['robot_statistics'].items():
        print(f"Robot {robot_id + 1}: Steps={stats['steps']}, "
              f"Path Length={stats['path_length']}, Deadlocks={stats['deadlocks']}")

    # Visualize results
    multi_robot.visualize_multi_robot()

    # Plot coverage progress
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results['coverage_history'])
    plt.title('Multi-Robot Coverage Progress')
    plt.xlabel('Steps')
    plt.ylabel('Coverage Rate')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    robot_ids = list(results['robot_statistics'].keys())
    path_lengths = [results['robot_statistics'][rid]['path_length'] for rid in robot_ids]
    deadlock_counts = [results['robot_statistics'][rid]['deadlocks'] for rid in robot_ids]

    x = np.arange(len(robot_ids))
    width = 0.35

    plt.bar(x - width / 2, path_lengths, width, label='Path Length', alpha=0.8)
    plt.bar(x + width / 2, deadlock_counts, width, label='Deadlocks', alpha=0.8)

    plt.xlabel('Robot ID')
    plt.ylabel('Count')
    plt.title('Robot Performance Comparison')
    plt.xticks(x, [f'Robot {i + 1}' for i in robot_ids])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()