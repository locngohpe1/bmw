"""
Helper functions and patches for algorithm comparison
Ensures compatibility between different project implementations
"""

import os
import sys
import math
import numpy as np
import threading
import contextlib
from unittest.mock import Mock


# ========== PYGAME HEADLESS MODE ==========
class HeadlessPygame:
    """Mock pygame for headless execution"""

    def __init__(self):
        self.QUIT = 'QUIT'
        self.KEYDOWN = 'KEYDOWN'
        self.K_SPACE = 'K_SPACE'
        self.K_LEFT = 'K_LEFT'
        self.K_RIGHT = 'K_RIGHT'
        self.K_UP = 'K_UP'
        self.K_DOWN = 'K_DOWN'
        self.K_ESCAPE = 'K_ESCAPE'

    def init(self):
        pass

    def quit(self):
        pass

    class display:
        @staticmethod
        def set_mode(size):
            return Mock()

        @staticmethod
        def set_caption(title):
            pass

        @staticmethod
        def flip():
            pass

    class time:
        @staticmethod
        def Clock():
            return Mock(tick=Mock(return_value=100), get_time=Mock(return_value=100))

        @staticmethod
        def delay(ms):
            pass

    class event:
        @staticmethod
        def get():
            return []  # No events in headless mode

    class draw:
        @staticmethod
        def rect(surface, color, rect):
            pass

        @staticmethod
        def circle(surface, color, pos, radius, width=0):
            pass

        @staticmethod
        def lines(surface, color, closed, points, width=1):
            pass

    class image:
        @staticmethod
        def load(path):
            return Mock(get_size=Mock(return_value=(24, 32)))

        @staticmethod
        def save(surface, path):
            pass

    class transform:
        @staticmethod
        def scale(image, size):
            return Mock(get_size=Mock(return_value=size))

    class font:
        @staticmethod
        def SysFont(name, size):
            return Mock(render=Mock(return_value=Mock()))


def enable_headless_mode():
    """Enable headless mode by mocking pygame"""
    if 'pygame' in sys.modules:
        # Replace pygame with headless version
        sys.modules['pygame'] = HeadlessPygame()
        # Also replace pg alias
        import pygame as pg
        return pg
    else:
        # Add headless pygame to modules
        headless_pg = HeadlessPygame()
        sys.modules['pygame'] = headless_pg
        sys.modules['pg'] = headless_pg
        return headless_pg


# ========== ALGORITHM-SPECIFIC PATCHES ==========

def patch_project_a_for_comparison():
    """Patch Project A for headless comparison"""
    try:
        import main_paper12 as project_a

        # Store original methods
        if not hasattr(project_a.Robot, '_original_run'):
            project_a.Robot._original_run = project_a.Robot.run

        def headless_run(self, max_steps=5000):
            """Headless version of robot.run()"""
            step = 0
            coverage_finish = False

            while step < max_steps:
                step += 1

                # Update dynamic obstacles
                if hasattr(self, 'dynamic_obstacles_manager'):
                    self.dynamic_obstacles_manager.update(0.1)

                # Check finish condition
                if self.logic.state == project_a.Q.FINISH:
                    if not coverage_finish:
                        coverage_finish = True
                        self.retreat()
                        self.charge()
                    break

                # Main algorithm logic
                wp = self.logic.get_wp(self.current_pos)
                if len(wp) == 0:
                    continue

                selected_cell = self.select_from_wp(wp)

                if selected_cell == self.current_pos:
                    self.task()
                else:
                    if self.logic.state == project_a.Q.NORMAL:
                        if not self.check_enough_energy(selected_cell):
                            self.charge_planning()
                            continue
                        self.move_to(selected_cell)
                    elif self.logic.state == project_a.Q.DEADLOCK:
                        path, dist = self.logic.cache_path, self.logic.cache_dist
                        if len(path) > 0:
                            self.follow_path_plan(path, check_energy=True, stop_on_unexpored=True)

            return step

        # Patch the run method
        project_a.Robot.headless_run = headless_run

        return True
    except Exception as e:
        print(f"Warning: Could not patch Project A: {e}")
        return False


def patch_project_d_for_comparison():
    """Patch Project D for headless comparison"""
    try:
        import main_paper3 as project_d

        # Store original methods if needed
        if not hasattr(project_d.CCPPInBWaveEnvironment, '_original_run'):
            project_d.CCPPInBWaveEnvironment._original_run = getattr(
                project_d.CCPPInBWaveEnvironment, 'run_ccpp_with_bwave_environment', None
            )

        def headless_ccpp_run(self, environment, battery_pos, energy_capacity=1000, dynamic_speed=0.1, max_steps=5000):
            """Headless version of CCPP execution"""

            # Initialize CCPP robot
            ROW_COUNT, COL_COUNT = environment.shape
            self.ccpp_robot = project_d.CCPPRobot(width=COL_COUNT, height=ROW_COUNT, sensor_range=2)

            # Convert obstacles
            static_obstacles = self.convert_bwave_to_ccpp_map(environment, COL_COUNT, ROW_COUNT)
            self.ccpp_robot.add_obstacles(static_obstacles)

            # Set initial position
            start_x, start_y = battery_pos[1], battery_pos[0]
            self.ccpp_robot.position = project_d.Position(start_x, start_y)
            self.ccpp_robot.grid_state[start_y, start_x] = project_d.GridState.VISITED.value
            self.ccpp_robot.path = [project_d.Position(start_x, start_y)]

            # Initialize energy
            self.energy_capacity = energy_capacity
            self.current_energy = energy_capacity

            step = 0
            while step < max_steps:
                step += 1

                # Check completion
                import torch
                total_unvisited = torch.sum(self.ccpp_robot.grid_state == project_d.GridState.UNVISITED.value).item()
                if total_unvisited == 0:
                    break

                # Update neural activity
                self.ccpp_robot.update_neural_activity()
                self.ccpp_robot.update_backtrack_list()

                # Get next position
                next_pos = self.ccpp_robot.select_next_position_with_priority()

                if next_pos is not None:
                    # Check energy
                    distance = np.linalg.norm([next_pos.x - self.ccpp_robot.position.x,
                                               next_pos.y - self.ccpp_robot.position.y])

                    if not self.check_energy_for_return(next_pos, battery_pos):
                        self.charge_robot()
                        continue

                    # Move
                    self.ccpp_robot.position = next_pos
                    self.ccpp_robot.path.append(next_pos)
                    self.ccpp_robot.grid_state[next_pos.y, next_pos.x] = project_d.GridState.VISITED.value
                    self.update_energy_system(distance, is_coverage=True)

                elif self.ccpp_robot.is_deadlock():
                    backtrack_point = self.ccpp_robot.select_best_backtrack_point()
                    if backtrack_point:
                        path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, backtrack_point)
                        if path and len(path) > 1:
                            for pos in path[1:]:
                                self.ccpp_robot.position = pos
                                self.ccpp_robot.path.append(pos)
                    else:
                        break

            return step

        # Add headless method
        project_d.CCPPInBWaveEnvironment.headless_ccpp_run = headless_ccpp_run

        return True
    except Exception as e:
        print(f"Warning: Could not patch Project D: {e}")
        return False


def patch_project_c_for_comparison():
    """Patch Project C for headless comparison"""
    try:
        import main_paper4 as project_c

        # Store original methods
        if not hasattr(project_c.Robot, '_original_run'):
            project_c.Robot._original_run = project_c.Robot.run

        def headless_run(self, max_steps=5000):
            """Headless version of Project C robot.run()"""
            step = 0
            loop_count = 0
            coverage_finish = False

            while step < max_steps:
                step += 1
                loop_count += 1

                # Update dynamic obstacles
                if hasattr(self, 'dynamic_obstacles_manager'):
                    self.dynamic_obstacles_manager.update(0.1)

                # Update Project C dynamic maps
                self.update_dynamic_map_b(loop_count)
                if loop_count % self.velocity == 0:
                    self.update_probability_map_and_seen_map_b()

                # Check finish condition
                if self.logic.state == project_c.Q_B.FINISH:
                    if not coverage_finish:
                        coverage_finish = True
                        self.retreat()
                        self.charge()
                    break

                # Execute task first (Project C requirement)
                if self.logic.state != project_c.Q_B.DEADLOCK:
                    self.task()

                # P-Decision Framework
                flag_b = self.detect_dynamic_obs_b(project_c.VISION_SENSOR_RANGE)

                if flag_b:
                    self.logic.set_map(self.seen_map)
                    self.logic.set_prob_map(self.prob_map)
                    max_bid_value, replan_wp = self.logic.get_replan_wp(self.current_pos)
                    wp = [replan_wp] if replan_wp else []

                    # Go-or-wait decision
                    designated_wp = self.logic.boustrophedon_moving(self.current_pos)
                    if wp != designated_wp and self.prob_map[self.current_pos] < project_c.MIN_PROB_THRESHOLD and len(
                            designated_wp) > 0:
                        designated_wp = designated_wp[0]
                        if self.prob_map[designated_wp] > 0:
                            continue  # Wait
                        else:
                            wp = [designated_wp]
                else:
                    wp = self.logic.get_wp(self.current_pos)

                if len(wp) == 0:
                    if self.logic.state == project_c.Q_B.DEADLOCK:
                        selected_cell = None
                    else:
                        continue
                else:
                    selected_cell = self.select_from_wp(wp)

                # Handle movement
                if selected_cell is None and self.logic.state == project_c.Q_B.DEADLOCK:
                    path = self.logic.escape_deadlock_path(self.current_pos)
                    if len(path) == 0:
                        self.logic.state = project_c.Q_B.FINISH
                        continue
                    else:
                        self.move_to(path[0])
                elif selected_cell is not None:
                    if self.logic.state == project_c.Q_B.NORMAL:
                        if not self.check_enough_energy(selected_cell):
                            self.charge_planning()
                            continue
                        self.move_to(selected_cell)

            return step

        # Patch the run method
        project_c.Robot.headless_run = headless_run

        return True
    except Exception as e:
        print(f"Warning: Could not patch Project C: {e}")
        return False


# ========== METRICS CALCULATION HELPERS ==========

def calculate_coverage_metrics(robot, algorithm_type):
    """Calculate standardized coverage metrics for any algorithm"""

    metrics = {
        'coverage_length': 0,
        'overlap_times': 0,
        'extreme_deadlocks': 0,
        'return_count': 0,
        'total_cells_visited': 0,
        'coverage_rate': 0
    }

    try:
        if algorithm_type == 'project_a':
            # Project A metrics
            if hasattr(robot, 'logic') and hasattr(robot.logic, 'weight_map'):
                visited_cells = np.sum(robot.logic.weight_map == 0)
                total_free_cells = np.sum(robot.logic.weight_map >= 0)
                metrics['total_cells_visited'] = visited_cells
                metrics['coverage_rate'] = visited_cells / total_free_cells if total_free_cells > 0 else 0

        elif algorithm_type == 'project_d':
            # Project D metrics (CCPP)
            if hasattr(robot, 'ccpp_robot'):
                import torch
                visited_cells = torch.sum(robot.ccpp_robot.grid_state == 2).item()  # VISITED = 2
                total_cells = robot.ccpp_robot.grid_state.numel()
                obstacle_cells = torch.sum(robot.ccpp_robot.grid_state == 1).item()  # OBSTACLE = 1
                free_cells = total_cells - obstacle_cells
                metrics['total_cells_visited'] = visited_cells
                metrics['coverage_rate'] = visited_cells / free_cells if free_cells > 0 else 0

                # Path-based metrics
                if hasattr(robot.ccpp_robot, 'path'):
                    metrics['coverage_length'] = len(robot.ccpp_robot.path)

        elif algorithm_type == 'project_c':
            # Project C metrics
            if hasattr(robot, 'static_map'):
                visited_cells = np.sum(robot.static_map == 2)
                total_cells = robot.static_map.size
                obstacle_cells = np.sum(robot.static_map == 1)
                free_cells = total_cells - obstacle_cells
                metrics['total_cells_visited'] = visited_cells
                metrics['coverage_rate'] = visited_cells / free_cells if free_cells > 0 else 0

    except Exception as e:
        print(f"Warning: Error calculating metrics for {algorithm_type}: {e}")

    return metrics


def normalize_results(results_list):
    """Normalize results across different algorithms for fair comparison"""

    normalized = []

    for result in results_list:
        if not result.get('success', False):
            normalized.append(result)
            continue

        normalized_result = result.copy()

        # Normalize coverage length by map size
        if 'map_size' in result:
            map_area = result['map_size'][0] * result['map_size'][1]
            normalized_result['coverage_length_normalized'] = result['coverage_length'] / map_area

        # Calculate efficiency metrics
        if result['coverage_length'] > 0:
            normalized_result['efficiency_score'] = (
                    result.get('total_cells_visited', 0) / result['coverage_length']
            )

        # Calculate overlap rate as percentage
        if result.get('total_cells_visited', 0) > 0:
            normalized_result['overlap_rate'] = (
                    result['overlap_times'] / result['total_cells_visited'] * 100
            )

        normalized.append(normalized_result)

    return normalized


# ========== FILE I/O HELPERS ==========

def ensure_project_imports():
    """Ensure all project files can be imported"""

    required_files = [
        'main_paper12.py',
        'main_paper3.py',
        'main_paper4.py',
        'logic.py',
        'grid_map.py',
        'dynamic_obstacles_manager.py'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all project files are in the same directory")
        return False

    return True


def create_test_map(size=(20, 20), obstacle_density=0.2, output_path="test_maps"):
    """Create a test map for algorithm comparison"""

    os.makedirs(output_path, exist_ok=True)

    rows, cols = size
    map_data = np.zeros((rows, cols), dtype=int)

    # Add random obstacles
    num_obstacles = int(rows * cols * obstacle_density)
    obstacle_positions = np.random.choice(rows * cols, num_obstacles, replace=False)

    for pos in obstacle_positions:
        row, col = pos // cols, pos % cols
        map_data[row, col] = 1

    # Ensure start position is free
    map_data[0, 0] = 0

    # Save map
    map_file = os.path.join(output_path, f"test_map_{rows}x{cols}_obs{int(obstacle_density * 100)}.txt")

    with open(map_file, 'w') as f:
        f.write(f"{cols} {rows}\n")
        for row in map_data:
            f.write(" ".join(map(str, row)) + "\n")

    print(f"📋 Created test map: {map_file}")
    return map_file


# ========== SETUP FUNCTION ==========

def setup_comparison_environment():
    """Setup environment for algorithm comparison"""

    print("🔧 Setting up comparison environment...")

    # Enable headless mode
    enable_headless_mode()

    # Check file dependencies
    if not ensure_project_imports():
        return False

    # Apply patches
    success_a = patch_project_a_for_comparison()
    success_d = patch_project_d_for_comparison()
    success_c = patch_project_c_for_comparison()

    if success_a:
        print("✅ Project A patched for comparison")
    if success_d:
        print("✅ Project D patched for comparison")
    if success_c:
        print("✅ Project C patched for comparison")

    return True

if __name__ == "__main__":
    # Test setup (only when run directly, not when imported)
    if setup_comparison_environment():
        print("✅ Comparison environment ready!")

        # Create test maps
        test_maps = [
            create_test_map((15, 15), 0.1),
            create_test_map((20, 20), 0.2),
            create_test_map((25, 25), 0.15)
        ]

        print(f"📋 Created {len(test_maps)} test maps for comparison")
    else:
        print("❌ Failed to setup comparison environment")