import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import time
import argparse
import sys
import os
import math
from datetime import datetime
import pandas as pd
import main_paper12 as project_a
import main_paper3 as project_d
import main_paper4 as project_c
from grid_map import Grid_Map
from dynamic_obstacles_manager import DynamicObstaclesManager

class AlgorithmComparison:
    def __init__(self):
        self.results = {}
        self.test_maps = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def setup_shared_environment(self, map_file, energy_capacity=1000, dynamic_speed=0.1):
        """Setup environment ONCE for all 3 algorithms"""
        print(f"🏗️ Setting up SHARED environment: {map_file}")

        # Setup UI and environment (shared for all algorithms)
        ui = Grid_Map()
        ui.read_map(map_file)

        print("🎮 MAP EDITOR - Configure environment for ALL algorithms:")
        print("- Left click: Static obstacles")
        print("- Shift + Left click: Dynamic obstacles")
        print("- Right click: Charging station")
        print("- Alt+F4 or Close window: Save and continue to algorithm comparison")
        print("- Space: Pause/Resume during editing")

        # Interactive editing phase
        environment, battery_pos = ui.edit_map()

        # Save shared configuration
        shared_config = {
            'map_file': map_file,
            'environment': environment.copy(),
            'battery_pos': battery_pos,
            'energy_capacity': energy_capacity,
            'dynamic_speed': dynamic_speed,
            'dynamic_obstacles': getattr(ui, 'dynamic_obstacles', []).copy(),
            'row_count': len(environment),
            'col_count': len(environment[0])
        }

        # Save to file for reuse
        config_file = f"shared_environment_{int(time.time())}.json"
        with open(config_file, 'w') as f:
            json.dump({
                'map_file': map_file,
                'battery_pos': battery_pos,
                'energy_capacity': energy_capacity,
                'dynamic_speed': dynamic_speed,
                'dynamic_obstacles_count': len(getattr(ui, 'dynamic_obstacles', [])),
                'environment_shape': environment.shape,
                'timestamp': time.time()
            }, f, indent=2, default=str)

        print(f"✅ Shared environment configured!")
        print(f"📋 Map size: {environment.shape}")
        print(f"🔋 Charging station: {battery_pos}")
        print(f"🚧 Static obstacles: {np.sum(environment == 1)}")
        print(f"🚶 Dynamic obstacles: {len(getattr(ui, 'dynamic_obstacles', []))}")
        print(f"💾 Configuration saved: {config_file}")

        return shared_config

    def run_project_a(self, shared_config):
        """Run Project A (BWave Framework) using shared environment"""
        print("\n" + "=" * 60)
        print("🚀 RUNNING PROJECT A - BWave Framework")
        print("=" * 60)

        try:
            # Extract from shared config
            environment = shared_config['environment']
            battery_pos = shared_config['battery_pos']
            energy_capacity = shared_config['energy_capacity']
            dynamic_speed = shared_config['dynamic_speed']

            # Setup UI from shared config
            ui = Grid_Map()
            ui.map = environment.copy()
            ui.row_count = shared_config['row_count']
            ui.col_count = shared_config['col_count']
            ui.battery_pos = battery_pos
            ui.dynamic_obstacles = shared_config['dynamic_obstacles'].copy()

            # Initialize robot
            robot = project_a.Robot(battery_pos, ui.row_count, ui.col_count)
            robot.set_map(environment)

            # Initialize dynamic obstacles manager
            robot.dynamic_obstacles_manager = DynamicObstaclesManager(ui, num_obstacles=0, speed_factor=dynamic_speed)
            if ui.dynamic_obstacles:
                robot.dynamic_obstacles_manager.initialize_obstacles()

            # Store start time
            start_time = time.time()

            # Run headless version
            results = self._run_project_a_headless(robot, robot.dynamic_obstacles_manager, ui, energy_capacity)
            execution_time = time.time() - start_time

            results.update({
                'algorithm': 'Project A - BWave Framework',
                'execution_time': execution_time,
                'success': True
            })

            return results

        except Exception as e:
            print(f"❌ Project A failed: {e}")
            return {
                'algorithm': 'Project A - BWave Framework',
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'total_path_length': 0,
                'overlap_rate': 0,
                'extreme_deadlocks': 0,
                'return_count': 0
            }

    def run_project_d(self, shared_config):
        """Run Project D (Neural Dynamics CCPP) using shared environment"""
        print("\n" + "=" * 60)
        print("🧠 RUNNING PROJECT D - Neural Dynamics CCPP")
        print("=" * 60)

        try:
            # Extract from shared config
            environment = shared_config['environment']
            battery_pos = shared_config['battery_pos']
            energy_capacity = shared_config['energy_capacity']
            dynamic_speed = shared_config['dynamic_speed']

            # Create CCPP environment
            ccpp_env = project_d.CCPPInBWaveEnvironment()

            # Setup environment from shared config
            ccpp_env.ui = Grid_Map()
            ccpp_env.ui.map = environment.copy()
            ccpp_env.ui.row_count = shared_config['row_count']
            ccpp_env.ui.col_count = shared_config['col_count']
            ccpp_env.ui.battery_pos = battery_pos
            ccpp_env.ui.dynamic_obstacles = shared_config['dynamic_obstacles'].copy()

            start_time = time.time()

            # Run headless version
            results = self._run_project_d_headless(ccpp_env, environment, battery_pos, energy_capacity, dynamic_speed)

            execution_time = time.time() - start_time

            results.update({
                'algorithm': 'Project D - Neural Dynamics CCPP',
                'execution_time': execution_time,
                'success': True
            })

            return results

        except Exception as e:
            print(f"❌ Project D failed: {e}")
            return {
                'algorithm': 'Project D - Neural Dynamics CCPP',
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'total_path_length': 0,
                'overlap_rate': 0,
                'extreme_deadlocks': 0,
                'return_count': 0
            }

    def run_project_c(self, shared_config):
        """Run Project C (Multi-UAV Two-Step Auction) using shared environment"""
        print("\n" + "=" * 60)
        print("🎯 RUNNING PROJECT C - Multi-UAV Two-Step Auction")
        print("=" * 60)

        try:
            # Extract from shared config
            environment = shared_config['environment']
            battery_pos = shared_config['battery_pos']
            energy_capacity = shared_config['energy_capacity']
            dynamic_speed = shared_config['dynamic_speed']

            # Setup UI from shared config
            ui = Grid_Map()
            ui.map = environment.copy()
            ui.row_count = shared_config['row_count']
            ui.col_count = shared_config['col_count']
            ui.battery_pos = battery_pos
            ui.dynamic_obstacles = shared_config['dynamic_obstacles'].copy()

            # Initialize robot
            robot = project_c.Robot(battery_pos, ui.row_count, ui.col_count)
            robot.set_map(environment)

            # Initialize dynamic obstacles
            robot.dynamic_obstacles_manager = DynamicObstaclesManager(ui, num_obstacles=0, speed_factor=dynamic_speed)
            if ui.dynamic_obstacles:
                robot.dynamic_obstacles_manager.initialize_obstacles()

            start_time = time.time()

            # Run headless version
            results = self._run_project_c_headless(robot, robot.dynamic_obstacles_manager, ui, energy_capacity)

            execution_time = time.time() - start_time

            results.update({
                'algorithm': 'Project C - Multi-UAV Auction',
                'execution_time': execution_time,
                'success': True
            })

            return results

        except Exception as e:
            print(f"❌ Project C failed: {e}")
            return {
                'algorithm': 'Project C - Multi-UAV Auction',
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'total_path_length': 0,
                'overlap_rate': 0,
                'extreme_deadlocks': 0,
                'return_count': 0
            }

    def _run_project_a_headless(self, robot, dynamic_obstacles, ui, energy_capacity, max_steps=5000):
        """Run Project A without pygame interface - BWave Framework metrics calculation"""

        # Initialize metrics according to secondpaper.pdf Section 5.1
        total_path_length = 0  # Equation (4): Σ(|πi|)
        coverage_segments_length = 0  # Sπ^coverage for overlap calculation
        advance_segments_length = 0  # Sπ^advance
        return_segments_length = 0  # Sπ^return
        overlap_rate = 0  # Equation (5): ((Σ Sπ^coverage_i / S_free) - 1) × 100%
        deadlock_count = 0  # Total deadlocks
        extreme_deadlock_count = 0  # Equation (6): d_extreme > √(h² + w²) / 4
        return_count = 1  # Number of returns to charging station

        # Map dimensions for extreme deadlock calculation
        map_height, map_width = len(robot.map), len(robot.map[0])
        extreme_deadlock_threshold = math.sqrt(map_height ** 2 + map_width ** 2) / 4

        # Free cells count for overlap rate calculation
        free_cells = np.sum((robot.map != 1) & (robot.map != 'o'))  # S_free

        step = 0

        while step < max_steps:
            step += 1

            # Update dynamic obstacles
            delta_time = 0.1
            dynamic_obstacles.update(delta_time)

            # Check completion - BWave Framework finish condition
            if robot.logic.state == project_a.Q.FINISH:
                print("✅ Project A Coverage Complete!")
                break

            # Algorithm step
            wp = robot.logic.get_wp(robot.current_pos)
            if len(wp) == 0:
                continue

            selected_cell = robot.select_from_wp(wp)

            if selected_cell == robot.current_pos:
                robot.task()
            else:
                if robot.logic.state == project_a.Q.NORMAL:
                    # Energy check triggers return sequence
                    if not robot.check_enough_energy(selected_cell):
                        return_count += 1

                        # Calculate return path length (Sπ^return)
                        return_path_start = robot.current_pos
                        robot.charge_planning()  # This includes retreat->charge->advance
                        return_path_end = robot.current_pos
                        return_segment_length = math.dist(return_path_start, return_path_end)
                        return_segments_length += return_segment_length
                        total_path_length += return_segment_length
                        continue

                    # Normal coverage movement (Sπ^coverage)
                    old_pos = robot.current_pos
                    robot.move_to(selected_cell)
                    segment_length = math.dist(old_pos, selected_cell)
                    coverage_segments_length += segment_length
                    total_path_length += segment_length

                elif robot.logic.state == project_a.Q.DEADLOCK:
                    deadlock_count += 1
                    path, deadlock_distance = robot.logic.cache_path, robot.logic.cache_dist

                    # Check extreme deadlock according to Equation (6)
                    if deadlock_distance > extreme_deadlock_threshold:
                        extreme_deadlock_count += 1

                    # Execute deadlock escape (Sπ^advance)
                    if len(path) > 0:
                        escape_start = robot.current_pos
                        robot.follow_path_plan(path, check_energy=True, stop_on_unexpored=True)
                        escape_end = robot.current_pos
                        advance_segment_length = math.dist(escape_start, escape_end)
                        advance_segments_length += advance_segment_length
                        total_path_length += advance_segment_length

        # Calculate overlap rate according to Equation (5)
        if free_cells > 0:
            cells_covered_by_coverage_segments = coverage_segments_length  # Approximation
            overlap_rate = ((cells_covered_by_coverage_segments / free_cells) - 1) * 100
            overlap_rate = max(0, overlap_rate)  # Non-negative
        else:
            overlap_rate = 0

        return {
            'total_path_length': total_path_length,  # Equation (4)
            'coverage_length': coverage_segments_length,  # Coverage portion only
            'overlap_rate': overlap_rate,  # Equation (5)
            'overlap_times': int(overlap_rate),  # Convert to times for compatibility
            'deadlocks': deadlock_count,  # Total deadlocks
            'extreme_deadlocks': extreme_deadlock_count,  # Equation (6)
            'return_count': return_count,  # Number of returns
            'total_steps': step
        }

    def _run_project_d_headless(self, ccpp_env, environment, battery_pos, energy_capacity, dynamic_speed,
                                max_steps=5000):
        """Run Project D with BWave-compatible metrics calculation"""

        # Initialize CCPP robot
        ROW_COUNT, COL_COUNT = environment.shape
        ccpp_env.ccpp_robot = project_d.CCPPRobot(width=COL_COUNT, height=ROW_COUNT, sensor_range=2)

        # Convert obstacles and setup
        static_obstacles = ccpp_env.convert_bwave_to_ccpp_map(environment, COL_COUNT, ROW_COUNT)
        ccpp_env.ccpp_robot.add_obstacles(static_obstacles)

        start_x, start_y = battery_pos[1], battery_pos[0]
        ccpp_env.ccpp_robot.position = project_d.Position(start_x, start_y)
        ccpp_env.ccpp_robot.grid_state[start_y, start_x] = project_d.GridState.VISITED.value
        ccpp_env.ccpp_robot.path = [project_d.Position(start_x, start_y)]

        ccpp_env.energy_capacity = energy_capacity
        ccpp_env.current_energy = energy_capacity

        # Initialize dynamic obstacles manager
        ccpp_env.dynamic_obstacles = DynamicObstaclesManager(ccpp_env.ui, num_obstacles=0, speed_factor=dynamic_speed)
        if hasattr(ccpp_env.ui, 'dynamic_obstacles') and ccpp_env.ui.dynamic_obstacles:
            ccpp_env.dynamic_obstacles.initialize_obstacles()

        # BWave-compatible metrics initialization
        total_path_length = 0
        coverage_segments_length = 0
        advance_segments_length = 0
        return_segments_length = 0
        deadlock_count = 0
        extreme_deadlock_count = 0
        return_count = 1

        # Extreme deadlock threshold
        extreme_deadlock_threshold = math.sqrt(ROW_COUNT ** 2 + COL_COUNT ** 2) / 4
        free_cells = ROW_COUNT * COL_COUNT - len(static_obstacles)

        step = 0
        while step < max_steps:
            step += 1

            # Update dynamic obstacles
            delta_time = 0.1
            if ccpp_env.dynamic_obstacles:
                ccpp_env.dynamic_obstacles.update(delta_time)

            # Check completion
            import torch
            total_unvisited = torch.sum(ccpp_env.ccpp_robot.grid_state == project_d.GridState.UNVISITED.value).item()
            if total_unvisited == 0:
                print("✅ Project D Coverage Complete!")
                break

            # Update neural activity
            ccpp_env.ccpp_robot.update_neural_activity()
            ccpp_env.ccpp_robot.update_backtrack_list()

            # Get next position
            next_pos = ccpp_env.ccpp_robot.select_next_position_with_priority()

            if next_pos is not None:
                # Calculate movement distance
                distance = math.sqrt((next_pos.x - ccpp_env.ccpp_robot.position.x) ** 2 +
                                     (next_pos.y - ccpp_env.ccpp_robot.position.y) ** 2)

                # Check energy - triggers return sequence
                if not ccpp_env.check_energy_for_return(next_pos, battery_pos):
                    return_count += 1
                    # Calculate return distance
                    charging_pos = project_d.Position(battery_pos[1], battery_pos[0])
                    return_distance = math.sqrt((ccpp_env.ccpp_robot.position.x - charging_pos.x) ** 2 +
                                                (ccpp_env.ccpp_robot.position.y - charging_pos.y) ** 2)
                    return_segments_length += return_distance
                    total_path_length += return_distance
                    ccpp_env.charge_robot()
                    continue

                # Normal coverage movement
                ccpp_env.ccpp_robot.position = next_pos
                ccpp_env.ccpp_robot.path.append(next_pos)
                ccpp_env.ccpp_robot.grid_state[next_pos.y, next_pos.x] = project_d.GridState.VISITED.value
                ccpp_env.update_energy_system(distance, is_coverage=True)
                coverage_segments_length += distance
                total_path_length += distance

            elif ccpp_env.ccpp_robot.is_deadlock():
                deadlock_count += 1
                backtrack_point = ccpp_env.ccpp_robot.select_best_backtrack_point()

                if backtrack_point:
                    # Calculate backtrack distance
                    backtrack_distance = math.sqrt((backtrack_point.x - ccpp_env.ccpp_robot.position.x) ** 2 +
                                                   (backtrack_point.y - ccpp_env.ccpp_robot.position.y) ** 2)

                    # Check extreme deadlock
                    if backtrack_distance > extreme_deadlock_threshold:
                        extreme_deadlock_count += 1

                    # Execute backtrack
                    path = ccpp_env.ccpp_robot.dynamic_a_star(ccpp_env.ccpp_robot.position, backtrack_point)
                    if path and len(path) > 1:
                        total_backtrack_distance = 0
                        for i in range(1, len(path)):
                            step_distance = math.sqrt(
                                (path[i].x - path[i - 1].x) ** 2 + (path[i].y - path[i - 1].y) ** 2)
                            total_backtrack_distance += step_distance
                            ccpp_env.ccpp_robot.position = path[i]
                            ccpp_env.ccpp_robot.path.append(path[i])
                            ccpp_env.update_energy_system(step_distance, is_coverage=False)

                        advance_segments_length += total_backtrack_distance
                        total_path_length += total_backtrack_distance
                else:
                    break

        # Calculate overlap rate
        if free_cells > 0:
            overlap_rate = ((coverage_segments_length / free_cells) - 1) * 100
            overlap_rate = max(0, overlap_rate)
        else:
            overlap_rate = 0

        return {
            'total_path_length': total_path_length,
            'coverage_length': coverage_segments_length,
            'overlap_rate': overlap_rate,
            'overlap_times': int(overlap_rate),
            'deadlocks': deadlock_count,
            'extreme_deadlocks': extreme_deadlock_count,
            'return_count': return_count,
            'total_steps': step
        }

    def _run_project_c_headless(self, robot, dynamic_obstacles, ui, energy_capacity, max_steps=5000):
        """Run Project C without pygame interface"""

        # Initialize metrics
        coverage_length = 0
        overlap_times = 0
        extreme_deadlocks = 0
        return_count = 1
        deadlock_count = 0

        step = 0
        loop_count = 0

        while step < max_steps:
            step += 1
            loop_count += 1

            # Update dynamic obstacles
            delta_time = 0.1
            dynamic_obstacles.update(delta_time)

            # Update Project C dynamic maps
            robot.update_dynamic_map_b(loop_count)
            if loop_count % robot.velocity == 0:
                robot.update_probability_map_and_seen_map_b()

            # Check completion
            if robot.logic.state == project_c.Q_B.FINISH:
                print("✅ Project C Coverage Complete!")
                break

            # Execute task first (Project C requirement)
            if robot.logic.state != project_c.Q_B.DEADLOCK:
                robot.task()

            # P-Decision Framework
            flag_b = robot.detect_dynamic_obs_b(project_c.VISION_SENSOR_RANGE)

            if flag_b:
                robot.logic.set_map(robot.seen_map)
                robot.logic.set_prob_map(robot.prob_map)
                max_bid_value, replan_wp = robot.logic.get_replan_wp(robot.current_pos)
                wp = [replan_wp] if replan_wp else []

                # Go-or-wait decision
                designated_wp = robot.logic.boustrophedon_moving(robot.current_pos)
                if wp != designated_wp and robot.prob_map[robot.current_pos] < project_c.MIN_PROB_THRESHOLD and len(
                        designated_wp) > 0:
                    designated_wp = designated_wp[0]
                    if robot.prob_map[designated_wp] > 0:
                        continue  # Wait
                    else:
                        wp = [designated_wp]
            else:
                wp = robot.logic.get_wp(robot.current_pos)

            if len(wp) == 0:
                if robot.logic.state == project_c.Q_B.DEADLOCK:
                    selected_cell = None
                else:
                    continue
            else:
                selected_cell = robot.select_from_wp(wp)

            # Handle movement
            if selected_cell is None and robot.logic.state == project_c.Q_B.DEADLOCK:
                deadlock_count += 1
                path = robot.logic.escape_deadlock_path(robot.current_pos)

                if len(path) == 0:
                    robot.logic.state = project_c.Q_B.FINISH
                    continue
                else:
                    # Check extreme deadlock
                    deadlock_distance = np.linalg.norm(np.array(path[-1]) - np.array(robot.current_pos))
                    map_diagonal = np.sqrt(len(robot.static_map) ** 2 + len(robot.static_map[0]) ** 2)
                    if deadlock_distance > map_diagonal / 4:
                        extreme_deadlocks += 1

                    # Execute deadlock escape
                    old_pos = robot.current_pos
                    robot.move_to(path[0])
                    coverage_length += np.linalg.norm(np.array(path[0]) - np.array(old_pos))

            elif selected_cell is not None:
                if robot.logic.state == project_c.Q_B.NORMAL:
                    if not robot.check_enough_energy(selected_cell):
                        return_count += 1
                        robot.charge_planning()
                        continue

                    # Move and track metrics
                    old_pos = robot.current_pos
                    robot.move_to(selected_cell)
                    coverage_length += np.linalg.norm(np.array(selected_cell) - np.array(old_pos))

                    # Check overlap
                    if robot.static_map[selected_cell] == 2:
                        overlap_times += 1

        return {
            'coverage_length': coverage_length,
            'overlap_times': overlap_times,
            'extreme_deadlocks': extreme_deadlocks,
            'return_count': return_count,
            'total_steps': step
        }

    def run_comparison(self, map_files, energy_capacity=1000, dynamic_speed=0.1, output_dir="comparison_results"):
        """Run comparison with SINGLE environment setup, then test all algorithms"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Use ONLY first map for setup
        first_map = map_files[0]

        print(f"\n{'=' * 80}")
        print(f"🗺️ SETTING UP ENVIRONMENT: {first_map}")
        print(f"{'=' * 80}")

        # Setup SHARED environment ONCE ONLY
        shared_config = self.setup_shared_environment(first_map, energy_capacity, dynamic_speed)

        print(f"\n🚀 Environment setup complete! Now running ALL 3 algorithms...")
        print(f"{'=' * 80}")

        map_results = {
            'map_file': first_map,
            'config': shared_config,
            'algorithms': {}
        }

        # Run all algorithms using SHARED environment
        algorithms = [
            ('Project A', self.run_project_a),
            ('Project D', self.run_project_d),
            ('Project C', self.run_project_c)
        ]

        for name, run_func in algorithms:
            print(f"\n🔄 Running {name} with shared environment...")
            result = run_func(shared_config)
            map_results['algorithms'][name] = result

            if result['success']:
                print(f"✅ {name} completed successfully")
                print(f"   Total Path Length: {result.get('total_path_length', 0):.2f}")
                print(f"   Execution: {result['execution_time']:.2f}s")
                print(f"   Returns: {result.get('return_count', 0)}")
            else:
                print(f"❌ {name} failed: {result.get('error', 'Unknown error')}")

        all_results = [map_results]

        # Generate reports
        self._generate_comparison_report(all_results, output_dir)

        return all_results

    def _generate_comparison_report(self, all_results, output_dir):
        """Generate comprehensive comparison report"""

        print(f"\n📊 GENERATING COMPARISON REPORT...")

        # Collect data for visualization
        algorithms = ['Project A', 'Project D', 'Project C']
        # BWave paper compliant metrics
        metrics = ['total_path_length', 'overlap_rate', 'execution_time', 'extreme_deadlocks', 'return_count', 'deadlocks']

        # Create summary DataFrame
        summary_data = []
        for map_result in all_results:
            map_name = os.path.basename(map_result['map_file'])
            for alg_name in algorithms:
                if alg_name in map_result['algorithms'] and map_result['algorithms'][alg_name]['success']:
                    alg_result = map_result['algorithms'][alg_name]
                    summary_data.append({
                        'Map': map_name,
                        'Algorithm': alg_name,
                        'Coverage Length': alg_result['coverage_length'],
                        'Overlap Times': alg_result['overlap_times'],
                        'Execution Time': alg_result['execution_time'],
                        'Extreme Deadlocks': alg_result['extreme_deadlocks'],
                        'Return Count': alg_result['return_count']
                    })

        df = pd.DataFrame(summary_data)

        # Save CSV
        csv_file = os.path.join(output_dir, f'comparison_results_{self.timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"📄 CSV report saved: {csv_file}")

        # Create visualizations
        self._create_comparison_charts(df, output_dir)

        # Generate JSON report
        json_file = os.path.join(output_dir, f'detailed_results_{self.timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"📄 JSON report saved: {json_file}")

        # Print summary
        self._print_summary(df)

    def _create_comparison_charts(self, df, output_dir):
        """Create comparison bar charts"""

        metrics = {
            'Coverage Length': 'Coverage Length Comparison',
            'Overlap Times': 'Overlap Times Comparison',
            'Execution Time': 'Execution Time Comparison (seconds)',
            'Extreme Deadlocks': 'Extreme Deadlocks Comparison',
            'Return Count': 'Return Count Comparison'
        }

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, (metric, title) in enumerate(metrics.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Group data by algorithm
            algorithm_data = {}
            for alg in df['Algorithm'].unique():
                algorithm_data[alg] = df[df['Algorithm'] == alg][metric].values

            # Create bar chart
            x = np.arange(len(list(algorithm_data.values())[0]))
            width = 0.25

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

            for j, (alg, values) in enumerate(algorithm_data.items()):
                ax.bar(x + j * width, values, width, label=alg, color=colors[j], alpha=0.8)

            ax.set_xlabel('Test Maps')
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Set x-axis labels as map indices
            ax.set_xticks(x + width)
            ax.set_xticklabels([f'Map {i + 1}' for i in range(len(x))])

        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()

        # Save chart
        chart_file = os.path.join(output_dir, f'comparison_charts_{self.timestamp}.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Charts saved: {chart_file}")

        plt.show()

    def _print_summary(self, df):
        """Print summary statistics"""
        print(f"\n{'=' * 80}")
        print("📈 COMPARISON SUMMARY")
        print(f"{'=' * 80}")

        # Calculate averages by algorithm
        summary = df.groupby('Algorithm').agg({
            'Coverage Length': ['mean', 'std'],
            'Overlap Times': ['mean', 'std'],
            'Execution Time': ['mean', 'std'],
            'Extreme Deadlocks': ['mean', 'std'],
            'Return Count': ['mean', 'std']
        }).round(2)

        print("\n📊 Average Performance (± Standard Deviation):")
        print("=" * 80)

        for alg in summary.index:
            print(f"\n🔹 {alg}:")
            for metric in ['Coverage Length', 'Overlap Times', 'Execution Time', 'Extreme Deadlocks', 'Return Count']:
                mean = summary.loc[alg, (metric, 'mean')]
                std = summary.loc[alg, (metric, 'std')]
                print(f"  {metric:20s}: {mean:8.2f} ± {std:6.2f}")

        print(f"\n{'=' * 80}")
        print("🏆 WINNERS BY METRIC:")
        print(f"{'=' * 80}")

        # Find winners for each metric
        winners = df.groupby('Algorithm').mean()

        metrics_desc = {
            'Coverage Length': 'Shortest path (lower is better)',
            'Overlap Times': 'Fewest overlaps (lower is better)',
            'Execution Time': 'Fastest execution (lower is better)',
            'Extreme Deadlocks': 'Fewest extreme deadlocks (lower is better)',
            'Return Count': 'Fewest returns (lower is better)'
        }

        for metric, desc in metrics_desc.items():
            winner = winners[metric].idxmin()
            winner_value = winners[metric].min()
            print(f"🥇 {metric:20s}: {winner:<15s} ({winner_value:.2f}) - {desc}")


def main():
    parser = argparse.ArgumentParser(description='Compare Coverage Path Planning Algorithms')
    parser.add_argument('--maps', nargs='+',
                        default=['map/real_map/denmark.txt'],
                        help='Map files to test')
    parser.add_argument('--energy', type=float, default=1000,
                        help='Robot energy capacity')
    parser.add_argument('--speed', type=float, default=0.1,
                        help='Dynamic obstacles speed factor')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='Output directory for results')

    args = parser.parse_args()

    print("🚀 ALGORITHM COMPARISON SYSTEM")
    print("=" * 50)
    print("Comparing:")
    print("  • Project A: BWave Framework (Papers 1+2)")
    print("  • Project D: Neural Dynamics CCPP (Paper 3)")
    print("  • Project C: Multi-UAV Auction (Paper 4)")
    print("=" * 50)

    # Run comparison
    comparison = AlgorithmComparison()
    results = comparison.run_comparison(
        map_files=args.maps,
        energy_capacity=args.energy,
        dynamic_speed=args.speed,
        output_dir=args.output
    )

    print(f"\n✅ Comparison completed! Results saved in '{args.output}' directory")
    print(f"📊 Charts and detailed reports are available for analysis")


if __name__ == "__main__":
    main()