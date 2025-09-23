import pygame as pg
import numpy as np
import torch
import time
import argparse
import math

# Import Project A environment
from grid_map import Grid_Map
from dynamic_obstacles_manager import DynamicObstaclesManager

# Import Project D algorithm
from project_D.ccpp_robot_main import CCPPRobot, GridState, Position


class CCPPInBWaveEnvironment:
    def __init__(self):
        """CCPP Robot hoạt động trong BWave Environment"""
        self.robot_speed = 1.0
        self.ui = None
        self.dynamic_obstacles = None
        self.ccpp_robot = None
        self.energy_capacity = 1000
        self.current_energy = 1000

        # Metrics tracking - BWave compliant
        self.total_travel_length = 0.0
        self.coverage_length = 0
        self.advance_length = 0  # ✅ NEW: Track advance movements
        self.retreat_length = 0  # ✅ NEW: Track retreat movements
        self.return_charge_count = 1
        self.deadlock_count = 0
        self.extreme_deadlock_count = 0
        self.execute_time = 0
        self.total_coverage_cells = 0
        self.total_free_cells = 0
        self.covered_positions = set()  # ✅ NEW: Track unique covered positions (no overlap)
        self.blank_cells = 0  # ✅ NEW: Track initial blank cells before dynamic obstacles
        self.count_cell_go_through = 0  # ✅ FIXED: Track coverage moves for correct overlap calculation

    def convert_bwave_to_ccpp_map(self, bwave_map, width, height):
        """Convert BWave map format to CCPP format"""
        # BWave: 0=free, 1=obstacle, 'd'=dynamic, 'e'=explored
        # CCPP: GridState.UNVISITED, GridState.OBSTACLE, GridState.VISITED

        obstacles = []
        for row in range(height):
            for col in range(width):
                if bwave_map[row][col] in (1, 'o'):  # Static obstacles
                    obstacles.append((col, row))  # CCPP uses (x,y), BWave uses (row,col)

        return obstacles

    def convert_ccpp_to_bwave_path(self, ccpp_path):
        """Convert CCPP path format to BWave visualization format"""
        # CCPP: Position(x, y)
        # BWave: (row, col)
        bwave_path = []
        for pos in ccpp_path:
            bwave_path.append((pos.y, pos.x))  # Convert x,y to row,col
        return bwave_path

    def get_dynamic_obstacle_positions(self):
        """Get current dynamic obstacle positions from BWave manager"""
        if self.dynamic_obstacles is None:
            return []
        positions = []
        for obstacle in self.dynamic_obstacles.obstacles:
            # Convert BWave (row, col) to CCPP (x, y)
            pos_x = obstacle['pos'][1]  # col -> x
            pos_y = obstacle['pos'][0]  # row -> y
            positions.append((pos_x, pos_y))
        return positions

    def update_energy_system(self, distance_moved, is_coverage=True):
        """Update energy system like Project A"""
        if is_coverage:
            energy_cost = distance_moved  # 1 unit per distance for coverage
        else:
            energy_cost = 0.5 * distance_moved  # 0.5 unit for advance/retreat
        self.current_energy -= energy_cost

        if self.current_energy <= 0:
            self.current_energy = 0
            return False  # Out of energy
        return True

    def check_energy_for_return(self, current_pos, battery_pos):
        """Check if enough energy to return to charging station - Paper compliant"""
        return_distance = math.sqrt((current_pos.x - battery_pos[1]) ** 2 +
                                    (current_pos.y - battery_pos[0]) ** 2)
        return_energy_needed = 0.5 * return_distance  # Half energy for return as in paper
        return self.current_energy >= return_energy_needed

    def charge_robot(self):
        """Charge robot to full capacity"""
        self.current_energy = self.energy_capacity
        self.return_charge_count += 1  # Increment when actually charging
        print(f"🔋 Robot charged! Total charges: {self.return_charge_count}")

    def run_ccpp_with_bwave_environment(self, map_file, energy_capacity=1000, dynamic_speed=0.1):
        """
        Main execution:
        1. Load BWave environment với UI interactions
        2. Setup CCPP robot
        3. Run CCPP algorithm trong BWave environment
        4. Real-time pygame visualization
        """

        print("=" * 80)
        print("CCPP ALGORITHM IN BWAVE ENVIRONMENT")
        print("Project D Algorithm + Project A Environment")
        print("=" * 80)

        # 1. Setup BWave Environment (giữ nguyên Project A)
        self.ui = Grid_Map()
        environment, battery_pos = self.ui.read_map(map_file)

        print("\n🎮 MAP EDITOR - Create obstacles and charging station:")
        print("- Left click: Static obstacles")
        print("- Shift + Left click: Dynamic obstacles")
        print("- Right click: Charging station")
        print("- Close window when done")

        # UI Editor phase - giữ nguyên Project A
        environment, battery_pos = self.ui.edit_map()

        ROW_COUNT = len(environment)
        COL_COUNT = len(environment[0])

        print(f"\n📏 Environment: {ROW_COUNT}x{COL_COUNT}")
        print(f"🔋 Energy Capacity: {energy_capacity}")
        print(f"🏠 Charging Station: {battery_pos}")

        # 2. Setup Dynamic Obstacles Manager (giữ nguyên Project A)
        self.dynamic_obstacles = DynamicObstaclesManager(
            self.ui, num_obstacles=0, speed_factor=dynamic_speed
        )

        # Initialize manual dynamic obstacles từ UI
        if hasattr(self.ui, 'dynamic_obstacles') and self.ui.dynamic_obstacles:
            self.dynamic_obstacles.initialize_obstacles()
            print(f"🚶 Created {len(self.ui.dynamic_obstacles)} dynamic obstacles")

        # 3. Setup CCPP Robot (Project D algorithm)
        self.ccpp_robot = CCPPRobot(width=COL_COUNT, height=ROW_COUNT, sensor_range=2)

        # Convert BWave obstacles to CCPP format
        static_obstacles = self.convert_bwave_to_ccpp_map(environment, COL_COUNT, ROW_COUNT)
        self.ccpp_robot.add_obstacles(static_obstacles)

        # ✅ FIX: Define start position from battery_pos
        start_x = battery_pos[1]  # Convert (row, col) to (x, y)
        start_y = battery_pos[0]  # Convert (row, col) to (x, y)

        # Set robot initial position
        self.ccpp_robot.position = Position(start_x, start_y)

        # CRITICAL FIX: Mark current position as VISITED immediately
        self.ccpp_robot.grid_state[start_y, start_x] = GridState.VISITED.value
        self.ccpp_robot.external_input[start_y, start_x] = 0.0
        self.ui.map[start_y][start_x] = 'e'
        self.total_coverage_cells = 1  # Starting position counts as first coverage
        self.count_cell_go_through = 1  # ✅ FIXED: Initialize coverage moves counter
        self.ui.task((start_y, start_x))  # Mark in UI as explored
        print(f"🤖 CCPP Robot initialized at ({start_x}, {start_y}) - marked as VISITED")
        print(f"🧠 Initial neural activity: {self.ccpp_robot.neural_activity[start_y, start_x].item():.2f}")

        # ✅ ENSURE ROBOT STARTS AT CHARGING STATION
        print(f"🏠 Charging Station at: {battery_pos} (row, col)")
        print(f"🤖 Robot starting position: ({start_x}, {start_y}) (x, y coordinates)")

        # Add starting position to path
        self.ccpp_robot.path = [Position(start_x, start_y)]

        # Update UI to show robot at charging station
        self.ui.update_vehicle_pos((start_y, start_x))  # UI uses (row, col)

        # Set energy system
        self.energy_capacity = energy_capacity
        self.current_energy = energy_capacity

        # ✅ AUTO CLEAR STARTING MARKS ('s') - Same as main_paper12.py
        for x in range(ROW_COUNT):
            for y in range(COL_COUNT):
                if self.ui.map[x, y] == 's':
                    self.ui.map[x, y] = 0

        self.ui.draw_map()
        pg.display.flip()
        time.sleep(0.5)

        # Tính tổng số free cells (S_free) - BWave compliant
        self.blank_cells = np.sum(environment == 0)
        print(f"Blank cells (before dynamic obstacles): {self.blank_cells}")

        # Tính tổng số free cells (S_free) - BWave compliant
        self.total_free_cells = np.sum(environment == 0)
        print(f"Total free cells in environment: {self.total_free_cells}")
        print(f"🤖 CCPP Robot initialized at ({start_x}, {start_y})")

        # 4. Main Algorithm Loop với BWave Visualization
        print("\n🚀 Starting CCPP Coverage Algorithm...")
        print("🎮 CONTROLS:")
        print("  SPACE: Pause/Resume")
        print("  LEFT/RIGHT: Change simulation speed")
        print("  UP/DOWN: Increase/Decrease dynamic obstacle speed")
        print("  ESC: Quit")
        print(f"🤖 Robot starting from charging station at ({start_x}, {start_y})")

        FPS = 40
        clock = pg.time.Clock()
        run = True
        pause = False
        step = 0
        max_steps = float('inf')  # Remove artificial limit - run until coverage complete

        self.execute_time = time.time()
        self.movement_timer = 0
        self.ROBOT_MOVE_INTERVAL = 0.01 / self.robot_speed
        while run and step < max_steps:
            # Check coverage completion early - Paper compliant termination
            total_unvisited = torch.sum(self.ccpp_robot.grid_state == GridState.UNVISITED.value).item()
            if total_unvisited == 0:
                print("✅ Coverage Complete - All accessible cells visited!")
                break
            current_time = time.time()
            delta_time = clock.get_time() / 1000.0
            self.movement_timer += delta_time

            # Update dynamic obstacles (Project A)
            if self.dynamic_obstacles:
                self.dynamic_obstacles.update(delta_time)

            # Pygame event handling
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        pause = not pause
                    elif event.key == pg.K_LEFT:
                        FPS = max(1, FPS // 2)
                        print(f"🐌 Speed: {FPS} FPS")
                    elif event.key == pg.K_RIGHT:
                        FPS = min(128000, FPS * 2)
                        print(f"🏃 Speed: {FPS} FPS")
                    elif event.key == pg.K_ESCAPE:
                        run = False
                    elif event.key == pg.K_UP:
                        # ✅ TĂNG VẬN TỐC VẬT CẢN ĐỘNG
                        if self.dynamic_obstacles and self.dynamic_obstacles.obstacles:
                            for obs in self.dynamic_obstacles.obstacles:
                                vx, vy = obs['velocity']
                                obs['velocity'] = (vx * 2, vy * 2)
                            print("↑ Tăng vận tốc vật cản động ×2")
                        else:
                            print("⚠️ Không có vật cản động nào để điều chỉnh")
                    elif event.key == pg.K_DOWN:
                        # ✅ GIẢM VẬN TỐC VẬT CẢN ĐỘNG
                        if self.dynamic_obstacles and self.dynamic_obstacles.obstacles:
                            for obs in self.dynamic_obstacles.obstacles:
                                vx, vy = obs['velocity']
                                obs['velocity'] = (vx / 2, vy / 2)
                            print("↓ Giảm vận tốc vật cản động ÷2")
                        else:
                            print("⚠️ Không có vật cản động nào để điều chỉnh")

            if pause:
                clock.tick(FPS)
                continue

            # 5. CCPP Algorithm Step (Project D) 
            algorithm_step_counter = getattr(self, 'algorithm_step_counter', 0)
            algorithm_step_counter += 1

            neural_update_counter = getattr(self, 'neural_update_counter', 0)
            neural_update_counter += 1

            # ✅ PAPER COMPLIANT: Run algorithm every step as in paper
            if self.movement_timer >= self.ROBOT_MOVE_INTERVAL:

                # ✅ NEURAL UPDATES: Every step as in paper equation (1)
                self.ccpp_robot.update_neural_activity()

                # Update backtracking list (Algorithm 1) - every step as in paper
                self.ccpp_robot.update_backtrack_list()

                # ✅ PAPER 3 COMPLIANT: Sensor-based detection only
                if hasattr(self.ccpp_robot, 'dynamic_obstacle_positions'):
                    self.ccpp_robot.dynamic_obstacle_positions = set()
                else:
                    self.ccpp_robot.dynamic_obstacle_positions = set()

                # Simulate sensor detection exactly like ccpp_robot_main.py
                if self.dynamic_obstacles and self.dynamic_obstacles.obstacles:
                    robot_x, robot_y = self.ccpp_robot.position.x, self.ccpp_robot.position.y
                    sensor_range = self.ccpp_robot.sensor_range

                    # Get current obstacle positions for sensor simulation
                    current_obstacle_positions = []
                    for obstacle in self.dynamic_obstacles.obstacles:
                        obs_row, obs_col = obstacle['pos']  # BWave format (row, col)
                        obs_x, obs_y = obs_col, obs_row  # Convert to CCPP format (x, y)
                        current_obstacle_positions.append((obs_x, obs_y))

                    # Use ccpp_robot's own sensor detection method
                    detected_obstacles = self.ccpp_robot.simulate_sensor_detection(current_obstacle_positions)

                    # Only add detected obstacles
                    for obs_x, obs_y in detected_obstacles:
                        self.ccpp_robot.dynamic_obstacle_positions.add((obs_x, obs_y))
                        # Mark as obstacle in CCPP grid only if detected and not visited
                        if (0 <= obs_x < self.ccpp_robot.width and 0 <= obs_y < self.ccpp_robot.height and
                                self.ccpp_robot.grid_state[obs_y, obs_x] != GridState.VISITED.value):
                            self.ccpp_robot.grid_state[obs_y, obs_x] = GridState.OBSTACLE.value
                            self.ccpp_robot.external_input[obs_y, obs_x] = -self.ccpp_robot.E

                # Try normal movement (Priority template)
                next_pos = self.ccpp_robot.select_next_position_with_priority()

                if next_pos is not None:
                    # Check energy before movement
                    distance = math.sqrt((next_pos.x - self.ccpp_robot.position.x) ** 2 +
                                         (next_pos.y - self.ccpp_robot.position.y) ** 2)

                    if not self.check_energy_for_return(next_pos, battery_pos):
                        # Need to return for charging
                        print("⚡ Low energy! Returning to charge...")
                        # ✅ PROPER ENERGY MANAGEMENT: Plan return path instead of teleport
                        charging_pos = Position(battery_pos[1], battery_pos[0])  # Convert (row,col) to (x,y)
                        return_path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, charging_pos)

                        if return_path and len(return_path) > 1:
                            # Execute return path
                            for i, pos in enumerate(return_path[1:]):  # Skip current position
                                distance = math.sqrt((pos.x - self.ccpp_robot.position.x) ** 2 +
                                                     (pos.y - self.ccpp_robot.position.y) ** 2)

                                self.ccpp_robot.position = pos
                                self.ccpp_robot.path.append(pos)

                                # ✅ FIXED: Count ALL movements for overlap calculation
                                self.count_cell_go_through += 1  # Track every movement including return to charge

                                # Update energy for return movement (0.5x cost) - NOT coverage
                                self.update_energy_system(distance, is_coverage=False)
                                self.total_travel_length += distance
                                self.retreat_length += distance  # ✅ NEW: Track retreat distance
                            # Charge robot
                            self.charge_robot()
                            print(f"🔋 Robot charged at ({charging_pos.x}, {charging_pos.y})")
                        else:
                            # Emergency teleport if no path found
                            print("🚨 Emergency teleport to charging station")
                            self.ccpp_robot.position = charging_pos
                            self.ccpp_robot.path.append(charging_pos)
                            self.charge_robot()
                    else:
                        # Normal movement
                        self.ccpp_robot.position = next_pos
                        self.ccpp_robot.path.append(next_pos)
                        self.ccpp_robot.grid_state[next_pos.y, next_pos.x] = GridState.VISITED.value
                        self.ccpp_robot.external_input[next_pos.y, next_pos.x] = 0.0

                        # BWave metrics tracking
                        self.total_travel_length += distance  # Distance-based
                        self.coverage_length += distance  # Coverage segment only
                        self.total_coverage_cells += 1  # Count each coverage task
                        self.covered_positions.add((next_pos.y, next_pos.x))  # ✅ NEW: Add unique position
                        self.count_cell_go_through += 1  # ✅ FIXED: Track coverage moves for correct overlap calculation
                        self.update_energy_system(distance, is_coverage=True)

                        # Update BWave UI map for visualization
                        self.ui.map[next_pos.y][next_pos.x] = 'e'  # Mark as explored

                elif self.ccpp_robot.is_deadlock():
                    # Deadlock situation (Algorithm 2) - BWave compliant
                    self.deadlock_count += 1
                    backtrack_point = self.ccpp_robot.select_best_backtrack_point()

                    # Tính extreme deadlock (dist > 1/4 diagonal)
                    if backtrack_point:
                        backtrack_distance = math.sqrt(
                            (backtrack_point.x - self.ccpp_robot.position.x) ** 2 +
                            (backtrack_point.y - self.ccpp_robot.position.y) ** 2
                        )
                        diagonal_quarter = math.sqrt(COL_COUNT ** 2 + ROW_COUNT ** 2) / 4
                        if backtrack_distance > diagonal_quarter:
                            self.extreme_deadlock_count += 1

                    if backtrack_point:
                        print(f"🔙 Backtracking to {backtrack_point.x}, {backtrack_point.y}")
                        # Use dynamic A* for backtracking
                        path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, backtrack_point)
                        if path and len(path) > 1:
                            for i, pos in enumerate(path[1:]):  # Skip current position
                                # ✅ IMPROVED: Check energy during backtracking
                                if not self.check_energy_for_return(pos, battery_pos):
                                    print("⚠️ Low energy during backtracking - returning to charge")
                                    break

                                # Move to backtrack path (can go through visited cells)
                                distance = math.sqrt((pos.x - self.ccpp_robot.position.x) ** 2 +
                                                     (pos.y - self.ccpp_robot.position.y) ** 2)

                                self.ccpp_robot.position = pos
                                self.ccpp_robot.path.append(pos)

                                # ✅ FIXED: Count ALL movements for overlap calculation
                                self.count_cell_go_through += 1  # Track every movement including backtrack

                                # Update energy for backtrack movement (0.5x cost) - NOT coverage
                                self.update_energy_system(distance, is_coverage=False)
                                self.total_travel_length += distance
                                self.retreat_length += distance  # ✅ NEW: Track backtrack as retreat
                                # ✅ IMPORTANT: Don't mark visited cells as explored again during backtracking
                                # Only mark truly unvisited cells
                                if self.ccpp_robot.grid_state[pos.y, pos.x].item() == GridState.UNVISITED.value:
                                    self.ccpp_robot.grid_state[pos.y, pos.x] = GridState.VISITED.value
                                    self.ccpp_robot.external_input[pos.y, pos.x] = 0.0
                                    self.ui.map[pos.y][pos.x] = 'e'
                                    self.ui.task((pos.y, pos.x))
                                    self.total_coverage_cells += 1
                                    self.covered_positions.add((pos.y, pos.x))  # ✅ NEW: Add unique position
                                    self.coverage_length += distance
                                print(
                                    f"Backtrack step {i + 1}: ({pos.x}, {pos.y}) - Energy: {self.current_energy:.1f}")
                        else:
                            # Cannot reach backtrack point, remove it from list
                            print(f"❌ Cannot reach backtrack point {backtrack_point.x}, {backtrack_point.y}")
                            if backtrack_point in self.ccpp_robot.backtrack_list:
                                self.ccpp_robot.backtrack_list.remove(backtrack_point)
                                print(f"🗑️  Removed unreachable backtrack point")
                    else:
                        print("⚠️ No valid backtrack points available")
                else:
                    # Check if coverage complete
                    total_unvisited = torch.sum(self.ccpp_robot.grid_state == GridState.UNVISITED.value).item()
                    if total_unvisited == 0:
                        print("✅ Coverage Complete!")
                        break
                    # Robot stuck but unvisited cells remain
                    # Force backtracking if stuck with remaining work
                    if total_unvisited > 0 and self.ccpp_robot.backtrack_list:
                        print("🚨 FORCE BACKTRACKING - Robot stuck but work remains")
                        self.deadlock_count += 1

                        backtrack_point = self.ccpp_robot.select_best_backtrack_point()
                        if backtrack_point:
                            path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, backtrack_point)
                            if path and len(path) > 1:
                                # Execute backtrack path step by step with proper movement
                                for i, pos in enumerate(path[1:]):  # Skip current position
                                    # Check energy before each step
                                    if not self.check_energy_for_return(pos, battery_pos):
                                        break

                                    # Calculate distance for this step
                                    distance = math.sqrt((pos.x - self.ccpp_robot.position.x) ** 2 +
                                                         (pos.y - self.ccpp_robot.position.y) ** 2)

                                    # Move robot step by step
                                    self.ccpp_robot.position = pos
                                    self.ccpp_robot.path.append(pos)

                                    # ✅ FIXED: Count ALL movements for overlap calculation
                                    self.count_cell_go_through += 1  # Track every movement including forced backtrack

                                    # Update energy for backtrack movement (0.5x cost)
                                    self.update_energy_system(distance, is_coverage=False)
                                    self.total_travel_length += distance
                                    self.retreat_length += distance  # ✅ NEW: Track force backtrack as retreat

                                    # Only mark truly unvisited cells
                                    if self.ccpp_robot.grid_state[pos.y, pos.x].item() == GridState.UNVISITED.value:
                                        self.ccpp_robot.grid_state[pos.y, pos.x] = GridState.VISITED.value
                                        self.ccpp_robot.external_input[pos.y, pos.x] = 0.0
                                        self.ui.map[pos.y][pos.x] = 'e'
                                        self.total_coverage_cells += 1  # Count coverage task
                                        self.covered_positions.add((pos.y, pos.x))  # ✅ NEW: Add unique position
                                        self.coverage_length += distance

                                    # Update UI visualization for each step
                                    robot_pos_bwave = (pos.y, pos.x)
                                    self.ui.update_vehicle_pos(robot_pos_bwave)
                                continue  # Continue the algorithm loop
                        # Paper compliant: Terminate if no backtrack points available
                        print("❌ No reachable unvisited cells found - terminating")
                        break
                self.movement_timer = 0
                step += 1

            # Store counter for next iteration
            self.algorithm_step_counter = algorithm_step_counter
            self.neural_update_counter = neural_update_counter

            # 6. BWave Visualization (Project A)
            self.ui.draw()

            # Draw dynamic obstacles
            if self.dynamic_obstacles:
                self.dynamic_obstacles.draw(self.ui.WIN)

            # Draw CCPP path
            if len(self.ccpp_robot.path) > 1:
                ccpp_path_bwave = self.convert_ccpp_to_bwave_path(self.ccpp_robot.path)
                self.ui.draw_path(ccpp_path_bwave, color=(255, 0, 0), width=2)

            # Draw current robot position
            if self.ccpp_robot.position:
                robot_pos_bwave = (self.ccpp_robot.position.y, self.ccpp_robot.position.x)
                self.ui.update_vehicle_pos(robot_pos_bwave)

            # Draw energy display
            self.ui.set_energy_display(self.current_energy)

            # Status display with dynamic obstacle info
            font = pg.font.SysFont(None, 24)
            status_text = f"Step: {step} | Deadlocks: {self.deadlock_count} | Energy: {self.current_energy:.1f}"
            status_surface = font.render(status_text, True, (255, 255, 255))
            self.ui.WIN.blit(status_surface, (10, 10))

            # ✅ DYNAMIC OBSTACLE SPEED INFO
            if self.dynamic_obstacles and self.dynamic_obstacles.obstacles:
                avg_speed = 0
                for obs in self.dynamic_obstacles.obstacles:
                    vx, vy = obs['velocity']
                    speed = math.sqrt(vx * vx + vy * vy)
                    avg_speed += speed
                avg_speed /= len(self.dynamic_obstacles.obstacles)

                speed_text = f"Dynamic Obstacles: {len(self.dynamic_obstacles.obstacles)} | Avg Speed: {avg_speed:.3f}"
                speed_surface = font.render(speed_text, True, (255, 255, 255))
                self.ui.WIN.blit(speed_surface, (10, 35))

            pg.display.flip()
            clock.tick(FPS)

        # 7. Final Results
        self.execute_time = time.time() - self.execute_time
        print("\n✅ CCPP Algorithm completed in BWave Environment!")

        # ===== OUTPUT =====
        print('\nCoverage:\t', self.coverage_length)
        print('Advance:\t', self.advance_length)
        print('Return:\t', self.retreat_length)
        print('-' * 8)
        print('Total Path Length:', self.total_travel_length)
        print('Time: ', self.execute_time)
        print('\n' + '=' * 50)
        print('BWAVE FRAMEWORK METRICS')
        print('=' * 50)

        # 1. Total Path Length (distance-based như BWave gốc)
        print(f'1. Total Path Length: {self.total_travel_length:.2f}')

        # 2. Overlap Rate - ✅ FIXED: Use correct formula to prevent negative values
        explored_cells = np.sum(self.ui.map == 'e')
        if explored_cells > 0:
            overlap_rate = (self.count_cell_go_through / explored_cells - 1) * 100
            print(f'2. Overlap Rate: {overlap_rate:.2f}%')
        else:
            print('2. Overlap Rate: 0.00%')

        # 3. Number of Returns
        print(f'3. Number of Returns: {self.return_charge_count}')

        # 4. Number of Deadlocks (total và extreme)
        print(f'4. Number of Deadlocks: {self.deadlock_count} (extreme: {self.extreme_deadlock_count})')

        # 5. Execution Time
        print(f'5. Execution Time: {self.execute_time:.3f}s')

        # 6. Coverage Rate (NEW)
        cover_cells = len(self.covered_positions)  # ✅ NEW: Unique coverage cells only
        if self.blank_cells > 0:
            coverage_rate = (cover_cells / self.blank_cells) * 100
            print(f'6. Coverage Rate: {coverage_rate:.2f}%')
        else:
            print('6. Coverage Rate: 0.00%')
        print('=' * 50)
        # Calculate final metrics for return value
        total_cells = COL_COUNT * ROW_COUNT
        obstacle_cells = len(static_obstacles)
        visited_cells = torch.sum(self.ccpp_robot.grid_state == GridState.VISITED.value).item()
        accessible_cells = total_cells - obstacle_cells
        final_coverage_rate = visited_cells / accessible_cells if accessible_cells > 0 else 0
        explored_cells = np.sum(self.ui.map == 'e')
        bwave_overlap_rate = (self.count_cell_go_through / explored_cells - 1) * 100 if explored_cells > 0 else 0
        print("🖼️  Press any key to close visualization...")
        waiting = True
        while waiting:
            try:
                for event in pg.event.get():
                    if event.type == pg.QUIT or event.type == pg.KEYDOWN:
                        waiting = False
                        break
                pg.time.wait(100)  # Prevent high CPU usage
            except KeyboardInterrupt:
                print("\n🚪 User interrupted - closing...")
                waiting = False
                break

        try:
            pg.quit()
        except:
            pass
        return {
            'coverage_rate': final_coverage_rate,
            'path_length': self.total_travel_length,  # Use distance-based như BWave
            'overlap_rate': bwave_overlap_rate,  # Use BWave formula
            'deadlock_count': self.deadlock_count,
            'return_charge_count': self.return_charge_count,
            'execution_time': self.execute_time,
            'total_steps': step
        }


def main():
    parser = argparse.ArgumentParser(description='CCPP Algorithm in BWave Environment')
    parser.add_argument('--map', type=str, default='map/real_map/cantwell.txt', help='Path to map file')
    parser.add_argument('--energy', type=float, default=1000,
                        help='Robot energy capacity')
    parser.add_argument('--speed', type=float, default=0.5, help='Dynamic obstacles speed factor')
    args = parser.parse_args()

    print("🚀 Starting CCPP in BWave Environment...")
    print(f"📁 Map: {args.map}")
    print(f"⚡ Energy: {args.energy}")
    print(f"🏃 Dynamic Speed: {args.speed}")

    ccpp_env = CCPPInBWaveEnvironment()
    try:
        results = ccpp_env.run_ccpp_with_bwave_environment(
            map_file=args.map,
            energy_capacity=args.energy,
            dynamic_speed=args.speed
        )
        print("\n🎉 Execution completed successfully!")
    except KeyboardInterrupt:
        print("\n🚪 Program interrupted by user (Ctrl+C)")
        print("🔍 Partial results may be incomplete")
        try:
            pg.quit()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        try:
            pg.quit()
        except:
            pass


if __name__ == "__main__":
    main()