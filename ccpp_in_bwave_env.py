import pygame as pg
import numpy as np
import torch
import time
import argparse
import math

# Import Project A environment (KHÔNG SỬA GỐC)
from grid_map import Grid_Map, EPSILON
from dynamic_obstacles_manager import DynamicObstaclesManager

# Import Project D algorithm (KHÔNG SỬA GỐC)
from project_D.ccpp_robot_main import CCPPRobot, GridState, Position

class CCPPInBWaveEnvironment:
    def __init__(self):
        """CCPP Robot hoạt động trong BWave Environment"""
        self.ui = None
        self.dynamic_obstacles = None
        self.ccpp_robot = None
        self.energy_capacity = 1000
        self.current_energy = 1000

        # Metrics tracking
        self.total_travel_length = 0
        self.coverage_length = 0
        self.return_charge_count = 1
        self.deadlock_count = 0
        self.execute_time = 0

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

    def clean_stale_dynamic_marks(self):
        """✅ Clean up old dynamic obstacle marks that are no longer occupied"""
        if not hasattr(self, 'ccpp_robot') or self.dynamic_obstacles is None:
            return

        # Get current actual dynamic obstacle positions
        current_dynamic_positions = set()
        for obstacle in self.dynamic_obstacles.obstacles:
            center_pos = obstacle['pos']  # (row, col)
            size = obstacle.get('size', 1.0)

            # Add all cells occupied by this obstacle
            if isinstance(size, tuple):
                height, width = size
                radius_h = height // 2
                radius_w = width // 2
                for dr in range(-radius_h, radius_h + 1):
                    for dc in range(-radius_w, radius_w + 1):
                        r, c = center_pos[0] + dr, center_pos[1] + dc
                        if (0 <= r < self.ccpp_robot.height and 0 <= c < self.ccpp_robot.width):
                            current_dynamic_positions.add((c, r))  # Convert to (x, y)
            else:
                radius = int(size // 2)
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        r, c = center_pos[0] + dr, center_pos[1] + dc
                        if (0 <= r < self.ccpp_robot.height and 0 <= c < self.ccpp_robot.width):
                            current_dynamic_positions.add((c, r))  # Convert to (x, y)

        # Clean up stale marks in CCPP grid
        for y in range(self.ccpp_robot.height):
            for x in range(self.ccpp_robot.width):
                # If cell is marked as dynamic but not actually occupied
                if (self.ui.map[y][x] == 'd' and
                        (x, y) not in current_dynamic_positions and
                        self.ccpp_robot.grid_state[y, x] != GridState.VISITED.value):

                    # Clean up the stale mark
                    self.ui.map[y][x] = 0  # Mark as free
                    # Reset to unvisited if it wasn't visited
                    if self.ccpp_robot.grid_state[y, x] != GridState.VISITED.value:
                        self.ccpp_robot.grid_state[y, x] = GridState.UNVISITED.value
                        self.ccpp_robot.external_input[y, x] = self.ccpp_robot.E

        return current_dynamic_positions

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
        """Check if enough energy to return to charging station"""
        return_distance = math.sqrt((current_pos.x - battery_pos[1]) ** 2 +
                                    (current_pos.y - battery_pos[0]) ** 2)
        return_energy_needed = 0.5 * return_distance  # Half energy for return
        return self.current_energy >= return_energy_needed

    def charge_robot(self):
        """Charge robot to full capacity"""
        self.current_energy = self.energy_capacity
        self.return_charge_count += 1
        print(f"🔋 Robot charged! Charge count: {self.return_charge_count}")

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

        # Save map if needed
        # self.ui.save_map('map/ccpp_test_map.txt')

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
        max_steps = 2000

        self.execute_time = time.time()

        while run and step < max_steps:
            current_time = time.time()
            delta_time = clock.get_time() / 1000.0

            # Update dynamic obstacles (Project A)
            if self.dynamic_obstacles:
                self.dynamic_obstacles.update(delta_time)

            # ✅ CLEAN STALE MARKS FIRST, then get current positions
            current_dynamic_positions = self.clean_stale_dynamic_marks()
            dynamic_positions = self.get_dynamic_obstacle_positions()

            # ✅ ONLY ADD ACTUAL CURRENT DYNAMIC OBSTACLES
            # Clear previous dynamic obstacle tracking
            if hasattr(self.ccpp_robot, 'dynamic_obstacle_positions'):
                self.ccpp_robot.dynamic_obstacle_positions.clear()
            else:
                self.ccpp_robot.dynamic_obstacle_positions = set()

            # Add current dynamic obstacles
            for pos_x, pos_y in current_dynamic_positions:
                self.ccpp_robot.dynamic_obstacle_positions.add((pos_x, pos_y))
                # Only mark as obstacle in CCPP if not already visited
                if (0 <= pos_x < self.ccpp_robot.width and 0 <= pos_y < self.ccpp_robot.height and
                        self.ccpp_robot.grid_state[pos_y, pos_x] != GridState.VISITED.value):
                    self.ccpp_robot.grid_state[pos_y, pos_x] = GridState.OBSTACLE.value
                    self.ccpp_robot.external_input[pos_y, pos_x] = -self.ccpp_robot.E

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

            # 5. CCPP Algorithm Step (Project D) - Chỉ chạy mỗi 4 frames
            algorithm_step_counter = getattr(self, 'algorithm_step_counter', 0)
            algorithm_step_counter += 1

            neural_update_counter = getattr(self, 'neural_update_counter', 0)
            neural_update_counter += 1

            # ✅ OPTIMIZATION: Run algorithm every 2 frames, neural updates every 8 frames
            if algorithm_step_counter % 2 == 0:  # Algorithm at 20 FPS instead of 10 FPS

                # ✅ NEURAL UPDATES: Only every 8 frames to reduce GPU load
                if neural_update_counter % 8 == 0:
                    self.ccpp_robot.update_neural_activity()
                    neural_update_counter = 0

                # Update backtrack list (Algorithm 1) - less frequent
                if algorithm_step_counter % 4 == 0:
                    self.ccpp_robot.update_backtrack_list()

                # ✅ DYNAMIC OBSTACLES: Sync with BWave manager
                if dynamic_positions:
                    self.ccpp_robot.dynamic_obstacle_positions = set((x, y) for x, y in dynamic_positions)
                else:
                    self.ccpp_robot.dynamic_obstacle_positions = set()

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
                            print(f"🔋 Planned return path: {len(return_path) - 1} steps")

                            # Execute return path
                            for i, pos in enumerate(return_path[1:]):  # Skip current position
                                distance = math.sqrt((pos.x - self.ccpp_robot.position.x) ** 2 +
                                                     (pos.y - self.ccpp_robot.position.y) ** 2)

                                self.ccpp_robot.position = pos
                                self.ccpp_robot.path.append(pos)

                                # Update energy for return movement (0.5x cost)
                                self.update_energy_system(distance, is_coverage=False)
                                self.total_travel_length += distance

                                print(
                                    f"  ↳ Return step {i + 1}: ({pos.x}, {pos.y}) - Energy: {self.current_energy:.1f}")

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

                        # Update energy
                        self.update_energy_system(distance, is_coverage=True)
                        self.coverage_length += distance
                        self.total_travel_length += distance

                        # Update BWave UI map for visualization
                        self.ui.map[next_pos.y][next_pos.x] = 'e'  # Mark as explored

                elif self.ccpp_robot.is_deadlock():
                    # Deadlock situation (Algorithm 2)
                    self.deadlock_count += 1
                    print(
                        f"🔴 Deadlock #{self.deadlock_count} detected at {self.ccpp_robot.position.x}, {self.ccpp_robot.position.y}")
                    # ✅ DEBUG INFO: Show neighbor states
                    neighbors = self.ccpp_robot.get_neighbors(self.ccpp_robot.position)
                    print("🔍 Neighbor states:")
                    for i, neighbor in enumerate(neighbors):
                        state = self.ccpp_robot.grid_state[neighbor.y, neighbor.x].item()
                        activity = self.ccpp_robot.neural_activity[neighbor.y, neighbor.x].item()
                        state_names = {0: "UNVISITED", 1: "VISITED", 2: "OBSTACLE", 3: "DEADLOCK"}
                        print(
                            f"  {i}: ({neighbor.x},{neighbor.y}) = {state_names.get(state, 'UNKNOWN')} (act: {activity:.2f})")
                    backtrack_point = self.ccpp_robot.select_best_backtrack_point()
                    if backtrack_point:
                        print(f"🔙 Backtracking to {backtrack_point.x}, {backtrack_point.y}")
                        # Use dynamic A* for backtracking - ALLOW movement through visited cells
                        path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, backtrack_point)
                        if path and len(path) > 1:
                            print(f"📍 Backtrack path length: {len(path) - 1} steps")
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

                                # Update energy for backtrack movement (0.5x cost)
                                self.update_energy_system(distance, is_coverage=False)
                                self.total_travel_length += distance

                                # ✅ IMPORTANT: Don't mark visited cells as explored again during backtracking
                                # Only mark truly unvisited cells
                                if self.ccpp_robot.grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                                    self.ccpp_robot.grid_state[pos.y, pos.x] = GridState.VISITED.value
                                    self.ccpp_robot.external_input[pos.y, pos.x] = 0.0
                                    self.ui.map[pos.y][pos.x] = 'e'
                                    self.coverage_length += distance

                                print(
                                    f"  ↳ Backtrack step {i + 1}: ({pos.x}, {pos.y}) - Energy: {self.current_energy:.1f}")
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

                    # CRITICAL FIX: If stuck but still have unvisited cells
                    print(f"⚠️ Robot stuck at ({self.ccpp_robot.position.x}, {self.ccpp_robot.position.y})")
                    print(f"📊 Unvisited cells remaining: {total_unvisited}")
                    print(f"🎯 Backtrack list size: {len(self.ccpp_robot.backtrack_list)}")

                    # Debug neighbors
                    neighbors = self.ccpp_robot.get_neighbors(self.ccpp_robot.position)
                    print("🔍 Neighbor analysis:")
                    for i, neighbor in enumerate(neighbors):
                        state = self.ccpp_robot.grid_state[neighbor.y, neighbor.x].item()
                        activity = self.ccpp_robot.neural_activity[neighbor.y, neighbor.x].item()
                        state_name = {0: "UNVISITED", 1: "VISITED", 2: "OBSTACLE", 3: "DEADLOCK"}
                        print(
                            f"  Neighbor {i}: ({neighbor.x},{neighbor.y}) - {state_name.get(state, 'UNKNOWN')} - Activity: {activity:.2f}")

                    # Force backtracking if stuck with remaining work
                    if total_unvisited > 0 and self.ccpp_robot.backtrack_list:
                        print("🚨 FORCE BACKTRACKING - Robot stuck but work remains")
                        self.deadlock_count += 1

                        backtrack_point = self.ccpp_robot.select_best_backtrack_point()
                        if backtrack_point:
                            print(f"🔙 Force backtrack to {backtrack_point.x}, {backtrack_point.y}")
                            path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, backtrack_point)
                            if path and len(path) > 1:
                                # Execute backtrack path immediately
                                for pos in path[1:]:
                                    distance = math.sqrt((pos.x - self.ccpp_robot.position.x) ** 2 +
                                                         (pos.y - self.ccpp_robot.position.y) ** 2)

                                    self.ccpp_robot.position = pos
                                    self.ccpp_robot.path.append(pos)
                                    self.update_energy_system(distance, is_coverage=False)
                                    self.total_travel_length += distance

                                    if self.ccpp_robot.grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                                        self.ccpp_robot.grid_state[pos.y, pos.x] = GridState.VISITED.value
                                        self.ccpp_robot.external_input[pos.y, pos.x] = 0.0
                                        self.ui.map[pos.y][pos.x] = 'e'
                                        self.coverage_length += distance

                                print(
                                    f"✅ Force backtrack completed to ({self.ccpp_robot.position.x}, {self.ccpp_robot.position.y})")
                                continue  # Continue the algorithm loop

                        # If still can't backtrack, try to find ANY unvisited cell
                        print("🆘 EMERGENCY: Finding any unvisited cell")
                        for y in range(ROW_COUNT):
                            for x in range(COL_COUNT):
                                if self.ccpp_robot.grid_state[y, x] == GridState.UNVISITED.value:
                                    emergency_target = Position(x, y)
                                    emergency_path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position,
                                                                                    emergency_target)
                                    if emergency_path and len(emergency_path) > 1:
                                        print(f"🚑 Emergency jump to unvisited cell ({x}, {y})")
                                        self.ccpp_robot.position = emergency_target
                                        self.ccpp_robot.path.append(emergency_target)
                                        # Add emergency target to backtrack list
                                        if emergency_target not in self.ccpp_robot.backtrack_list:
                                            self.ccpp_robot.backtrack_list.append(emergency_target)
                                        break
                            else:
                                continue
                            break
                        else:
                            print("❌ No reachable unvisited cells found - terminating")
                            break

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

        # Calculate final metrics
        total_cells = COL_COUNT * ROW_COUNT
        obstacle_cells = len(static_obstacles) + len(dynamic_positions)
        visited_cells = torch.sum(self.ccpp_robot.grid_state == GridState.VISITED.value).item()
        accessible_cells = total_cells - obstacle_cells
        final_coverage_rate = visited_cells / accessible_cells if accessible_cells > 0 else 0

        overlap_rate = ((len(self.ccpp_robot.path) / visited_cells) - 1) * 100 if visited_cells > 0 else 0

        print("\n" + "=" * 80)
        print("🎯 CCPP IN BWAVE ENVIRONMENT - FINAL RESULTS")
        print("=" * 80)
        print(f"📊 Coverage Rate: {final_coverage_rate:.2%}")
        print(f"📏 Total Path Length: {len(self.ccpp_robot.path)}")
        print(f"📈 Overlap Rate: {overlap_rate:.2f}%")
        print(f"🔴 Deadlocks: {self.deadlock_count}")
        print(f"🔋 Charging Returns: {self.return_charge_count}")
        print(f"⏱️  Execution Time: {self.execute_time:.2f}s")
        print(f"🎮 Total Steps: {step}")

        print("\n✅ CCPP Algorithm completed in BWave Environment!")

        # Keep window open for result viewing
        print("🖼️  Press any key to close visualization...")
        waiting = True
        while waiting:
            for event in pg.event.get():
                if event.type == pg.QUIT or event.type == pg.KEYDOWN:
                    waiting = False

        pg.quit()

        return {
            'coverage_rate': final_coverage_rate,
            'path_length': len(self.ccpp_robot.path),
            'overlap_rate': overlap_rate,
            'deadlock_count': self.deadlock_count,
            'return_charge_count': self.return_charge_count,
            'execution_time': self.execute_time,
            'total_steps': step
        }

    def _find_emergency_target(self):
        """Find any reachable unvisited cell for emergency recovery"""
        # Search in expanding radius from current position
        max_radius = max(self.ui.row_count, self.ui.col_count)

        for radius in range(1, max_radius):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) != radius:  # Only check cells at exact radius
                        continue

                    target_x = self.ccpp_robot.position.x + dx
                    target_y = self.ccpp_robot.position.y + dy

                    if (0 <= target_x < self.ccpp_robot.width and 0 <= target_y < self.ccpp_robot.height and
                            self.ccpp_robot.grid_state[target_y, target_x] == GridState.UNVISITED.value):

                        emergency_target = Position(target_x, target_y)
                        # Test if reachable
                        test_path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, emergency_target)
                        if test_path and len(test_path) > 1:
                            return emergency_target

        return None

def main():
    parser = argparse.ArgumentParser(description='CCPP Algorithm in BWave Environment')
    parser.add_argument('--map', type=str, default='map/real_map/denmark.txt',
                        help='Path to map file')
    parser.add_argument('--energy', type=float, default=1000,
                        help='Robot energy capacity')
    parser.add_argument('--speed', type=float, default=0.1,
                        help='Dynamic obstacles speed factor')

    args = parser.parse_args()

    print("🚀 Starting CCPP in BWave Environment...")
    print(f"📁 Map: {args.map}")
    print(f"⚡ Energy: {args.energy}")
    print(f"🏃 Dynamic Speed: {args.speed}")

    ccpp_env = CCPPInBWaveEnvironment()
    results = ccpp_env.run_ccpp_with_bwave_environment(
        map_file=args.map,
        energy_capacity=args.energy,
        dynamic_speed=args.speed
    )

    print("\n🎉 Execution completed successfully!")


if __name__ == "__main__":
    main()