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

        # Set robot starting position (convert BWave (row,col) to CCPP (x,y))
        start_x, start_y = battery_pos[1], battery_pos[0]
        self.ccpp_robot.position = Position(start_x, start_y)
        self.ccpp_robot.path = [self.ccpp_robot.position]

        # Set energy system
        self.energy_capacity = energy_capacity
        self.current_energy = energy_capacity

        print(f"🤖 CCPP Robot initialized at ({start_x}, {start_y})")

        # 4. Main Algorithm Loop với BWave Visualization
        print("\n🚀 Starting CCPP Coverage Algorithm...")
        print("Press SPACE to pause, LEFT/RIGHT arrows to change speed, ESC to quit")

        FPS = 10
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

            # Get current dynamic obstacle positions for CCPP
            dynamic_positions = self.get_dynamic_obstacle_positions()

            # Update CCPP robot with dynamic obstacles
            for pos_x, pos_y in dynamic_positions:
                self.ccpp_robot.add_dynamic_obstacle(pos_x, pos_y)

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
                        FPS = min(60, FPS * 2)
                        print(f"🏃 Speed: {FPS} FPS")
                    elif event.key == pg.K_ESCAPE:
                        run = False

            if pause:
                clock.tick(FPS)
                continue

            # 5. CCPP Algorithm Step (Project D)
            if not pause:
                # Update neural activities
                self.ccpp_robot.update_neural_activity()

                # Update backtrack list (Algorithm 1)
                self.ccpp_robot.update_backtrack_list()

                # Try normal movement (Priority template)
                next_pos = self.ccpp_robot.select_next_position_with_priority()

                if next_pos is not None:
                    # Check energy before movement
                    distance = math.sqrt((next_pos.x - self.ccpp_robot.position.x) ** 2 +
                                         (next_pos.y - self.ccpp_robot.position.y) ** 2)

                    if not self.check_energy_for_return(next_pos, battery_pos):
                        # Need to return for charging
                        print("⚡ Low energy! Returning to charge...")
                        self.ccpp_robot.position = Position(battery_pos[1], battery_pos[0])
                        self.ccpp_robot.path.append(self.ccpp_robot.position)
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
                    print(f"🔴 Deadlock #{self.deadlock_count} detected!")

                    backtrack_point = self.ccpp_robot.select_best_backtrack_point()
                    if backtrack_point:
                        # Use dynamic A* for backtracking
                        path = self.ccpp_robot.dynamic_a_star(self.ccpp_robot.position, backtrack_point)
                        if path and len(path) > 1:
                            for pos in path[1:]:
                                self.ccpp_robot.position = pos
                                self.ccpp_robot.path.append(pos)
                                if self.ccpp_robot.grid_state[pos.y, pos.x] == GridState.UNVISITED.value:
                                    self.ccpp_robot.grid_state[pos.y, pos.x] = GridState.VISITED.value
                                    self.ccpp_robot.external_input[pos.y, pos.x] = 0.0
                                    self.ui.map[pos.y][pos.x] = 'e'
                else:
                    # Check if coverage complete
                    total_unvisited = torch.sum(self.ccpp_robot.grid_state == GridState.UNVISITED.value).item()
                    if total_unvisited == 0:
                        print("✅ Coverage Complete!")
                        break

                step += 1

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

            # Status display
            font = pg.font.SysFont(None, 24)
            status_text = f"Step: {step} | Deadlocks: {self.deadlock_count} | Energy: {self.current_energy:.1f}"
            status_surface = font.render(status_text, True, (255, 255, 255))
            self.ui.WIN.blit(status_surface, (10, 10))

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