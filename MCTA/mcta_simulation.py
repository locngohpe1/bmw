# bmw/MCTA/mcta_simulation.py
import math
import numpy as np
import pygame as pg
import time
import threading
import sys
import os

# Import handling for both standalone and launcher execution
try:
    # Try direct import (when run via launcher)
    from grid_map import Grid_Map, EPSILON
    from dynamic_obstacles_manager import DynamicObstaclesManager
    from mcta_uav_robot import MCTAUAVRobot
    from mcta_coordinator import MCTACoordinator
except ImportError:
    # Add paths if running standalone
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    bmw_dir = os.path.dirname(current_dir)
    sys.path.insert(0, bmw_dir)
    sys.path.insert(0, current_dir)

    from grid_map import Grid_Map, EPSILON
    from dynamic_obstacles_manager import DynamicObstaclesManager
    from mcta_uav_robot import MCTAUAVRobot
    from mcta_coordinator import MCTACoordinator


class MCTASimulation:
    """
    Main MCTA Simulation integrating with Project A environment
    """

    def __init__(self, map_file='map/real_map/denmark.txt', num_uavs=4, energy_capacity=1000):
        self.num_uavs = num_uavs
        self.energy_capacity = energy_capacity

        # Fix path resolution - get absolute path from bmw folder
        if not os.path.isabs(map_file):
            # Get bmw folder path (parent of MCTA folder)
            bmw_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            map_file = os.path.join(bmw_folder, map_file)

        print(f"Loading map from: {map_file}")

        # Initialize Project A environment
        self.ui = Grid_Map()

        # Check if we should start with map editing
        if not os.path.exists(map_file):
            print(f"Map file {map_file} not found, starting map editor...")
            self.edit_map_mode()
            return
        else:
            self.environment, self.battery_pos = self.ui.read_map(map_file)

            # Always start with map editing mode for user interaction
            print("🎨 Map loaded. Starting editor for modifications...")
            print("Controls:")
            print("  Left click: Create static obstacle (black)")
            print("  Shift + Left click: Create dynamic obstacle (orange marker)")
            print("  Right click: Set charging station (yellow)")
            print("  Close window: Start MCTA simulation")
            self.environment, self.battery_pos = self.ui.edit_map()
        self.row_count = len(self.environment)
        self.col_count = len(self.environment[0])

        # MCTA Components
        self.coordinator = MCTACoordinator()
        self.uavs = []

        # Dynamic Obstacles (from Project A)
        self.dynamic_obstacles = None

        # Simulation State
        self.run = True
        self.pause = False
        self.fps = 40
        self.last_time = time.time()

        # Performance Metrics
        self.start_time = 0
        self.total_coverage_cells = 0
        self.total_free_cells = np.sum(self.environment == 0)
        self.simulation_complete = False

        # Thread safety
        self._update_lock = threading.Lock()

        print(f"MCTA Simulation initialized:")
        print(f"  Map size: {self.row_count}×{self.col_count}")
        print(f"  UAVs: {num_uavs}")
        print(f"  Battery station: {self.battery_pos}")
        print(f"  Total free cells: {self.total_free_cells}")

    def initialize_uavs(self):
        """
        Initialize UAV fleet at battery position (as per requirements)
        """
        self.uavs = []

        for i in range(self.num_uavs):
            uav_id = f"UAV_{i + 1}"
            uav = MCTAUAVRobot(
                uav_id=uav_id,
                initial_pos=self.battery_pos,
                row_count=self.row_count,
                col_count=self.col_count,
                coordinator=self.coordinator
            )

            # Configure UAV
            uav.set_map(self.environment)
            uav.set_battery_pos(self.battery_pos)
            uav.set_energy_capacity(self.energy_capacity)

            # Register with coordinator
            self.coordinator.register_uav(uav)
            self.uavs.append(uav)

        print(f"Initialized {len(self.uavs)} UAVs at position {self.battery_pos}")

    def initialize_dynamic_obstacles(self, speed_factor=0.1):
        """
        Initialize dynamic obstacles using Project A system
        """
        original_cwd = os.getcwd()
        bmw_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(bmw_folder)

        try:
            self.dynamic_obstacles = DynamicObstaclesManager(
                self.ui,
                num_obstacles=0,
                speed_factor=speed_factor
            )
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

        # Initialize manual obstacles if any exist
        if hasattr(self.ui, 'dynamic_obstacles') and self.ui.dynamic_obstacles:
            for obstacle in self.ui.dynamic_obstacles:
                obstacle['hidden'] = False  # MCTA doesn't use hidden concept
                obstacle['discovered'] = True

            self.dynamic_obstacles.initialize_obstacles()
            print(f"Initialized {len(self.ui.dynamic_obstacles)} dynamic obstacles")

        # Clear visual markers from UI
        self.ui.clear_visual_markers()

    def update_uav_sensor_data(self):
        """
        Update UAV sensor data with detected dynamic obstacles
        """
        if not self.dynamic_obstacles or not self.dynamic_obstacles.obstacles:
            return

        for uav in self.uavs:
            detected_obstacles = {}

            # Check all dynamic obstacles within sensing range
            for obstacle in self.dynamic_obstacles.obstacles:
                obstacle_pos = obstacle['pos']

                if uav.is_within_sensing_range(obstacle_pos):
                    detected_obstacles[obstacle_pos] = {
                        'type': 'dynamic',
                        'threat': 0.8,  # Dynamic obstacles have high but passable threat
                        'id': obstacle['id'],
                        'size': obstacle.get('size', 1.0)
                    }

            # Update UAV knowledge
            uav.update_sensor_data(detected_obstacles)

    def run_simulation(self):
        """
        Main simulation loop
        """
        print("Starting MCTA simulation...")

        clock = pg.time.Clock()
        self.start_time = time.time()

        while self.run:
            # Handle events
            self.handle_events()

            if self.pause:
                clock.tick(self.fps)
                continue

            # Thread-safe updates
            with self._update_lock:
                # Update dynamic obstacles
                current_time = time.time()
                delta_time = current_time - self.last_time
                self.last_time = current_time

                if self.dynamic_obstacles:
                    self.dynamic_obstacles.update(delta_time)

                # Update UAV sensor data
                self.update_uav_sensor_data()

            # MCTA coordination round
            if not self.simulation_complete:
                self.run_mcta_coordination_round()

            # Update display
            self.update_display()

            # Check completion
            if self.coordinator.check_all_uavs_finished() and not self.simulation_complete:
                self.simulation_complete = True
                self.finalize_simulation()

            clock.tick(self.fps)

        pg.quit()

    def run_mcta_coordination_round(self):
        """
        Execute one MCTA coordination round
        """
        # Coordinator manages the auction and conflict resolution
        assignments = self.coordinator.coordinate_auction_round(
            grid_map=self.get_combined_grid_map(),
            known_obstacles=self.get_all_known_obstacles()
        )

        # Execute assignments
        for uav in self.uavs:
            if uav.uav_id in assignments:
                assignment = assignments[uav.uav_id]
                if assignment:
                    uav.set_assignment(assignment)

            # Execute UAV step
            result = uav.run_mcta_step()

            if result == "TASK":
                self.total_coverage_cells += 1

    def get_combined_grid_map(self):
        """
        Get combined grid map with all UAV knowledge
        """
        # For MCTA, each UAV has its own map knowledge
        # Return a combined view for coordinator
        combined_map = {}

        # Start with base environment
        for row in range(self.row_count):
            for col in range(self.col_count):
                pos = (row, col)
                if self.environment[row][col] == 1:
                    combined_map[pos] = 'o'  # Static obstacle
                else:
                    combined_map[pos] = 'u'  # Start as unvisited

        # Update with UAV coverage
        for uav in self.uavs:
            for row in range(self.row_count):
                for col in range(self.col_count):
                    pos = (row, col)
                    if uav.map[pos] == 'e':
                        combined_map[pos] = 'e'  # Covered

        return combined_map

    def get_all_known_obstacles(self):
        """
        Get all known obstacles from all UAVs
        """
        all_obstacles = {}

        for uav in self.uavs:
            all_obstacles.update(uav.known_obstacles)

        return all_obstacles

    def update_display(self):
        """
        Update Pygame display with MCTA visualization
        """
        # Draw base map
        self.ui.draw_map()

        # Draw UAVs with different colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

        for i, uav in enumerate(self.uavs):
            if uav.state != "SLEEP":
                color = colors[i % len(colors)]

                # Draw UAV
                center = (
                    int((uav.current_pos[1] + 0.5) * EPSILON),
                    int((uav.current_pos[0] + 0.5) * EPSILON)
                )
                pg.draw.circle(self.ui.WIN, color, center, EPSILON // 3, 3)

                # Draw UAV ID
                font = pg.font.SysFont(None, 16)
                text = font.render(uav.uav_id, True, color)
                text_rect = text.get_rect(center=(center[0], center[1] - 20))
                self.ui.WIN.blit(text, text_rect)

                # Draw sensor range
                sensor_color = (*color, 30)  # Semi-transparent
                pg.draw.circle(self.ui.WIN, color, center, int(uav.sensing_radius * EPSILON), 1)

        # Draw dynamic obstacles
        if self.dynamic_obstacles:
            self.dynamic_obstacles.draw(self.ui.WIN)

        # Draw info panel
        self.draw_info_panel()

        pg.display.flip()

    def draw_info_panel(self):
        """
        Draw information panel with MCTA metrics
        """
        font = pg.font.SysFont(None, 24)
        y_offset = 10

        # Simulation time
        elapsed_time = time.time() - self.start_time
        time_text = font.render(f"Time: {elapsed_time:.1f}s", True, (255, 255, 255))
        self.ui.WIN.blit(time_text, (10, y_offset))
        y_offset += 25

        # Coverage progress
        if self.total_free_cells > 0:
            coverage_rate = (self.total_coverage_cells / self.total_free_cells) * 100
            coverage_text = font.render(f"Coverage: {coverage_rate:.1f}%", True, (255, 255, 255))
            self.ui.WIN.blit(coverage_text, (10, y_offset))
            y_offset += 25

        # UAV status
        active_uavs = sum(1 for uav in self.uavs if uav.state != "SLEEP")
        status_text = font.render(f"Active UAVs: {active_uavs}/{self.num_uavs}", True, (255, 255, 255))
        self.ui.WIN.blit(status_text, (10, y_offset))
        y_offset += 25

        # Coordination stats
        coord_stats = self.coordinator.get_coordination_stats()
        conflicts_text = font.render(f"Conflicts: {coord_stats['total_conflicts']}", True, (255, 255, 255))
        self.ui.WIN.blit(conflicts_text, (10, y_offset))

    def handle_events(self):
        """
        Handle Pygame events
        """
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.run = False
            elif event.type == pg.KEYDOWN:
                self.pause = not self.pause
                if self.pause:
                    # Create tmp directory if it doesn't exist
                    import os
                    os.makedirs('tmp', exist_ok=True)
                    pg.image.save(self.ui.WIN, 'tmp/mcta_screenshot.png')
                elif event.key == pg.K_LEFT:  # Slow down
                    self.fps = max(5, self.fps // 2)
                    print(f"FPS: {self.fps}")
                elif event.key == pg.K_RIGHT:  # Speed up
                    self.fps = min(120, self.fps * 2)
                    print(f"FPS: {self.fps}")
                elif event.key == pg.K_r:  # Reset simulation
                    self.reset_simulation()

    def reset_simulation(self):
        """
        Reset simulation to initial state
        """
        print("Resetting MCTA simulation...")

        self.simulation_complete = False
        self.total_coverage_cells = 0
        self.start_time = time.time()

        # Reset UAVs
        for uav in self.uavs:
            uav.reset_to_initial_state()
            uav.current_pos = self.battery_pos

        # Reset coordinator
        self.coordinator = MCTACoordinator()
        for uav in self.uavs:
            self.coordinator.register_uav(uav)

    def finalize_simulation(self):
        """
        Finalize simulation and calculate metrics
        """
        execution_time = time.time() - self.start_time

        # Calculate MCTA metrics
        coverage_rate = (self.total_coverage_cells / self.total_free_cells) * 100 if self.total_free_cells > 0 else 0

        # Calculate total path length
        total_path_length = sum(uav.total_flight_mileage for uav in self.uavs)

        # Calculate overlap rate
        overlap_rate = ((
                                    self.total_coverage_cells / self.total_free_cells) - 1) * 100 if self.total_free_cells > 0 else 0

        # Get coordination stats
        coord_stats = self.coordinator.get_coordination_stats()

        print("\n" + "=" * 50)
        print("MCTA SIMULATION RESULTS")
        print("=" * 50)
        print(f"1. Total Path Length: {total_path_length:.2f}")
        print(f"2. Coverage Rate: {coverage_rate:.2f}%")
        print(f"3. Overlap Rate: {overlap_rate:.2f}%")
        print(f"4. Average Flight Deviation: {coord_stats['average_flight_deviation']:.2f}")
        print(f"5. Total Conflicts: {coord_stats['total_conflicts']}")
        print(f"6. Execution Time: {execution_time:.3f}s")
        print(f"7. Active UAVs at End: {coord_stats['active_uavs']}")

        print("\nIndividual UAV Performance:")
        for uav in self.uavs:
            status = uav.get_status()
            print(f"  {status['uav_id']}: {status['flight_mileage']:.1f} distance, {status['task_count']} tasks")

        print("=" * 50)

    def edit_map_mode(self):
        """
        Enter map editing mode (using Project A interface)
        """
        print("Entering map editing mode...")
        print("Controls:")
        print("  Left click: Create static obstacle")
        print("  Shift + Left click: Create dynamic obstacle")
        print("  Right click: Set charging station")
        print("  Close window: Start simulation")

        # Change working directory to bmw folder for edit_map
        import os
        original_cwd = os.getcwd()
        bmw_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(bmw_folder)

        try:
            environment, battery_pos = self.ui.edit_map()
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

        # Update simulation with new map
        self.environment = environment
        self.battery_pos = battery_pos
        self.row_count = len(environment)
        self.col_count = len(environment[0])
        self.total_free_cells = np.sum(environment == 0)

        # Reinitialize UAVs with new map
        self.initialize_uavs()
        self.initialize_dynamic_obstacles()

        print("Map editing complete. Simulation ready.")


def main():
    """
    Main entry point for MCTA simulation
    """
    import argparse

    parser = argparse.ArgumentParser(description='MCTA Multi-UAV Coverage Simulation')
    parser.add_argument('--map', type=str, default='map/real_map/denmark.txt',
                        help='Path to map file (relative to bmw folder)')
    parser.add_argument('--uavs', type=int, default=4, help='Number of UAVs')
    parser.add_argument('--energy', type=float, default=1000, help='Energy capacity per UAV')
    parser.add_argument('--speed', type=float, default=0.1, help='Dynamic obstacle speed factor')
    parser.add_argument('--edit', action='store_true', help='Start in map editing mode')

    args = parser.parse_args()

    print("MCTA Multi-UAV Coverage Path Planning")
    print("=====================================")
    print(f"Map file: {args.map}")
    print(f"Number of UAVs: {args.uavs}")
    print(f"Energy capacity: {args.energy}")
    print(f"Dynamic obstacle speed: {args.speed}")
    print("=====================================")

    try:
        # Initialize simulation
        simulation = MCTASimulation(
            map_file=args.map,
            num_uavs=args.uavs,
            energy_capacity=args.energy
        )

        if args.edit:
            # Map editing mode
            simulation.edit_map_mode()
        else:
            # Initialize with default map
            simulation.initialize_uavs()
            simulation.initialize_dynamic_obstacles(args.speed)

        # Run simulation
        simulation.run_simulation()

    except FileNotFoundError as e:
        print(f"❌ Error: Map file not found!")
        print(f"   Looking for: {args.map}")
        print(f"   Make sure the file exists in bmw/{args.map}")
        print(f"   Available maps in bmw/map/:")

        # Try to list available maps
        bmw_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map_folder = os.path.join(bmw_folder, 'map')
        if os.path.exists(map_folder):
            for root, dirs, files in os.walk(map_folder):
                for file in files:
                    if file.endswith('.txt'):
                        rel_path = os.path.relpath(os.path.join(root, file), bmw_folder)
                        print(f"     {rel_path}")
        else:
            print(f"     Map folder not found at: {map_folder}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()