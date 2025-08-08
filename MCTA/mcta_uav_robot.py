# bmw/MCTA/mcta_uav_robot.py
import math
import numpy as np
import pygame as pg
import time
import sys
import os

# Import handling for both standalone and launcher execution
try:
    # Try direct import (when run via launcher)
    from mcta_logic import MCTALogic, Q
except ImportError:
    # Add paths if running standalone
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    bmw_dir = os.path.dirname(current_dir)
    sys.path.insert(0, bmw_dir)
    sys.path.insert(0, current_dir)
    from mcta_logic import MCTALogic, Q

# Import Project A components
try:
    from a_star import GridMapGraph, a_star_search
except ImportError:
    print("⚠️  a_star.py not found - some features may be limited")


class MCTAUAVRobot:
    """
    MCTA UAV Robot implementation
    Integrates MCTA logic with Project A's environment and energy system
    """

    def __init__(self, uav_id, initial_pos, row_count, col_count, coordinator=None):
        self.uav_id = uav_id
        self.current_pos = initial_pos
        self.row_count = row_count
        self.col_count = col_count
        self.coordinator = coordinator

        # MCTA Core Logic
        self.mcta_logic = MCTALogic(uav_id, sensing_radius=10)
        self.mcta_logic.is_valid_pos = self.is_valid_pos  # Bind method

        # Robot State
        self.state = "NORMAL"  # NORMAL, DEADLOCK, SLEEP
        self.angle = math.pi / 2  # Initial direction (up)

        # Energy System (compatible with Project A)
        self.energy = 1000.0  # Will be set by main system
        self.energy_capacity = 1000.0

        # Flight Tracking (for MCTA coordination)
        self.total_flight_mileage = 0.0

        # Map Knowledge (MCTA UAVs know structure but not dynamic obstacles)
        self.map = None  # Will be set by environment
        self.known_obstacles = {}  # {pos: {'type': 'static'/'dynamic', 'threat': float}}

        # Sensor System
        self.sensing_radius = 10  # 10D radius as per MCTA paper
        self.sensor_active = True

        # Waiting System (for conflict resolution)
        self.waiting_steps = 0
        self.max_wait_steps = 3

        # Performance Tracking
        self.move_count = 0
        self.task_count = 0
        self.deadlock_count = 0

        # Charging System (reuse from Project A)
        self.battery_pos = initial_pos  # Will be updated
        self.move_status = 0  # 0: coverage, 1: retreat, 2: charge, 3: advance
        self.cache_path = []

    def set_map(self, environment):
        """
        Set the environment map (known structure, unknown dynamic obstacles)
        """
        self.map = np.full((self.row_count, self.col_count), 'u', dtype=object)

        # Set known static obstacles and free space
        for x in range(len(environment)):
            for y in range(len(environment[0])):
                if environment[x][y] == 1:
                    self.map[x, y] = 'o'  # Static obstacle
                    self.known_obstacles[(x, y)] = {'type': 'static', 'threat': 1.0}
                else:
                    self.map[x, y] = 'u'  # Unvisited

    def set_battery_pos(self, battery_pos):
        """Set charging station position"""
        self.battery_pos = battery_pos

    def set_energy_capacity(self, capacity):
        """Set energy capacity"""
        self.energy_capacity = capacity
        self.energy = capacity

    def update_sensor_data(self, detected_obstacles):
        """
        Update known obstacles from sensor detection
        Only called when obstacles are within sensing radius
        """
        for pos, obstacle_info in detected_obstacles.items():
            if self.is_within_sensing_range(pos):
                self.known_obstacles[pos] = obstacle_info

                # Update map if it's a new dynamic obstacle
                if obstacle_info['type'] == 'dynamic' and self.map[pos] not in ['o', 'e']:
                    self.map[pos] = 'd'
                    print(f"UAV {self.uav_id} detected dynamic obstacle at {pos}")

    def is_within_sensing_range(self, pos):
        """
        Check if position is within sensing radius (10D as per MCTA paper)
        """
        distance = math.sqrt(
            (pos[0] - self.current_pos[0]) ** 2 +
            (pos[1] - self.current_pos[1]) ** 2
        )
        return distance <= self.sensing_radius

    def get_sensing_scope_cells(self):
        """
        Get all cells within sensing radius
        Returns list of positions that UAV can currently sense
        """
        cells = []
        for dx in range(-self.sensing_radius, self.sensing_radius + 1):
            for dy in range(-self.sensing_radius, self.sensing_radius + 1):
                if dx * dx + dy * dy <= self.sensing_radius * self.sensing_radius:
                    cell = (self.current_pos[0] + dx, self.current_pos[1] + dy)
                    if self.is_valid_pos(cell):
                        cells.append(cell)
        return cells

    def run_mcta_step(self):
        """
        Execute one MCTA step (to be called from main simulation loop)
        """
        if self.state == "SLEEP":
            return None

        # Handle waiting from conflict resolution
        if self.waiting_steps > 0:
            self.waiting_steps -= 1
            print(f"UAV {self.uav_id} waiting ({self.waiting_steps} steps remaining)")
            return None

        # Energy check
        if not self.check_energy_sufficiency():
            print(f"UAV {self.uav_id} needs to charge")
            self.initiate_charging_sequence()
            return None

        # Get waypoint assignment from coordinator
        if self.coordinator:
            # This will be called by coordinator during auction round
            assignment = self.get_current_assignment()
            if assignment:
                return self.execute_movement(assignment)

        # Fallback: individual waypoint selection
        wp_list = self.mcta_logic.get_wp(self.current_pos, self.map, self.known_obstacles)

        if not wp_list:
            self.state = "SLEEP"
            print(f"UAV {self.uav_id} finished coverage - entering sleep mode")
            return None

        # Select best waypoint
        selected_wp = self.select_waypoint(wp_list)
        return self.execute_movement(selected_wp)

    def select_waypoint(self, wp_list):
        """
        Select best waypoint from list considering energy and obstacles
        """
        if len(wp_list) == 1:
            return wp_list[0]

        # Filter out waypoints that would cause energy problems
        valid_wps = [wp for wp in wp_list if self.check_energy_for_waypoint(wp)]

        if not valid_wps:
            return wp_list[0]  # Take first even if energy is tight

        # Select waypoint with minimum travel cost
        return min(valid_wps, key=self.calculate_travel_cost)

    def calculate_travel_cost(self, waypoint):
        """
        Calculate travel cost including distance and turning
        """
        distance = math.dist(self.current_pos, waypoint)

        # Calculate turning cost
        target_angle = self.get_angle_to(waypoint)
        turn_angle = abs(self.angle - target_angle)
        if turn_angle > math.pi:
            turn_angle = 2 * math.pi - turn_angle

        # Total cost: distance + turning penalty
        return 2 * distance + 1 * turn_angle

    def execute_movement(self, target_pos):
        """
        Execute movement to target position with energy consumption
        """
        if target_pos == self.current_pos:
            # Task current position
            self.task_current_cell()
            return "TASK"

        # Check for obstacles in target position
        if self.is_obstacle_at(target_pos):
            print(f"UAV {self.uav_id} detected obstacle at target {target_pos}")
            return "BLOCKED"

        # Calculate energy cost
        distance = math.dist(self.current_pos, target_pos)
        energy_cost = distance  # 1 energy per unit distance for coverage

        if self.move_status in [1, 3]:  # Retreat or advance
            energy_cost = 0.5 * distance

        # Execute movement
        if self.energy >= energy_cost:
            self.energy -= energy_cost
            self.total_flight_mileage += distance
            self.move_count += 1

            # Update position and orientation
            self.angle = self.get_angle_to(target_pos)
            self.current_pos = target_pos

            # Task if unvisited
            if self.map[target_pos] == 'u':
                self.task_current_cell()

            print(f"UAV {self.uav_id} moved to {target_pos} (energy: {self.energy:.1f})")
            return "MOVED"
        else:
            print(f"UAV {self.uav_id} insufficient energy for movement")
            return "LOW_ENERGY"

    def task_current_cell(self):
        """
        Task/cover the current cell
        """
        if self.map[self.current_pos] == 'u':
            self.map[self.current_pos] = 'e'
            self.task_count += 1
            print(f"UAV {self.uav_id} tasked cell {self.current_pos}")

    def wait_one_step(self):
        """
        Set UAV to wait one step (called by coordinator for conflict resolution)
        """
        self.waiting_steps = max(self.waiting_steps, 1)

    def check_energy_sufficiency(self):
        """
        Check if UAV has enough energy to continue operations
        """
        # Estimate energy needed to return to charging station
        return_distance = math.dist(self.current_pos, self.battery_pos)
        return_energy = 0.5 * return_distance  # Return uses half energy

        # Keep some buffer for next move
        buffer_energy = 10.0

        return self.energy > (return_energy + buffer_energy)

    def check_energy_for_waypoint(self, waypoint):
        """
        Check if UAV has enough energy to reach waypoint and return
        """
        move_distance = math.dist(self.current_pos, waypoint)
        return_distance = math.dist(waypoint, self.battery_pos)

        total_energy_needed = move_distance + 0.5 * return_distance + 5.0  # 5.0 buffer

        return self.energy >= total_energy_needed

    def initiate_charging_sequence(self):
        """
        Initiate charging sequence (retreat → charge → advance)
        """
        self.move_status = 1  # Retreat mode
        # Implementation would use A* to find path back to charging station
        # For now, simplified
        print(f"UAV {self.uav_id} initiating charging sequence")

    def is_obstacle_at(self, pos):
        """
        Check if there's an obstacle at given position
        """
        if not self.is_valid_pos(pos):
            return True

        # Check map
        if self.map[pos] in ['o', 'd']:
            return True

        # Check known obstacles
        if pos in self.known_obstacles:
            return self.known_obstacles[pos]['threat'] >= 1.0

        return False

    def is_valid_pos(self, pos):
        """
        Check if position is within grid bounds
        """
        row, col = pos
        return 0 <= row < self.row_count and 0 <= col < self.col_count

    def get_angle_to(self, target_pos):
        """
        Calculate angle to target position
        """
        dx = target_pos[1] - self.current_pos[1]
        dy = target_pos[0] - self.current_pos[0]
        return math.atan2(dy, dx)

    def get_current_assignment(self):
        """
        Get current waypoint assignment from coordinator
        """
        # This will be set by coordinator during auction round
        return getattr(self, '_current_assignment', None)

    def set_assignment(self, waypoint):
        """
        Set current waypoint assignment (called by coordinator)
        """
        self._current_assignment = waypoint

    def get_status(self):
        """
        Get comprehensive status for monitoring
        """
        return {
            'uav_id': self.uav_id,
            'position': self.current_pos,
            'state': self.state,
            'energy': self.energy,
            'flight_mileage': self.total_flight_mileage,
            'move_count': self.move_count,
            'task_count': self.task_count,
            'waiting_steps': self.waiting_steps,
            'sensing_radius': self.sensing_radius,
            'known_obstacles': len(self.known_obstacles)
        }

    def draw_sensor_range(self, surface, epsilon):
        """
        Draw sensor range as circle on the display
        """
        if self.sensor_active:
            center_x = int((self.current_pos[1] + 0.5) * epsilon)
            center_y = int((self.current_pos[0] + 0.5) * epsilon)
            radius = int(self.sensing_radius * epsilon)

            # Draw semi-transparent circle
            sensor_color = (0, 255, 255, 50)  # Cyan with transparency
            pg.draw.circle(surface, sensor_color[:3], (center_x, center_y), radius, 2)

    def reset_to_initial_state(self):
        """
        Reset UAV to initial state for new simulation
        """
        self.energy = self.energy_capacity
        self.total_flight_mileage = 0.0
        self.move_count = 0
        self.task_count = 0
        self.deadlock_count = 0
        self.waiting_steps = 0
        self.state = "NORMAL"
        self.known_obstacles = {}

        # Reset map to unvisited (keep static obstacles)
        for pos in np.ndindex(self.map.shape):
            if self.map[pos] not in ['o']:  # Keep static obstacles
                self.map[pos] = 'u'