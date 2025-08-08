# bmw/MCTA/mcta_logic.py
import math
import numpy as np
from collections import deque
import random


class Q:  # State Management
    NORMAL, DEADLOCK, FINISH = range(3)


class MCTALogic:
    """
    Core MCTA Logic implementing Two-Step Auction Algorithm
    Based on paper: "Multi-UAV Coverage through Two-Step Auction in Dynamic Environments"
    """

    def __init__(self, uav_id, sensing_radius=10):
        self.uav_id = uav_id
        self.sensing_radius = sensing_radius
        self.state = Q.NORMAL

        # Weight areas for threat calculation (W2 > W1 > W3)
        self.weight_areas = {
            'S1': 1,  # Current module cells close to adjacent module
            'S2': 3,  # Adjacent module cells closest to current module center
            'S3': 1  # Adjacent module cells far from current module center
        }

        # Module priority when bid values are equal (m1 > m2 > m4 > m3)
        self.module_priority = {1: 4, 2: 3, 4: 2, 3: 1}  # Higher value = higher priority

        # Cache for deadlock escape
        self.cache_path = []
        self.cache_dist = 0

    def get_wp(self, current_pos, grid_map, known_obstacles):
        """
        Main waypoint selection using Two-Step Auction
        Algorithm 2 from MCTA paper - adapted for individual cells
        """

        # Step 1: Get valid neighbors (Set D)
        set_D = self.get_set_D(current_pos, grid_map)
        wp = []

        if len(set_D) == 0:
            # Deadlock - use Wavefront escape
            self.state = Q.DEADLOCK
            wp = self.get_local_extreme_wp(current_pos, grid_map)
            if len(wp) == 0:
                self.state = Q.FINISH
            return wp

        # Step 2: Use two-step auction to rank all possible moves
        self.state = Q.NORMAL
        ranked_waypoints = self.two_step_auction(current_pos, grid_map, known_obstacles)

        # Filter out invalid waypoints and return valid ones
        valid_wp = []
        for waypoint in ranked_waypoints:
            if waypoint in set_D:  # Must be a valid neighbor
                valid_wp.append(waypoint)

        # If no valid waypoints from auction, use fallback
        if not valid_wp:
            valid_wp = self.max_potential_cells(set_D, grid_map, known_obstacles)

        return valid_wp[:1] if valid_wp else []  # Return only best choice to reduce conflicts
    def two_step_auction(self, current_pos, grid_map, known_obstacles):
        """
        Algorithm 1: Two-Step Auction from MCTA paper
        Returns: Four modules sorted by bidding price ci and orientation
        """

        # Get 4 adjacent cells (treating each as module)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        modules = []

        for i, (dx, dy) in enumerate(directions):
            module_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if self.is_valid_and_reachable(module_pos, grid_map):
                # Calculate bidding value
                bid_value = self.calculate_bid_value(module_pos, current_pos, known_obstacles)
                modules.append({
                    'pos': module_pos,
                    'bid_value': bid_value,
                    'direction': i + 1,  # 1=up, 2=right, 3=down, 4=left
                    'priority': self.module_priority[i + 1]
                })

        # Sort by bid value (descending), then by priority (descending)
        modules.sort(key=lambda x: (-x['bid_value'], -x['priority']))

        return [m['pos'] for m in modules]

    def calculate_bid_value(self, module_pos, current_pos, known_obstacles):
        """
        Equation (4): ci = 1/(ζi + ζm)
        """
        # Calculate ζi for module_pos
        zeta_i = self.calculate_threat_level(module_pos, known_obstacles)

        # Assume UAV is in module_pos, calculate ζm
        zeta_m = self.calculate_max_future_threat(module_pos, known_obstacles)

        # Avoid division by zero
        denominator = max(zeta_i + zeta_m, 1e-6)
        return 1.0 / denominator

    def calculate_threat_level(self, pos, known_obstacles):
        """
        Equation (3): ζ = Σ(η × Wd)
        Calculate threat level considering surrounding areas S1, S2, S3
        """
        threat_sum = 0.0

        # For simplicity, consider 8-neighborhood as different areas
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for i, (dx, dy) in enumerate(neighbors):
            neighbor_pos = (pos[0] + dx, pos[1] + dy)

            # Get threat level η for this cell
            eta = self.get_threat_level_eta(neighbor_pos, known_obstacles)

            # Assign weight based on relative position (simplified area assignment)
            if i in [1, 6]:  # Direct vertical/horizontal neighbors (S2 - high weight)
                weight = self.weight_areas['S2']
            elif i in [0, 2, 5, 7]:  # Diagonal neighbors (S1 - medium weight)
                weight = self.weight_areas['S1']
            else:  # Other positions (S3 - low weight)
                weight = self.weight_areas['S3']

            threat_sum += eta * weight

        return threat_sum

    def calculate_max_future_threat(self, assumed_pos, known_obstacles):
        """
        Calculate maximum threat from assumed position looking at 3 other directions
        (excluding the direction we came from)
        """
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        max_threat = 0.0

        for dx, dy in directions:
            future_pos = (assumed_pos[0] + dx, assumed_pos[1] + dy)
            threat = self.calculate_threat_level(future_pos, known_obstacles)
            max_threat = max(max_threat, threat)

        return max_threat

    def get_threat_level_eta(self, pos, known_obstacles):
        """
        Get threat level η ∈ [0,1] for a specific cell
        0 = safe, 1 = impassable obstacle
        """
        # Check if position is valid
        if not self.is_valid_pos(pos):
            return 1.0  # Out of bounds = impassable

        # Check static obstacles (from map)
        if pos in known_obstacles:
            obstacle_info = known_obstacles[pos]
            if obstacle_info['type'] == 'static':
                return 1.0  # Static obstacle = impassable
            elif obstacle_info['type'] == 'dynamic':
                # Dynamic obstacles have variable threat based on movement
                return 0.8  # High threat but passable

        return 0.0  # Free space = safe

    def get_set_D(self, current_pos, grid_map):
        """
        Get valid neighboring cells that are unvisited
        """
        neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        set_D = []

        for dx, dy in neighbors:
            neighbor = (current_pos[0] + dx, current_pos[1] + dy)

            if (self.is_valid_pos(neighbor) and
                    grid_map[neighbor] == 'u'):  # Unvisited
                set_D.append(neighbor)

        return set_D

    def both_vertical_neighbors_exist(self, pos, set_D):
        """
        Check if both vertical neighbors (up and down) exist in set_D
        """
        up_neighbor = (pos[0] - 1, pos[1])
        down_neighbor = (pos[0] + 1, pos[1])
        return up_neighbor in set_D and down_neighbor in set_D

    def calculate_border_distances(self, pos, grid_map):
        """
        Calculate distances to upper and lower borders
        """
        row, col = pos

        # Distance to upper border (counting consecutive free cells)
        up_dist = 0
        for r in range(row - 1, -1, -1):
            if grid_map[(r, col)] == 'u':
                up_dist += 1
            else:
                break

        # Distance to lower border
        down_dist = 0
        for r in range(row + 1, len(grid_map)):
            if grid_map[(r, col)] == 'u':
                down_dist += 1
            else:
                break

        return up_dist, down_dist

    def max_potential_cells(self, cell_list, grid_map, known_obstacles):
        """
        Find cells with maximum threat potential (minimum threat level)
        Lower threat = higher potential for coverage
        """
        if not cell_list:
            return []

        max_list = []
        min_threat = float('inf')

        for cell in cell_list:
            threat = self.calculate_threat_level(cell, known_obstacles)

            if threat < min_threat:
                min_threat = threat
                max_list.clear()
                max_list.append(cell)
            elif abs(threat - min_threat) < 1e-6:
                max_list.append(cell)

        return max_list

    def next_to_obstacle(self, cell, grid_map):
        """
        Check if cell is adjacent to any obstacle (for trap region detection)
        """
        neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

        for dx, dy in neighbors:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if (self.is_valid_pos(neighbor) and
                    grid_map.get(neighbor, 'o') in ('o', 1)):  # Static obstacle
                return True
        return False

    def get_local_extreme_wp(self, current_pos, grid_map):
        """
        Deadlock escape using Wavefront algorithm
        Find nearest unvisited cell using shortest path
        """
        return_matrix = np.full(grid_map.shape, None, dtype=object)
        for x in range(return_matrix.shape[0]):
            for y in range(return_matrix.shape[1]):
                return_matrix[x, y] = [None, math.inf]

        queue = deque()
        visited_matrix = np.zeros(grid_map.shape, dtype=bool)
        candidate_results = []
        stop_depth = -1

        queue.append((current_pos, 0))
        visited_matrix[current_pos] = True
        return_matrix[current_pos] = [None, 0]

        while queue:
            cur_node, cur_depth = queue.popleft()

            if stop_depth != -1 and cur_depth != stop_depth:
                if candidate_results:
                    escape_wp = min(candidate_results, key=lambda x: return_matrix[x][1])
                    self.cache_path = self.get_wavefront_path(return_matrix, escape_wp)
                    self.cache_dist = return_matrix[escape_wp][1]
                    return [escape_wp]
                break

            # Check if current node is unvisited
            if grid_map[cur_node] == 'u':
                if stop_depth == -1:
                    stop_depth = cur_depth
                if cur_node not in candidate_results:
                    candidate_results.append(cur_node)

            if stop_depth != -1:
                continue

            # Explore neighbors
            neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
            for dx, dy in neighbors:
                neighbor = (cur_node[0] + dx, cur_node[1] + dy)

                if not self.is_valid_pos(neighbor):
                    continue
                if grid_map[neighbor] in ('o', 1):  # Static obstacle
                    continue
                if grid_map[neighbor] == 'd':  # Dynamic obstacle
                    continue

                new_dist = return_matrix[cur_node][1] + math.dist(cur_node, neighbor)
                if new_dist < return_matrix[neighbor][1]:
                    return_matrix[neighbor][0] = cur_node
                    return_matrix[neighbor][1] = new_dist

                if not visited_matrix[neighbor]:
                    visited_matrix[neighbor] = True
                    queue.append((neighbor, cur_depth + 1))

        if candidate_results:
            escape_wp = min(candidate_results, key=lambda x: return_matrix[x][1])
            self.cache_path = self.get_wavefront_path(return_matrix, escape_wp)
            self.cache_dist = return_matrix[escape_wp][1]
            return [escape_wp]

        return []

    def get_wavefront_path(self, return_matrix, target_pos):
        """
        Reconstruct path from return matrix
        """
        path = []
        current = target_pos

        while current is not None:
            path.append(current)
            current = return_matrix[current][0]

        return list(reversed(path))

    def is_valid_pos(self, pos):
        """
        Check if position is within grid bounds
        """
        # This will be set by the parent UAV class with actual grid dimensions
        return True  # Placeholder - will be overridden

    def is_valid_and_reachable(self, pos, grid_map):
        """
        Check if position is valid and reachable (not blocked by static obstacles)
        """
        if not self.is_valid_pos(pos):
            return False
        return grid_map.get(pos, 'o') not in ('o', 1)  # Not a static obstacle