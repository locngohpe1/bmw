import math
import numpy as np
from collections import deque
from a_star_projectB import GridMapGraph, a_star_search


class Q:  # State
    START, NORMAL, DEADLOCK, FINISH = range(4)


neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


class LogicAlgorithm:
    def __init__(self, row_count, col_count):
        self.state = Q.START
        self.weight_map = np.zeros((row_count, col_count))
        self.prob_map = np.zeros((row_count, col_count))
        self.threat_map = np.zeros((row_count, col_count))

        # MCTA Framework Parameters
        self.W1 = 0.5  # Weight for area S1 (close to current position)
        self.W2 = 1.0  # Weight for area S2 (adjacent cells)
        self.W3 = 0.3  # Weight for area S3 (far from current position)

        # Multi-UAV support
        self.uav_flight_mileage = {}
        self.recent_positions = []
        self.loop_detection_window = 8

        # Coverage tracking
        self.visited_cells = set()
        self.coverage_priority = {}

        # Direction for fallback boustrophedon
        self.direction = 4

    def init_weight_map(self, environment):
        row_count, col_count = len(environment), len(environment[0])
        for x, row in enumerate(environment):
            for y, val in enumerate(row):
                if isinstance(val, str):
                    if val in ('o', '1'):
                        self.weight_map[x, y] = 1  # obstacle
                    elif val == 'e':
                        self.weight_map[x, y] = 2  # visited
                    elif val == 'd':
                        self.weight_map[x, y] = 3  # dynamic obstacle
                    else:
                        self.weight_map[x, y] = 0  # free space
                else:
                    self.weight_map[x, y] = int(val)

    def set_map(self, map):
        self.weight_map = map

    def set_prob_map(self, map):
        self.prob_map = map

    def set_threat_map(self, threat_map):
        """Set threat level map for MCTA framework"""
        self.threat_map = threat_map

    def get_adjacent_cells_mcta(self, cur_pos):
        """
        Get 4 adjacent cells for MCTA two-step auction
        Priority order: m1 > m2 > m4 > m3 (up > left > right > down)
        """
        row, col = cur_pos

        adjacent_cells = [
            ((row - 1, col), 1, "up"),  # m1: up (highest priority)
            ((row, col - 1), 2, "left"),  # m2: left
            ((row, col + 1), 4, "right"),  # m4: right
            ((row + 1, col), 3, "down")  # m3: down (lowest priority)
        ]

        valid_cells = []
        for (cell_pos, priority, direction) in adjacent_cells:
            if (0 <= cell_pos[0] < len(self.weight_map) and
                    0 <= cell_pos[1] < len(self.weight_map[0]) and
                    self.weight_map[cell_pos[0], cell_pos[1]] != 1):  # Not static obstacle
                valid_cells.append((cell_pos, priority, direction))

        return valid_cells

    def calculate_threat_level_zeta(self, cell_pos):
        """
        Calculate ζ (zeta) for cell according to MCTA Equation (3)
        ζ = Σ(η × Wd) where η is threat level and Wd is weight for area
        """
        row, col = cell_pos

        if (row < 0 or row >= len(self.prob_map) or
                col < 0 or col >= len(self.prob_map[0])):
            return float('inf')

        # Base threat from current cell
        base_threat = self.prob_map[row, col] / 100.0

        # Calculate weighted threat from surrounding areas (S1, S2, S3)
        surrounding_threat = 0.0

        # Check 3x3 area around cell
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = row + dr, col + dc
                if (0 <= r < len(self.prob_map) and 0 <= c < len(self.prob_map[0])):
                    if (dr, dc) != (0, 0):  # Not current cell
                        # Determine area and corresponding weight
                        distance = max(abs(dr), abs(dc))
                        if distance == 1:
                            if abs(dr) + abs(dc) == 1:  # Direct neighbors
                                weight = self.W2  # S2 area
                            else:  # Diagonal neighbors
                                weight = self.W1  # S1 area
                        else:
                            weight = self.W3  # S3 area (farther)

                        eta = self.prob_map[r, c] / 100.0  # Normalize to [0,1]
                        surrounding_threat += eta * weight

        total_zeta = base_threat + surrounding_threat * 0.2  # Scale surrounding influence
        return total_zeta

    def get_position_score_bonus(self, pos, cur_pos):
        """
        Calculate position-based bonus to break ties and encourage systematic coverage
        """
        # Distance-based exploration (prefer moving away from edges in small areas)
        row, col = pos
        map_height, map_width = len(self.weight_map), len(self.weight_map[0])

        # Prefer positions that lead to more open space
        open_neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = row + dr, col + dc
                if (0 <= r < map_height and 0 <= c < map_width and
                        self.weight_map[r, c] == 0):  # Unvisited
                    open_neighbors += 1

        # Position diversity bonus (prefer positions far from recent history)
        diversity_bonus = 0
        if hasattr(self, 'recent_positions') and len(self.recent_positions) > 0:
            min_dist = min(math.dist(pos, old_pos) for old_pos in self.recent_positions[-4:])
            diversity_bonus = min_dist * 0.1

        # Direction consistency bonus (prefer continuing in same general direction)
        direction_bonus = 0
        if hasattr(self, 'recent_positions') and len(self.recent_positions) >= 2:
            last_pos = self.recent_positions[-1]
            prev_pos = self.recent_positions[-2]

            # Calculate previous direction
            prev_dir = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
            # Calculate current direction
            cur_dir = (pos[0] - cur_pos[0], pos[1] - cur_pos[1])

            # Bonus for continuing in similar direction
            if prev_dir == cur_dir:
                direction_bonus = 0.2

        return open_neighbors * 0.1 + diversity_bonus + direction_bonus

    def loop_check_mechanism(self, cur_pos):
        """
        Enhanced loop detection to prevent 4-cell oscillation
        """
        if not hasattr(self, 'recent_positions'):
            self.recent_positions = []

        # Add current position to history
        self.recent_positions.append(cur_pos)

        # Maintain sliding window
        if len(self.recent_positions) > self.loop_detection_window:
            self.recent_positions.pop(0)

        # Enhanced loop detection
        if len(self.recent_positions) >= 4:
            # Check for 4-cell loop (most common oscillation)
            if (len(self.recent_positions) >= 4 and
                    self.recent_positions[-1] == self.recent_positions[-5:].count(self.recent_positions[-1]) >= 2):
                return True

            # Check for immediate back-and-forth
            if (len(self.recent_positions) >= 3 and
                    self.recent_positions[-1] == self.recent_positions[-3]):
                return True

            # Check for square loop pattern
            if len(self.recent_positions) >= 4:
                last_4 = self.recent_positions[-4:]
                if len(set(last_4)) == 4:  # All different positions in last 4 moves
                    # Check if this pattern repeats
                    if len(self.recent_positions) >= 8:
                        prev_4 = self.recent_positions[-8:-4]
                        if last_4 == prev_4:
                            print(f"4-cell loop detected: {last_4}")
                            return True

        return False

    def mcta_two_step_auction(self, cur_pos, uav_id=0):
        """
        MCTA Two-step Auction Algorithm (Algorithm 1)

        Input: UAV position p, orientation o
        Output: Four cells sorted by bidding price ci and orientation o
        """
        self.state = Q.NORMAL
        adjacent_cells = self.get_adjacent_cells_mcta(cur_pos)

        if len(adjacent_cells) == 0:
            self.state = Q.DEADLOCK
            return []

        bid_values = []

        # Algorithm 1: Two-step Auction
        for cell_pos, priority, direction in adjacent_cells:
            # Step 1: Calculate ζi for current cell i
            zeta_i = self.calculate_threat_level_zeta(cell_pos)

            # Step 2: Assume UAV is in cell i, calculate ζm (max threat of next cells)
            next_adjacent = self.get_adjacent_cells_mcta(cell_pos)
            max_zeta = 0.0

            for next_cell_pos, _, _ in next_adjacent:
                if next_cell_pos != cur_pos:  # Don't count returning to current position
                    next_zeta = self.calculate_threat_level_zeta(next_cell_pos)
                    if next_zeta != float('inf'):
                        max_zeta = max(max_zeta, next_zeta)

            # Step 3: Calculate bid value ci = 1/(ζi + ζm) - Equation (4)
            total_threat = zeta_i + max_zeta

            if total_threat == 0:
                bid_value = 100.0  # Very high value for completely safe areas
            elif zeta_i == float('inf'):
                continue  # Skip inaccessible cells
            else:
                bid_value = 1.0 / (1.0 + total_threat)

            # MCTA Enhancement: Exploration bonus and visit penalty
            cell_state = self.weight_map[cell_pos[0], cell_pos[1]]
            if cell_state == 0:  # Unvisited
                bid_value *= 3.0  # Strong exploration bonus
            elif cell_state == 2:  # Already visited
                bid_value *= 0.1  # Strong penalty for revisiting

            # Enhanced anti-oscillation system
            recent_penalty = 1.0
            if hasattr(self, 'recent_positions') and len(self.recent_positions) > 0:
                # Strong penalty for immediate backtrack
                if len(self.recent_positions) >= 1 and cell_pos == self.recent_positions[-1]:
                    recent_penalty *= 0.01  # Very strong penalty
                # Medium penalty for recent positions
                elif cell_pos in self.recent_positions[-3:]:
                    recent_penalty *= 0.1
                # Light penalty for positions in recent history
                elif cell_pos in self.recent_positions[-6:]:
                    recent_penalty *= 0.5

            # Position-based tie breaking
            position_bonus = self.get_position_score_bonus(cell_pos, cur_pos)

            # Apply all modifiers
            bid_value = bid_value * recent_penalty + position_bonus

            # Add small random factor to break perfect ties
            bid_value += np.random.uniform(0, 0.001)

            bid_values.append((bid_value, priority, cell_pos, direction))

        # Sort by bid value (descending), then by priority (ascending for equal values)
        # This implements the priority: m1 > m2 > m4 > m3
        bid_values.sort(key=lambda x: (-x[0], x[1]))

        return bid_values

    def obstacle_avoidance_mechanism(self, cur_pos, target_cell):
        """
        Obstacle avoidance mechanism based on MCTA heuristic rules
        """
        # Check if target cell is accessible
        if (target_cell[0] < 0 or target_cell[0] >= len(self.weight_map) or
                target_cell[1] < 0 or target_cell[1] >= len(self.weight_map[0])):
            return False

        # Static obstacle check
        if self.weight_map[target_cell[0], target_cell[1]] == 1:
            return False

        # Dynamic obstacle check (high probability areas)
        if hasattr(self, 'prob_map') and self.prob_map[target_cell[0], target_cell[1]] > 85:
            return False

        return True

    def reverse_auction_mechanism(self, conflicting_cells, uav_list):
        """
        Reverse auction mechanism for multi-UAV conflict resolution
        The cell selects UAV with least flight mileage (unfair treatment)
        """
        selected_uavs = {}

        for cell_pos in conflicting_cells:
            min_mileage = float('inf')
            selected_uav = None

            for uav_id in uav_list:
                mileage = self.uav_flight_mileage.get(uav_id, 0)
                if mileage < min_mileage:
                    min_mileage = mileage
                    selected_uav = uav_id

            selected_uavs[cell_pos] = selected_uav

            # Update flight mileage for workload balancing
            if selected_uav is not None:
                self.uav_flight_mileage[selected_uav] = self.uav_flight_mileage.get(selected_uav, 0) + 1

        return selected_uavs

    def energy_constraint_model(self, uav_id, current_energy, target_cell, current_pos):
        """
        Energy constraint model - Equation (5): Bi,k = B - Σl(i,k)
        """
        travel_distance = math.dist(current_pos, target_cell)
        required_energy = travel_distance  # 1 unit energy per cell distance

        return current_energy >= required_energy

    def force_systematic_movement(self, current_pos):
        """
        Force systematic movement to break loops - enhanced boustrophedon
        """
        row_count, col_count = len(self.weight_map), len(self.weight_map[0])
        (x, y) = current_pos

        print(f"Forcing systematic movement from {current_pos}")

        # Try to find unvisited cell in systematic order
        # Priority: right, down, left, up (typical boustrophedon pattern)
        systematic_order = [
            (x, y + 1, "right"),  # Right first
            (x + 1, y, "down"),  # Then down
            (x, y - 1, "left"),  # Then left
            (x - 1, y, "up")  # Finally up
        ]

        for next_x, next_y, direction in systematic_order:
            if (0 <= next_x < row_count and 0 <= next_y < col_count and
                    self.weight_map[next_x, next_y] == 0):  # Unvisited
                print(f"Systematic movement: {direction} to ({next_x}, {next_y})")
                return [(next_x, next_y)]

        # If no unvisited neighbors, try visited cells to escape
        for next_x, next_y, direction in systematic_order:
            if (0 <= next_x < row_count and 0 <= next_y < col_count and
                    self.weight_map[next_x, next_y] != 1):  # Not obstacle
                print(f"Escape movement: {direction} to ({next_x}, {next_y})")
                return [(next_x, next_y)]

        # Last resort: deadlock
        self.state = Q.DEADLOCK
        return []

    def get_wp(self, current_pos, uav_id=0, current_energy=math.inf):
        """
        MCTA Framework Main Algorithm with enhanced loop prevention
        """
        # Strong loop detection
        if self.loop_check_mechanism(current_pos):
            print(f"LOOP DETECTED at {current_pos} - forcing systematic movement")
            # Force systematic movement to break loop
            return self.force_systematic_movement(current_pos)

        # Energy constraint check
        if current_energy <= 1.0:
            self.state = Q.FINISH
            return []

        # Primary Algorithm: MCTA Two-step Auction
        bid_results = self.mcta_two_step_auction(current_pos, uav_id)

        if len(bid_results) == 0:
            print(f"No bid results at {current_pos}")
            self.state = Q.DEADLOCK
            return []

        # Enhanced selection with loop prevention
        for bid_value, priority, cell_pos, direction in bid_results:
            if self.obstacle_avoidance_mechanism(current_pos, cell_pos):
                if self.energy_constraint_model(uav_id, current_energy, cell_pos, current_pos):
                    # Additional check: avoid immediate loops
                    if (hasattr(self, 'recent_positions') and len(self.recent_positions) >= 3 and
                            cell_pos in self.recent_positions[-3:]):
                        print(f"Skipping {cell_pos} due to recent visit")
                        continue

                    print(f"MCTA selected: {cell_pos} (bid: {bid_value:.3f}, dir: {direction})")
                    return [cell_pos]

        # If all auction results rejected, force systematic movement
        print(f"All auction results rejected, forcing systematic movement")
        return self.force_systematic_movement(current_pos)

    def get_replan_wp(self, cur_pos, uav_id=0, current_energy=math.inf):
        """
        Dynamic replanning using MCTA two-step auction
        Used when dynamic obstacles are detected
        """
        bid_results = self.mcta_two_step_auction(cur_pos, uav_id)

        if len(bid_results) == 0:
            return 0, None

        # Return best accessible cell
        for bid_value, priority, cell_pos, direction in bid_results:
            if self.obstacle_avoidance_mechanism(cur_pos, cell_pos):
                return bid_value, cell_pos

        return 0, None

    # Keep existing methods for compatibility
    def four_neighbours(self, cur_pos):
        relative_pos = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        neighbours = []
        for dx, dy in relative_pos:
            x, y = cur_pos[0] + dx, cur_pos[1] + dy
            if x < 0 or x >= len(self.weight_map):
                continue
            if y < 0 or y >= len(self.weight_map[0]):
                continue
            if self.weight_map[x, y] == 1:
                continue
            neighbours.append((x, y))
        return neighbours

    def boustrophedon_moving(self, current_pos):
        """Fallback boustrophedon motion for systematic coverage"""
        row_count, col_count = len(self.weight_map), len(self.weight_map[0])
        (x, y) = current_pos

        # Prioritize vertical movement first, then horizontal
        if (x + 1) < row_count and self.weight_map[x + 1][y] == 0:
            return [(x + 1, y)]
        if (x - 1) >= 0 and self.weight_map[x - 1][y] == 0:
            return [(x - 1, y)]
        if y + 1 < col_count and self.weight_map[x][y + 1] == 0:
            if self.direction == 3:
                return [(x, y + 1)]
        self.direction = 4
        if y - 1 >= 0 and self.weight_map[x][y - 1] == 0:
            return [(x, y - 1)]
        self.direction = 3
        if y + 1 < col_count and self.weight_map[x][y + 1] == 0:
            return [(x, y + 1)]

        return []

    def escape_deadlock_path(self, current_pos):
        """Find path to nearest unvisited cell using BFS"""
        weight_map = self.weight_map
        queue = deque()
        visited = []
        parent = dict()
        deadlock_wp = None
        path = []

        queue.append(current_pos)
        visited.append(current_pos)
        parent[current_pos] = -1

        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        flag = True
        while queue:
            if flag == False:
                break
            cur_node = queue.popleft()

            for dx, dy in neighbors:
                x, y = cur_node[0] + dx, cur_node[1] + dy

                if x < 0 or x >= len(weight_map): continue
                if y < 0 or y >= len(weight_map[0]): continue

                if weight_map[x, y] in (1, 3):
                    continue  # obstacle
                elif weight_map[x, y] == 2 or weight_map[x, y] == 4:  # visited
                    if (x, y) not in visited:
                        visited.append((x, y))
                        queue.append((x, y))
                        parent[x, y] = cur_node
                    continue
                else:
                    deadlock_wp = (x, y)  # unvisited
                    parent[deadlock_wp] = cur_node
                    flag = False
                    break

        if deadlock_wp == None:
            return []

        while parent[deadlock_wp] != -1:
            path.append(deadlock_wp)
            deadlock_wp = parent[deadlock_wp]

        return path[::-1]

    def escape_deadlock_dynamic(self, cur_pos, goal):
        """Dynamic deadlock escape using MCTA principles"""
        bid_value_list = self.mcta_two_step_auction(cur_pos)

        if len(bid_value_list) == 0:
            return 0, None

        # Return best bid result for deadlock escape
        best_bid_value, priority, cell_pos, direction = bid_value_list[0]
        return best_bid_value, cell_pos

    def update_explored(self, pos):
        """Update exploration status"""
        self.weight_map[pos] = 2
        self.visited_cells.add(pos)

    def get_score_max(self, score_dict: dict):
        """Get position with maximum score"""
        if not score_dict:
            return 0, None

        max_score = -math.inf
        max_score_pos = None
        for pos, score in score_dict.items():
            if score > max_score:
                max_score = score
                max_score_pos = pos

        return max_score, max_score_pos

    def predict(self, current_pos, step_count):
        """Predict future waypoints for planning"""
        waypoint_list = [current_pos]
        temporary_visited_list = []

        for _ in range(step_count):
            wp = self.get_wp(current_pos)
            if self.state == Q.FINISH:
                break
            elif self.state == Q.NORMAL:
                if len(wp) > 0:
                    current_pos = wp[0]
                    if self.weight_map[current_pos] == 2:
                        pass  # Already visited
                    temporary_visited_list.append(current_pos)
                    self.weight_map[current_pos] = 2
                    waypoint_list.append(current_pos)
            elif self.state == Q.DEADLOCK:
                break

        # Restore original state
        for pos in temporary_visited_list:
            self.weight_map[pos] = 0

        return waypoint_list

    def get_deadlock_wp(self, current_pos):
        """Find waypoint to escape deadlock"""
        weight_map = self.weight_map
        queue = deque()
        visited = []

        queue.append(current_pos)
        visited.append(current_pos)

        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        while queue:
            cur_node = queue.popleft()
            for dx, dy in neighbors:
                x, y = cur_node[0] + dx, cur_node[1] + dy

                if x < 0 or x >= len(weight_map): continue
                if y < 0 or y >= len(weight_map[0]): continue

                if weight_map[x, y] in (1, 3):
                    continue  # obstacle
                elif weight_map[x, y] == 2 or weight_map[x, y] == 4:  # visited
                    if (x, y) not in visited:
                        visited.append((x, y))
                        queue.append((x, y))
                    continue
                else:
                    return [(x, y)]  # unvisited

        return []


if __name__ == "__main__":
    logic = LogicAlgorithm(7, 7)

    environment = np.zeros((7, 7))
    environment[3, 3] = environment[3, 4] = environment[4, 3] = environment[4, 4] = 1
    logic.init_weight_map(environment)

    print(logic.weight_map)