import math
from collections import defaultdict


class MCTACoordinator:
    """
    Multi-UAV Coordination using Two-Step Auction and Reverse Auction
    Based on MCTA paper conflict resolution mechanism
    """

    def __init__(self):
        self.uavs = {}  # {uav_id: UAV_instance}
        self.auction_results = {}  # Current round auction results
        self.conflict_history = []  # Track conflicts for analysis

    def register_uav(self, uav):
        """Register a UAV in the coordination system"""
        self.uavs[uav.uav_id] = uav
        print(f"Registered UAV {uav.uav_id} in MCTA coordinator")

    def coordinate_auction_round(self, grid_map, known_obstacles):
        """
        Main coordination function - runs one complete auction round
        Returns: {uav_id: assigned_waypoint} or {uav_id: None} for conflicts
        """

        # Step 1: Collect bids from all active UAVs
        all_bids = self.collect_bids(grid_map, known_obstacles)

        if not all_bids:
            return {}

        # Step 2: Detect conflicts (multiple UAVs wanting same cell)
        conflicts = self.detect_conflicts(all_bids)

        # Step 3: Resolve conflicts using reverse auction
        assignments = self.resolve_conflicts(conflicts, all_bids)

        # Step 4: Assign waypoints to UAVs
        final_assignments = self.finalize_assignments(assignments, all_bids)

        return final_assignments

    def collect_bids(self, grid_map, known_obstacles):
        """
        Collect auction bids from all active UAVs
        Returns: {uav_id: [sorted_waypoints]}
        """
        all_bids = {}

        for uav_id, uav in self.uavs.items():
            if uav.state != "SLEEP" and uav.energy > 0:
                # Get waypoints using two-step auction
                waypoints = uav.mcta_logic.two_step_auction(
                    uav.current_pos, grid_map, known_obstacles
                )

                if waypoints:
                    all_bids[uav_id] = waypoints
                    print(f"UAV {uav_id} bids: {waypoints[:2]}")  # Show top 2 choices

        return all_bids

    def detect_conflicts(self, all_bids):
        """
        Detect conflicts where multiple UAVs want the same cell
        Returns: {target_cell: [list_of_competing_uav_ids]}
        """
        cell_to_uavs = defaultdict(list)

        # Group UAVs by their preferred target cell (first choice)
        for uav_id, waypoints in all_bids.items():
            if waypoints:
                preferred_cell = waypoints[0]  # Top priority waypoint
                cell_to_uavs[preferred_cell].append(uav_id)

        # Return only cells with conflicts (multiple UAVs)
        conflicts = {
            cell: uav_list
            for cell, uav_list in cell_to_uavs.items()
            if len(uav_list) > 1
        }

        if conflicts:
            print(f"Conflicts detected: {conflicts}")
            self.conflict_history.append(conflicts)

        return conflicts

    def resolve_conflicts(self, conflicts, all_bids):
        """
        Resolve conflicts using reverse auction mechanism
        Module selects UAV with lowest flight mileage (most "unfair" treatment)
        Returns: {uav_id: assigned_cell}
        """
        assignments = {}

        for target_cell, competing_uavs in conflicts.items():
            # Reverse auction: cell chooses UAV with minimum flight mileage
            winner_uav_id = min(
                competing_uavs,
                key=lambda uav_id: self.uavs[uav_id].total_flight_mileage
            )

            assignments[winner_uav_id] = target_cell

            # Other UAVs must wait one time step
            for uav_id in competing_uavs:
                if uav_id != winner_uav_id:
                    self.uavs[uav_id].wait_one_step()
                    print(f"UAV {uav_id} waits (lost conflict to UAV {winner_uav_id})")

            print(
                f"Conflict at {target_cell}: UAV {winner_uav_id} wins (mileage: {self.uavs[winner_uav_id].total_flight_mileage:.1f})")

        return assignments

    def finalize_assignments(self, conflict_assignments, all_bids):
        """
        Finalize assignments for all UAVs, including non-conflicted ones
        Returns: {uav_id: assigned_waypoint or None}
        """
        final_assignments = {}

        # NEW (CORRECT):
        # Track all assigned cells to avoid double assignment
        all_assigned_cells = set(conflict_assignments.values())

        for uav_id, waypoints in all_bids.items():
            if uav_id in conflict_assignments:
                # UAV won a conflict
                final_assignments[uav_id] = conflict_assignments[uav_id]
            elif waypoints:
                # No conflict - try to assign best available waypoint
                assigned = False
                for waypoint in waypoints:
                    if waypoint not in all_assigned_cells:
                        final_assignments[uav_id] = waypoint
                        all_assigned_cells.add(waypoint)
                        assigned = True
                        break

                if not assigned:
                    final_assignments[uav_id] = None  # No valid assignment
            else:
                final_assignments[uav_id] = None  # No waypoints available

        return final_assignments

    def get_coordination_stats(self):
        """
        Get statistics about coordination performance
        """
        total_conflicts = len(self.conflict_history)
        total_conflict_instances = sum(
            len(competing_uavs) - 1  # -1 because winner doesn't count as conflict
            for conflicts in self.conflict_history
            for competing_uavs in conflicts.values()
        )

        active_uavs = sum(1 for uav in self.uavs.values() if uav.state != "SLEEP")

        # Calculate workload balance (flight mileage deviation)
        mileages = [uav.total_flight_mileage for uav in self.uavs.values()]
        if mileages:
            avg_mileage = sum(mileages) / len(mileages)
            avg_deviation = sum(abs(m - avg_mileage) for m in mileages) / len(mileages)
        else:
            avg_deviation = 0

        return {
            'total_conflicts': total_conflicts,
            'total_conflict_instances': total_conflict_instances,
            'active_uavs': active_uavs,
            'average_flight_deviation': avg_deviation,
            'individual_mileages': {uav_id: uav.total_flight_mileage for uav_id, uav in self.uavs.items()}
        }

    def emergency_coordination(self, emergency_uav_id):
        """
        Handle emergency situations (e.g., low battery, critical failure)
        Give priority to emergency UAV in next auction round
        """
        if emergency_uav_id in self.uavs:
            emergency_uav = self.uavs[emergency_uav_id]

            # Temporarily boost emergency UAV priority by reducing its flight mileage for conflict resolution
            original_mileage = emergency_uav.total_flight_mileage
            emergency_uav.total_flight_mileage = 0  # Highest priority

            print(f"Emergency priority given to UAV {emergency_uav_id}")

            # Restore original mileage after next coordination round
            return original_mileage

        return None

    def check_all_uavs_finished(self):
        """
        Check if all UAVs have finished their coverage tasks
        """
        return all(uav.state == "SLEEP" for uav in self.uavs.values())

    def get_fleet_status(self):
        """
        Get current status of entire UAV fleet
        """
        status = {}
        for uav_id, uav in self.uavs.items():
            status[uav_id] = {
                'position': uav.current_pos,
                'state': uav.state,
                'energy': uav.energy,
                'flight_mileage': uav.total_flight_mileage,
                'waiting': getattr(uav, 'waiting_steps', 0)
            }
        return status

    def optimize_fleet_coordination(self):
        """
        Optimize fleet coordination by analyzing patterns and adjusting strategies
        """
        stats = self.get_coordination_stats()

        # If too many conflicts, suggest spreading UAVs more
        if stats['total_conflicts'] > len(self.uavs) * 2:
            print("High conflict rate detected - consider spreading UAV starting positions")

        # If workload very unbalanced, adjust flight mileage calculation
        if stats['average_flight_deviation'] > 50:
            print("Workload imbalance detected - consider dynamic mileage adjustment")

        return stats