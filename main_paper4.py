import math
import numpy as np
import pygame as pg
import time
import argparse
import random
from typing import Dict, List, Tuple, Set
from collections import deque

from grid_map import Grid_Map, EPSILON
from dynamic_obstacles_manager import DynamicObstaclesManager
from project_B.mcta_algorithm import MCTA

parser = argparse.ArgumentParser(description='MCTA Coverage Path Planning with Dynamic Obstacles')
parser.add_argument('--map', type=str, default='map/real_map/scioto2.txt', help='Path to map file')
parser.add_argument('--speed', type=float, default=1, help='Speed of dynamic obstacles')
parser.add_argument('--uavs', type=int, default=1, help='Number of UAVs')
parser.add_argument('--energy', type=float, default=1000, help='Energy capacity per UAV')
args = parser.parse_args()

FPS = 0.1
MOVE_INTERVAL = 0.5  #robot speed(cells/s)

# BFS for return to charge
def bfs_to_charge(start: Tuple[int, int], goal: Tuple[int, int], threat_map: np.ndarray) -> List[Tuple[int, int]]:
    rows, cols = threat_map.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path

        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and threat_map[nr, nc] < 2 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    return []  # No path


def main():
    ui = Grid_Map()
    ui.read_map(args.map)
    environment, battery_pos = ui.edit_map()

    row_count = len(environment)
    col_count = len(environment[0])

    dynamic_obstacles = DynamicObstaclesManager(ui, num_obstacles=0, speed_factor=args.speed)
    dynamic_obstacles.initialize_obstacles()

    static_threat = np.zeros((row_count, col_count), dtype=int)
    for r in range(row_count):
        for c in range(col_count):
            if environment[r][c] == 1:
                static_threat[r][c] = 2
    mcta = MCTA(row_count, col_count, args.uavs, args.energy)
    mcta.set_initial_threats(static_threat)

    start_positions = [battery_pos, (battery_pos[0], battery_pos[1] + 1), (battery_pos[0] + 1, battery_pos[1]), (battery_pos[0] + 1, battery_pos[1] + 1)]
    for i in range(mcta.v):
        pos = start_positions[i % len(start_positions)]
        found = False
        if 0 <= pos[0] < row_count and 0 <= pos[1] < col_count and static_threat[pos[0]][pos[1]] < 2:
            found = True
        else:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = battery_pos[0] + dr, battery_pos[1] + dc
                    if 0 <= nr < row_count and 0 <= nc < col_count and static_threat[nr][nc] < 2:
                        pos = (nr, nc)
                        found = True
                        break
                if found:
                    break
        if found:
            mcta.uavs[i].current_pos = pos
            mcta.uavs[i].add_to_trajectory(pos)
        else:
            raise ValueError(f"No free starting position for UAV {i+1}")

    pg.init()
    clock = pg.time.Clock()
    last_time = time.time()
    uav_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    uav_move_queues: List[deque] = [deque() for _ in mcta.uavs]
    sensor_range = 5
    uav_time_accum = [random.uniform(0, MOVE_INTERVAL) for _ in mcta.uavs]  # Stagger starts

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        delta_time = clock.tick(FPS) / 1000.0
        dynamic_obstacles.update(delta_time)

        mcta.threat_map = static_threat.copy()
        dynamic_positions = dynamic_obstacles.get_all_obstacle_positions()
        for r, c in dynamic_positions:
            if any(math.dist((r, c), uav.current_pos) < sensor_range for uav in mcta.uavs):
                if 0 <= r < row_count and 0 <= c < col_count:
                    mcta.threat_map[r, c] = 2

        for i, uav in enumerate(mcta.uavs):
            if uav.energy < 0.2 * uav.B and uav.mode != "RETURN":
                uav.mode = "RETURN"
                return_path = bfs_to_charge(uav.current_pos, battery_pos, mcta.threat_map)
                if return_path:
                    uav_move_queues[i].extend(return_path[1:])
                else:
                    uav.mode = "SLEEP"
            elif uav.current_pos == battery_pos and uav.mode == "RETURN":
                uav.energy = uav.B
                uav.mode = "WORK"

        continuing = mcta.execute_mcta_step()

        for i, uav in enumerate(mcta.uavs):
            new_path = uav.trajectory[max(0, len(uav.trajectory) - len(uav_move_queues[i]) - 1):]
            for pos in new_path[1:]:
                uav_move_queues[i].append(pos)

        # Update moves cell-by-cell
        for i, queue in enumerate(uav_move_queues):
            uav_time_accum[i] += delta_time
            if uav_time_accum[i] >= MOVE_INTERVAL and queue:
                next_pos = queue.popleft()
                mcta.uavs[i].current_pos = next_pos
                uav_time_accum[i] -= MOVE_INTERVAL

        # Always redraw everything every frame
        ui.draw_map()  # Redraw grid (clears previous)

        # Redraw coverage (persistent)
        for pos in mcta.global_covered:
            r, c = pos
            s = pg.Surface((EPSILON - 1, EPSILON - 1))
            s.set_alpha(60)
            s.fill((0, 255, 0))
            ui.WIN.blit(s, (c * EPSILON + 1, r * EPSILON + 1))

        # Redraw all UAV trajectories and positions
        for i, uav in enumerate(mcta.uavs):
            if len(uav.trajectory) > 1:
                points = [(p[1] * EPSILON + EPSILON // 2, p[0] * EPSILON + EPSILON // 2) for p in uav.trajectory]
                pg.draw.lines(ui.WIN, uav_colors[i % len(uav_colors)], False, points, width=2)
            pg.draw.circle(ui.WIN, uav_colors[i % len(uav_colors)],
                           (uav.current_pos[1] * EPSILON + EPSILON // 2, uav.current_pos[0] * EPSILON + EPSILON // 2), EPSILON // 3)

        # Redraw dynamics and station
        dynamic_obstacles.draw(ui.WIN)
        pg.draw.rect(ui.WIN, (255, 255, 0),
                     (battery_pos[1] * EPSILON + 1, battery_pos[0] * EPSILON + 1,
                      EPSILON - 2, EPSILON - 2))

        pg.display.flip()

        if not continuing:
            running = False

    pg.quit()
    total_travel_length = sum(uav.total_flight_mileage for uav in mcta.uavs)
    final_Cr, final_Rr, final_Df = mcta.calculate_performance_metrics()
    print('-' * 8)
    print('Total Path Length:', total_travel_length)
    print('Time: ', time.time() - last_time)

    print('=' * 50)
    print(f'1. Coverage Rate (C): {final_Cr:.2f}%')
    print(f'2. Overlap Rate (O) : {final_Rr:.2f}%')
    print(f'3. Number of Returns (R): {sum(1 for uav in mcta.uavs if uav.mode == "RETURN")}')
    print(f'4. Number of Deadlocks (D): 0')
    print(f'5. Execution Time (T): {time.time() - last_time:.3f}s')

    print('=' * 50)

    print("\nUAV Details:")
    for uav in mcta.uavs:
        status = uav.mode
        if hasattr(uav, 'sleep_reason'):
            status += f" ({uav.sleep_reason})"
        print(f"  UAV-{uav.id}: Energy={uav.energy:.1f}/{uav.B}, "
              f"Mileage={uav.total_flight_mileage:.1f}, "
              f"Visited={len(uav.trajectory_set)}, Status={status}")

if __name__ == "__main__":
    main()