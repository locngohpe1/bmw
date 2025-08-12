import pygame as pg
import numpy as np
import random
import math
from grid_map import EPSILON


class DynamicObstaclesManager:
    def __init__(self, grid_map, num_obstacles=2, speed_factor=0.5):
        self.grid_map = grid_map
        self.obstacles = []
        self.epsilon = EPSILON
        self.num_obstacles = num_obstacles
        self.next_id = 1
        self.speed_factor = speed_factor

        self.human_icon = pg.image.load('assets/human_icon3.png')
        self.human_icon = pg.transform.scale(self.human_icon, (16, 26))

    def initialize_obstacles(self):
        if hasattr(self.grid_map, 'dynamic_obstacles'):
            for manual_obs in self.grid_map.dynamic_obstacles:
                pos = manual_obs['pos']

                base_velocity = (
                    random.uniform(-0.03, 0.03),
                    random.uniform(-0.03, 0.03)
                )

                max_attempts = 30
                attempt = 0
                while (abs(base_velocity[0]) < 0.02 and abs(base_velocity[1]) < 0.02) and attempt < max_attempts:
                    base_velocity = (
                        random.uniform(-0.03, 0.03),
                        random.uniform(-0.03, 0.03)
                    )
                    attempt += 1

                if abs(base_velocity[0]) < 0.02 and abs(base_velocity[1]) < 0.02:
                    base_velocity = (0.025, 0.025)

                velocity = (
                    base_velocity[0] * max(self.speed_factor, 0.1),
                    base_velocity[1] * max(self.speed_factor, 0.1)
                )

                obstacle = {
                    'id': manual_obs['id'],
                    'pos': pos,
                    'velocity': velocity,
                    'size': manual_obs.get('size', 1.0),
                    'color': (255, 0, 0),
                    'exact_pos': (pos[0] + 0.5, pos[1] + 0.5)
                }

                self.obstacles.append(obstacle)
                self.next_id += 1

    def _clear_obstacle_cells(self, center_pos, size):
        radius = int(max(size) / 2) if isinstance(size, tuple) else int(size / 2)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row, col = center_pos[0] + dr, center_pos[1] + dc
                if (0 <= row < len(self.grid_map.map) and
                        0 <= col < len(self.grid_map.map[0]) and
                        self.grid_map.map[row, col] == 'd'):
                    self.grid_map.map[row, col] = 0

    def update(self, delta_time):
        if not self.obstacles:
            return

        map_width = len(self.grid_map.map[0])
        map_height = len(self.grid_map.map)

        for obstacle in self.obstacles:
            old_pos = obstacle['pos']
            old_exact = obstacle['exact_pos']

            new_x = old_exact[0] + obstacle['velocity'][0] * delta_time * 15
            new_y = old_exact[1] + obstacle['velocity'][1] * delta_time * 15

            size = obstacle.get('size', 1.0)
            obstacle_radius = max(size) / 2 if isinstance(size, tuple) else size / 2

            if new_x - obstacle_radius < 0 or new_x + obstacle_radius >= map_height:
                obstacle['velocity'] = (-obstacle['velocity'][0], obstacle['velocity'][1])
                new_x = old_exact[0]

            if new_y - obstacle_radius < 0 or new_y + obstacle_radius >= map_width:
                obstacle['velocity'] = (obstacle['velocity'][0], -obstacle['velocity'][1])
                new_y = old_exact[1]

            new_cell = (int(new_x), int(new_y))
            collision_with_static = False
            radius = int(max(size) / 2) if isinstance(size, tuple) else int(size / 2)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    check_row = new_cell[0] + dr
                    check_col = new_cell[1] + dc
                    if 0 <= check_row < map_height and 0 <= check_col < map_width:
                        if self.grid_map.map[check_row, check_col] in (1, 'o'):
                            collision_with_static = True
                            break
                if collision_with_static:
                    break

            if collision_with_static and new_cell != old_pos:
                obstacle['velocity'] = (-obstacle['velocity'][0], -obstacle['velocity'][1])
                new_x = old_exact[0]
                new_y = old_exact[1]
                new_cell = old_pos

            obstacle['exact_pos'] = (new_x, new_y)
            obstacle['pos'] = (int(new_x), int(new_y))

            if old_pos != obstacle['pos']:
                self._clear_obstacle_cells(old_pos, size)

    def draw(self, surface):
        for obstacle in self.obstacles:
            x = obstacle['exact_pos'][1] * self.epsilon
            y = obstacle['exact_pos'][0] * self.epsilon
            icon_w, icon_h = self.human_icon.get_size()
            draw_x = int(x + (self.epsilon - icon_w) / 2)
            draw_y = int(y + (self.epsilon - icon_h) / 2)
            surface.blit(self.human_icon, (draw_x, draw_y))

    def get_obstacle_info(self, obstacle_id):
        for obstacle in self.obstacles:
            if obstacle['id'] == obstacle_id:
                return obstacle
        return None

    def get_all_obstacle_positions(self):
        positions = []
        for obstacle in self.obstacles:
            radius = int(max(obstacle['size']) / 2) if isinstance(obstacle['size'], tuple) else int(
                obstacle['size'] / 2)
            center_pos = obstacle['pos']
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    row, col = center_pos[0] + dr, center_pos[1] + dc
                    if (0 <= row < len(self.grid_map.map) and
                            0 <= col < len(self.grid_map.map[0])):
                        positions.append((row, col))
        return positions

    def _mark_obstacle_cells(self, center_pos, size):
        pass