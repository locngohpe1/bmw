import pygame as pg
import numpy as np
import random
import math
from grid_map import EPSILON

class DynamicObstaclesManager:
    def __init__(self, grid_map, num_obstacles=0, speed_factor=0.5):
        self.grid_map = grid_map
        self.obstacles = []
        self.epsilon = EPSILON
        self.num_obstacles = num_obstacles
        self.next_id = 1
        self.speed_factor = speed_factor

        self.human_icon = pg.image.load('assets/human_icon3.png')
        self.human_icon = pg.transform.scale(self.human_icon, (16, 20))

    def initialize_obstacles(self):
        if hasattr(self.grid_map, 'dynamic_obstacles'):
            for manual_obs in self.grid_map.dynamic_obstacles:
                pos = manual_obs['pos']

                base_velocity = (
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5)
                )

                obstacle = {
                    'id': manual_obs['id'],
                    'pos': pos,
                    'velocity': base_velocity,
                    'size': manual_obs.get(2,1),
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

    def _has_static_collision(self, cell, size, map_height, map_width):
        collision = False
        radius = int(max(size) / 2) if isinstance(size, tuple) else int(size / 2)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                check_row = cell[0] + dr
                check_col = cell[1] + dc
                if 0 <= check_row < map_height and 0 <= check_col < map_width:
                    if self.grid_map.map[check_row, check_col] in (1, 'o'):
                        collision = True
                        break
            if collision:
                break
        return collision

    def update(self, delta_time):
        if not self.obstacles:
            return

        map_width = len(self.grid_map.map[0])
        map_height = len(self.grid_map.map)

        for obstacle in self.obstacles:
            old_pos = obstacle['pos']
            old_exact = obstacle['exact_pos']
            vx, vy = obstacle['velocity']
            size = obstacle.get('size', 1.0)
            radius = max(size) / 2 if isinstance(size, tuple) else size / 2
            dt = delta_time * 15  # Speed multiplier

            # Move and reflect in x-direction (rows, vertical movement)
            new_x = old_exact[0] + vx * dt
            # Border check and reflection for x
            if new_x - radius < 0:
                new_x = radius
                vx = -vx
            elif new_x + radius >= map_height:
                new_x = map_height - radius
                vx = -vx
            # Static collision check for x-move only
            temp_cell = (int(new_x), int(old_exact[1]))
            if self._has_static_collision(temp_cell, size, map_height, map_width):
                vx = -vx
                # Clamp to approximate non-penetration (simple: revert to old_x, next frame moves reflected)
                new_x = old_exact[0]

            # Move and reflect in y-direction (columns, horizontal movement)
            new_y = old_exact[1] + vy * dt
            # Border check and reflection for y
            if new_y - radius < 0:
                new_y = radius
                vy = -vy
            elif new_y + radius >= map_width:
                new_y = map_width - radius
                vy = -vy
            # Static collision check for y-move only (using updated new_x)
            temp_cell = (int(new_x), int(new_y))
            if self._has_static_collision(temp_cell, size, map_height, map_width):
                vy = -vy
                # Clamp to old_y
                new_y = old_exact[1]

            # Update final position and velocity
            obstacle['exact_pos'] = (new_x, new_y)
            obstacle['pos'] = (int(new_x), int(new_y))
            obstacle['velocity'] = (vx, vy)

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