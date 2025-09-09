import math
import numpy as np
from collections import deque

LLOCAL_NEIGHBOR_0_WIDTH = 3
neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

class Q:
    START, NORMAL, DEADLOCK, FINISH = range(4)

def init_weight_map_mask(row, col):
    weight_map = np.zeros([row, col])
    for i, row_value in enumerate(weight_map):
        for j, col_value in enumerate(row_value):
            weight_map[i][j] = col - j + 1
    return weight_map

class Logic:
    def __init__(self, row_count, col_count, grid_map=None):
        self.grid_map = grid_map
        self.state = Q.START
        self.weight_map = init_weight_map_mask(row_count, col_count)
        self._prev_wp = []
        self.cache_path = []
        self.cache_dist = 0

    def get_wp(self, current_pos):
        row_count, col_count = len(self.weight_map), len(self.weight_map[0])
        x_cur, y_cur = current_pos
        wp = []

        self.state = Q.NORMAL
        weight_map = self.weight_map
        set_D = self.get_set_D(current_pos)

        if weight_map[x_cur][y_cur] > 0:
            if (x_cur - 1, y_cur) in set_D and (x_cur + 1, y_cur) in set_D:
                max_cells = self.max_potential_cells(set_D)
                max_local_potential = weight_map[max_cells[0]]
                if weight_map[x_cur][y_cur] < max_local_potential:
                    for cell in max_cells:
                        if self.next_to_neighbor(cell):
                            wp.append(cell)
                    if len(wp) > 0:
                        return wp

                d_up, d_down = 1, 1
                while x_cur + d_up < row_count and self.weight_map[x_cur + d_up, y_cur] > 0:
                    d_up += 1
                while x_cur - d_down >= 0 and self.weight_map[x_cur - d_down, y_cur] > 0:
                    d_down += 1

                if d_up <= d_down:
                    wp.append((x_cur + 1, y_cur))
                else:
                    wp.append((x_cur - 1, y_cur))
            else:
                wp.append(current_pos)
        elif len(set_D) != 0:
            wp = self.max_potential_cells(set_D)

        if len(wp) != 0:
            self._prev_wp = wp
            return wp

        self.state = Q.DEADLOCK
        wp = self.get_local_extreme_wp(current_pos)

        if len(wp) == 0:
            self.state = Q.FINISH
        return wp

    def get_local_extreme_wp(self, current_pos):
        weight_map = self.weight_map
        return_matrix = np.zeros(weight_map.shape, dtype=object)
        for x in range(len(return_matrix)):
            for y in range(len(return_matrix[0])):
                return_matrix[x, y] = [None, math.inf]

        queue = deque()
        visited_matrix = np.zeros(weight_map.shape, dtype=bool)
        candidate_res = []
        stop_depth = -1

        queue.append((current_pos, 0))
        visited_matrix[current_pos] = True
        return_matrix[current_pos] = [None, 0]

        while queue:
            cur_node, cur_depth = queue.popleft()

            if stop_depth != -1 and cur_depth != stop_depth:
                escape_wp = min(candidate_res, key=lambda x: return_matrix[x][1])
                self.cache_path = self.get_wavefront_path(return_matrix, escape_wp)
                self.cache_dist = return_matrix[escape_wp][1]
                return [escape_wp]

            if weight_map[cur_node] > 0:
                if stop_depth == -1:
                    stop_depth = cur_depth
                if cur_node not in candidate_res:
                    candidate_res.append(cur_node)
            if stop_depth != -1:
                continue

            for dx, dy in neighbors:
                x, y = cur_node[0] + dx, cur_node[1] + dy

                if x < 0 or x >= len(weight_map):
                    continue
                if y < 0 or y >= len(weight_map[0]):
                    continue
                if weight_map[x, y] == -1:
                    continue

                if hasattr(self, 'grid_map') and self.grid_map:
                    if self.grid_map.map[x, y] == 'd':
                        continue

                new_dist = return_matrix[cur_node][1] + math.dist(cur_node, (x, y))
                if new_dist < return_matrix[x, y][1]:
                    return_matrix[x, y][0] = cur_node
                    return_matrix[x, y][1] = new_dist

                if not visited_matrix[x, y]:
                    visited_matrix[x, y] = True
                    queue.append(((x, y), cur_depth + 1))

        if len(candidate_res) > 0:
            escape_wp = min(candidate_res, key=lambda x: return_matrix[x][1])
            self.cache_path = self.get_wavefront_path(return_matrix, escape_wp)
            self.cache_dist = return_matrix[escape_wp][1]
            return [escape_wp]

        self.cache_path = None
        return []

    def get_wavefront_path(self, return_matrix, cur_pos):
        path = []
        cur = cur_pos
        while cur is not None:
            path.append(cur)
            cur = return_matrix[cur][0]
        path = list(reversed(path))
        return path

    def get_set_D(self, current_pos):
        x_cur, y_cur = current_pos
        weight_map = self.weight_map
        set_D = []

        for (dx, dy) in neighbors:
            x = x_cur + dx
            y = y_cur + dy

            if x < 0 or x >= len(weight_map):
                continue
            if y < 0 or y >= len(weight_map[0]):
                continue

            if weight_map[x][y] > 0:
                set_D.append((x, y))

        return set_D

    def max_potential_cells(self, cell_list):
        weight_map = self.weight_map
        max_list = []
        max_value = 0

        for row, col in cell_list:
            if max_value < weight_map[row][col]:
                max_value = weight_map[row][col]
                max_list.clear()
                max_list.append((row, col))
            elif max_value == weight_map[row][col]:
                max_list.append((row, col))

        return max_list

    def set_weight_map(self, environment):
        for x, row in enumerate(environment):
            for y, val in enumerate(row):
                if val == 1:
                    self.weight_map[x, y] = -1

    def update_explored(self, pos):
        self.weight_map[pos] = 0

    def next_to_neighbor(self, cell):
        rotate = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        weight_map = self.weight_map

        for i in range(len(rotate)):
            x, y = np.add(cell, rotate[i])
            if x < 0 or x >= len(weight_map):
                continue
            if y < 0 or y >= len(weight_map[0]):
                continue

            if weight_map[x, y] == -1:
                return True
        return False

    def set_special_areas(self, special_areas):
        col_count = len(self.weight_map[0])
        for region in special_areas:
            for x, y in region.cell_list:
                self.weight_map[x, y] = col_count + 3 - region.min_y + (y - region.min_y)