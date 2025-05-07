from math import dist, inf
import numpy as np
from collections import deque

neighbors = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
# neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
# neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

# Matrix of [position from, distance to battery] of each grid cell (Dijkstra shortest path) 
def return_path_matrix(environment, battery_position):
    return_matrix = np.zeros(environment.shape, dtype=object)
    for x in range(len(return_matrix)):
        for y in range(len(return_matrix[0])):
            return_matrix[x, y] = [None, inf]

    queue = deque()
    visited_matrix = np.zeros(environment.shape, dtype=bool)

    queue.append(battery_position)
    visited_matrix[battery_position] = True
    return_matrix[battery_position] = [None, 0]

    while queue:
        cur_node = queue.popleft()
        # traverse neighbors
        for dx, dy in neighbors:
            x, y = cur_node[0] + dx, cur_node[1] + dy
            
            if x < 0 or x >= len(environment): continue
            if y < 0 or y >= len(environment[0]): continue
            if environment[x, y] == 1: continue # obstacle

            new_dist = return_matrix[cur_node][1] + dist(cur_node, (x, y))
            if new_dist < return_matrix[x, y][1]:
                return_matrix[x, y][0] = cur_node
                return_matrix[x, y][1] = new_dist
            
            if not visited_matrix[x, y]:
                visited_matrix[x, y] = True
                queue.append((x, y))
    
    return return_matrix

# Support function of return_path_matrix above
def get_return_path(return_matrix, cur_pos):
    path = []
    dist = return_matrix[cur_pos][1]

    cur = cur_pos
    while cur is not None:
        path.append(cur)
        cur = return_matrix[cur][0]

    return path

from special_area import Boustrophedon_Cellular_Decomposition, create_regions
def get_special_area(environment, reverse_dir=False):
    special_areas_id = []
    decomposed, region_count, adj_graph = Boustrophedon_Cellular_Decomposition(environment, special_areas_id, reverse_dir)
    regions = create_regions(decomposed, region_count)

    special_areas = []
    for region in regions:
        if region is None: continue
        if region.region_id in special_areas_id:
            special_areas.append(region)

    return special_areas
    