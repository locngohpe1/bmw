import math
import numpy as np

# get the list of fewest cells to go in straight line from start point to goal point
def direct_cell_list(pos_from, pos_to):
    cell_list = obstruct_cell_list(pos_from, pos_to)
    length = len(cell_list)
    i = 2

    # remove redundant cell
    while i < length:
        if check_neighbor(cell_list[i], cell_list[i - 2]) == True:
            cell_list.pop(i - 1)
            length -= 1
        else:
            i += 1
        
    return cell_list

'''
param strict:
    False: only cells that obstruct the line
    True: cells that obstruct the line and cells that are near the line base on threashold
'''
def obstruct_cell_list(pos_from, pos_to, strict=False):
    threshold = 0.3 # threshold value: [0, 0.5]
    start = (pos_from[0] + 0.5, pos_from[1] + 0.5)
    goal = (pos_to[0] + 0.5, pos_to[1] + 0.5)

    vecto = (goal[0] - start[0], goal[1] - start[1])
    angle = - np.arctan2(vecto[0], vecto[1])

    (x, y) = pos_from
    cell_list = [pos_from]

    sx, sy = sign(vecto[0]), sign(vecto[1])
    dx = abs(0.5 / math.sin(angle)) if vecto[0] != 0 else math.inf
    dy = abs(0.5 / math.cos(angle)) if vecto[1] != 0 else math.inf
    sum_x, sum_y = dx, dy

    while (x, y) != pos_to:
        # if sum_x == sum_y, increment both x and y
        (movx, movy) = (sum_x < sum_y or math.isclose(sum_x, sum_y), sum_y < sum_x or math.isclose(sum_x, sum_y)) # bugfix: sin, cos not give accurate res

        prev_x, prev_y = x, y
        prev_sum_x, prev_sum_y = sum_x, sum_y
        if movx:
            x += sx
            sum_x += 2 * dx

        if movy:
            y += sy
            sum_y += 2 * dy

        if strict:
            if movx and movy: 
                cell_list.extend([(prev_x, prev_y + sy), (prev_x + sx, prev_y)])
            elif movx and not movy: 
                projection_y = (abs(prev_sum_x * math.cos(angle)) - 0.5) % 1
                if projection_y < threshold:
                    cell_list.append((x, prev_y - sy))
                elif projection_y > 1 - threshold:
                    cell_list.append((prev_x, prev_y + sy))
            elif movy and not movx:
                projection_x = (abs(prev_sum_y * math.sin(angle)) - 0.5) % 1
                if projection_x < threshold:
                    cell_list.append((prev_x - sx, y))
                elif projection_x > 1 - threshold:
                    cell_list.append((prev_x + sx, prev_y))

        cell_list.append((x, y))

    return cell_list

def get_neighbor_pos(pos, row_count, col_count):
    neighbor_list = []
    for row in range(pos[0] - 1, pos[0] + 2):
        for col in range(pos[1] - 1, pos[1] + 2):
            if row < 0 or row >= row_count: continue
            if col < 0 or col >= col_count: continue
            if (row, col) == pos: continue
            neighbor_list.append((row, col))
    
    return neighbor_list

# assume pos1 and pos2 are valid position
def check_neighbor(pos1, pos2):
    if abs(pos2[0] - pos1[0]) <= 1 and abs(pos2[1] - pos1[1]) <= 1:
        return True
    else:
        return False

def sign(n):
    return int(np.sign(n))
