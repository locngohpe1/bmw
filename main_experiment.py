import math
import numpy as np
import pygame as pg
import time
import csv

from a_star import GridMapGraph, a_star_search
from logic import Logic, Q
from grid_map import Grid_Map

# coverage:             1 unit of energy / cell width
# advance & retreat:    0.5 unit of energy / cell width
ENERGY_CAPACITY = 1000

ui = Grid_Map()
file_name = 'map/real_map/cantwell.txt'
ui.read_map(file_name)
ENVIRONMENT, battery_pos = ui.edit_map()
# ui.save_map('map/empty_map.txt')

ROW_COUNT = len(ENVIRONMENT)
COL_COUNT = len(ENVIRONMENT[0])

FPS = 10000

total_travel_length = 0
coverage_length, retreat_length, advance_length = 0, 0, 0
return_charge_count = 1
count_cell_go_through = 1
deadlock_count = 0
extreme_deadlock_count = 0
execute_time = time.time()

# Find special area
from optimization import get_special_area
special_areas = get_special_area(ENVIRONMENT)

# Pre-calculate return path to charge station from each cell in ENVIRONMENT
from optimization import return_path_matrix, get_return_path
return_matrix = return_path_matrix(ENVIRONMENT, battery_pos)

def check_valid_pos(pos):
    row, col = pos
    if row < 0 or row >= ROW_COUNT: return False
    if col < 0 or col >= COL_COUNT: return False
    return True

class Robot:
    def __init__(self, battery_pos, map_row_count, map_col_count):
        self.logic = Logic(map_row_count, map_col_count)
        '''
        map:
            'u': unvisited
            'e': explored
            'o': obstacle
        '''
        self.map = None
        self.current_pos = battery_pos

        # The angle between the robot direction and left to right axis in rad [0, 2pi)
        # (up direction at the start)
        self.angle = math.pi / 2

        self.battery_pos = battery_pos
        self.energy = ENERGY_CAPACITY
        self.estimated_return_energy = 0

        self.move_status = 0 # 0: normal coverage, 1: retreat, 2: charge, 3: advance
        self.cache_path = [] # store temporary path (e.g.: retreat, advance)
        self.cache_wp = None
    
    def set_map(self, environment):
        row_count, col_count = len(environment), len(environment[0])
        self.map = np.full((row_count, col_count), 'u')

        for x in range(len(environment)):
            for y in range(len(environment[0])):
                if environment[x, y] == 1:
                    self.map[x, y] = 'o'

        self.logic.set_weight_map(environment)

    def run(self):
        global FPS, deadlock_count, extreme_deadlock_count
        clock = pg.time.Clock()
        run = True
        pause = False
        coverage_finish = False
        
        while run:
            # ui.draw()
            # clock.tick(FPS)
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:     # pause
                        pause = not pause
                    elif event.key == pg.K_LEFT:    # slow down
                        FPS /= 2
                    elif event.key == pg.K_RIGHT:   # speed up
                        FPS *= 2
                if event.type == pg.QUIT:
                    run = False
        
            if pause:
                continue

            if self.logic.state == Q.FINISH:
                if not coverage_finish:
                    coverage_finish = True
                    self.retreat()

                    global execute_time
                    execute_time = time.time() - execute_time

                    print('Coverage Finish')

                # FN (do nothing until close window)
                continue

            wp = self.logic.get_wp(self.current_pos)
            if len(wp) == 0: continue
            selected_cell = self.select_from_wp(wp)
            
            if selected_cell == self.current_pos:
                self.task()
            else:
                # CP 0
                if self.logic.state == Q.NORMAL:
                    if self.check_enough_energy(selected_cell) == False:
                        self.charge_planning()
                        continue
                    self.move_to(selected_cell)
                
                # CP l (l > 0)
                elif self.logic.state == Q.DEADLOCK:
                    # self.bug2(selected_cell)
                    # graph = GridMapGraph(self.map)
                    # path, dist = a_star_search(graph, self.current_pos, selected_cell)
                    path, dist = self.logic.cache_path, self.logic.cache_dist
                    print(f"Deadlock ({round(dist, 2)})")

                    deadlock_count += 1
                    if dist > math.sqrt(ROW_COUNT**2 + COL_COUNT**2) / 4: extreme_deadlock_count += 1

                    self.follow_path_plan(path, time_delay=0.05, check_energy=True, stop_on_unexpored=True)

    def select_from_wp(self, wp):
        new_wp = self.get_better_wp(wp)
        if len(new_wp) > 0: wp = new_wp

        return min(wp, key=self.travel_cost)

    def task(self):
        current_pos = self.current_pos
        self.map[current_pos] = 'e'
        self.logic.update_explored(current_pos)
        ui.task(current_pos)
    
    def move_to(self, pos):
        global total_travel_length, coverage_length, retreat_length, advance_length, count_cell_go_through
        dist = energy = math.dist(self.current_pos, pos)

        if self.move_status in (1, 3): # retreat or advance cost half energy as coverage
            energy = 0.5 * energy

        if self.energy < energy:
            raise Exception('Robot run out of battery')
        self.energy -= energy

        self.rotate_to(pos)
        self.current_pos = pos

        if self.move_status == 0:
            ui.move_to(pos)
            coverage_length += dist
        elif self.move_status == 1:
            ui.move_retreat(pos)
            retreat_length += dist
        elif self.move_status == 3:
            ui.move_advance(pos)
            advance_length += dist
        
        total_travel_length += dist
        if self.move_status == 0: # coverage
            count_cell_go_through += 1

        ui.set_energy_display(self.energy)

    def travel_cost(self, pos_to):
        pos_from = self.current_pos
        turn_angle = abs(self.angle - self.get_angle(pos_to))
        if turn_angle > math.pi: # always take the smaller angle to turn
            turn_angle = 2 * math.pi - turn_angle 
        travel_dist = math.dist(pos_from, pos_to)

        # cost of travel distance, turning rad
        cost = 2 * travel_dist + 1 * turn_angle
        return cost
    
    def get_angle(self, pos_to):
        pos_from = self.current_pos
        vecto = (pos_to[0] - pos_from[0], pos_to[1] - pos_from[1])
        angle = - np.arctan2(vecto[0], vecto[1])
        return angle % (2 * math.pi)
    
    def rotate_to(self, pos_to):
        self.angle = self.get_angle(pos_to)

    def check_enough_energy(self, wp):
        return_dist_from_wp = return_matrix[wp][1]
        expected_energy = math.dist(self.current_pos, wp) + 0.5 * return_dist_from_wp
        if self.energy < expected_energy:
            return False
        else:
            return True

    def charge_planning(self):
        global return_charge_count
        return_charge_count += 1
        # retreat
        self.retreat()

        # charge
        self.charge()
        # time.sleep(0.1)

        # advance
        self.advance()

        # coverage
        self.move_status = 0
    
    def retreat(self):
        return_path = get_return_path(return_matrix, self.current_pos)
        self.cache_path = return_path # save for reuse in advance path

        self.move_status = 1
        ui.set_charge_path(return_path)
        self.follow_path_plan(return_path, time_delay=0.05)
    
    def charge(self):
        self.move_status = 2
        self.energy = ENERGY_CAPACITY

    def advance(self):
        self.move_status = 3
        advance_path = list(reversed(self.cache_path))
        ui.set_charge_path(advance_path)
        self.follow_path_plan(advance_path, time_delay=0.05)
    
    def follow_path_plan(self, path, time_delay=0, check_energy=False, stop_on_unexpored=False):
        clock = pg.time.Clock()
        for pos in path:
            # clock.tick(FPS)

            while check_energy == True and self.check_enough_energy(pos) == False:
                self.charge_planning()

            self.move_to(pos)
            # ui.draw()
            # time.sleep(time_delay) # comment to temporary remove delay

            if stop_on_unexpored:
                if self.logic.weight_map[pos] > 0: return

        # self.cache_path.clear()
    
    # TEST
    def get_better_wp(self, wp):
        if len(wp) == 1: return wp

        new_wp = []
        x_up, y_up = min(wp, key = lambda x: x[0])
        x_down, y_down = max(wp, key = lambda x: x[0])

        if not check_valid_pos((x_up - 1, y_up)) or self.map[(x_up - 1, y_up)] in ('o', 'e'):
            # if abs(self.angle - self.get_angle((x_up, y_up))) <= math.pi / 2:
                new_wp.append((x_up, y_up)) 
        if not check_valid_pos((x_down + 1, y_down)) or self.map[(x_down + 1, y_down)] in ('o', 'e'):
            # if abs(self.angle - self.get_angle((x_down, y_down))) <= math.pi / 2:
                new_wp.append((x_down, y_down))
        return new_wp

    def set_special_areas(self, special_areas):
        self.logic.set_special_areas(special_areas)

        # TEST
        self.set_inner_special_areas(special_areas)
        # np.savetxt('tmp/weight_map.txt', self.logic.weight_map, fmt='%-3d')
    
    def set_inner_special_areas(self, special_areas):
        candidate_areas = get_special_area(ENVIRONMENT, reverse_dir=True)
        for parent_region in special_areas:
            for child_region in candidate_areas:
                if not (set(child_region.cell_list) <= set(parent_region.cell_list)): continue

                floor_weight = -1
                for pos in parent_region.cell_list:
                    if pos[1] == child_region.max_y:
                        floor_weight = self.logic.weight_map[pos] + 2
                        break

                for x, y in child_region.cell_list:
                    self.logic.weight_map[x, y] = floor_weight + (child_region.max_y - y)

def main():
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.set_map(ENVIRONMENT)
    robot.set_special_areas(special_areas)
    # robot.etm.save_map('potential_field.txt')
    # robot.etm.read_map('potential_field.txt')

    robot.run()
    
    print('\nCoverage:\t', coverage_length)
    print('Retreat:\t', retreat_length)
    print('Advance:\t', advance_length)
    print('-' * 8)
    print('Total:', total_travel_length)

    overlap_rate = (count_cell_go_through / np.sum(robot.map == 'e') - 1) * 100
    print('\nOverlap rate:', overlap_rate)
    print('Number of return:', return_charge_count)
    print('Number of extreme deadlock:', extreme_deadlock_count, '/', deadlock_count)
    print('Time: ', execute_time)

    # with open('result.csv','a',newline='') as fd:
    #     writer = csv.writer(fd)
    #     col_list = [file_name[15:-4], coverage_length, retreat_length, advance_length, total_travel_length, overlap_rate, return_charge_count, f'{extreme_deadlock_count}/{deadlock_count}', execute_time]
    #     writer.writerow(col_list)

if __name__ == "__main__":
    main()
