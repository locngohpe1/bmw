import numpy as np
import pygame as pg
import copy
import colorsys
import random

# Color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# GREY = (197, 198, 208)
GREY = (158, 159, 166)
RED = (255, 0, 0)
GREEN = (108, 217, 115)
YELLOW = (255, 243, 128)
BLUE = (5, 16, 148)
LIGHT_BLUE = (0, 191, 255)
BROWN = (76, 1, 33)

# epsilon value (size of 𝜀-cell)
EPSILON = 25

BORDER = 1
INFO_BAR_HEIGHT = 30

pg.init()
font = pg.font.SysFont(None, 30)

def hsv2rgb(h, s, v): 
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(n): 
    huePartition = 1.0 / (n + 1) 
    return [hsv2rgb(huePartition * value, 1.0, 1.0) for value in range(0, n)]
    return [LIGHT_BLUE] * n

class Grid_Map:
    def __init__(self):
        # pg.init()
        pg.display.set_caption("Coverage")
        self.WIN = None

        self.map = None
        self.row_count = 0
        self.col_count = 0

        self.battery_pos = (0, 0)
        self.vehicle_pos = (0, 0)

        self.battery_img = pg.Rect(BORDER, BORDER, EPSILON - BORDER, EPSILON - BORDER)
        self.vehicle_img = pg.Rect(BORDER, BORDER, EPSILON - BORDER, EPSILON - BORDER)

        self.trajectories = [[(0, 0)]] # list of trajectories (currently only coverage path)

        self.move_status = 0 # 0: normal coverage, 1: retreat, 2: charge, 3: advance
        self.charge_path_plan = [] # share between retreat & advance

        self.info_bar = None
        self.energy_display = None

    def read_map(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.col_count, self.row_count = [int(i) for i in f.readline().strip().split()]

            display_size = [EPSILON * self.col_count + 1, EPSILON * self.row_count + INFO_BAR_HEIGHT]
            self.info_bar = pg.Rect(0, EPSILON * self.row_count + BORDER, EPSILON * self.col_count, EPSILON * self.row_count + INFO_BAR_HEIGHT)
            self.WIN = pg.display.set_mode(display_size)

            map = []
            for idx, line in enumerate(f):
                line =[int(value) for value in line.strip().split()]
                map.append(line)
            
            if len(map) == 0:
                map = np.zeros((self.row_count, self.col_count), dtype=object)

            self.map = np.array(map, dtype=object)
        
        return copy.deepcopy(map), self.battery_pos

    def edit_map(self):
        done = False
        draw_obstacle = False 
        prev_cell = None

        while done == False:
            pos = pg.mouse.get_pos()
            col = pos[0] // EPSILON
            row = pos[1] // EPSILON

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                elif event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pressed = pg.mouse.get_pressed()
                    if mouse_pressed[0]: # left mouse click: obstacle
                        draw_obstacle = True
                    if mouse_pressed[2]: # right mouse click: starting position
                        if self.check_valid_pos((row, col)) == False: continue
                        self.update_battery_pos((row, col))
                        self.trajectories[0] = [(row, col)]
                        self.map[row][col] = 0
                        
                elif event.type == pg.MOUSEBUTTONUP:
                    draw_obstacle = False
                    prev_cell = None
            
            # check boolean flag to allow holding left click to draw
            if draw_obstacle:
                if self.check_valid_pos((row, col)) == False: continue
                if (prev_cell != (row, col)):
                    prev_cell = (row, col)
                    if self.map[row][col] == 0 and self.battery_pos != (row, col):
                        self.map[row][col] = 1
                    else:
                        self.map[row][col] = 0

            # pygame draw
            self.draw_map()
            pg.draw.rect(self.WIN, YELLOW, self.battery_img)
            pg.display.flip()

        return copy.deepcopy(self.map), self.battery_pos

    def save_map(self, output_file):
        map = self.map
        
        with open(output_file, "w", encoding="utf-8") as f:
            col_count, row_count = len(map[0]), len(map)
            f.write(str(col_count) + ' ' + str(row_count) + '\n')
            for row in map:
                line = [str(value) for value in row]
                line = " ".join(line)
                f.write(line +'\n')
            print("Save map done!")
    
    def draw(self):
        self.draw_map()
    
        for coverage_path in self.trajectories:
            self.draw_path(coverage_path)
        
        if self.move_status == 1: # retreat
            self.draw_path(self.charge_path_plan, BROWN, 4)
        elif self.move_status == 3: # advance
            self.draw_path(self.charge_path_plan, BLUE, 4)

        pg.draw.rect(self.WIN, YELLOW, self.battery_img)
        # pg.draw.rect(self.WIN, RED, self.vehicle_img)
        # vehicle_center = ((self.vehicle_pos[1] + 1/2) * EPSILON + BORDER, (self.vehicle_pos[0] + 1/2) * EPSILON + BORDER)
        # pg.draw.circle(self.WIN, BLACK, vehicle_center, EPSILON / 3, width=0)

        energy_display_img = font.render('Energy: ' + str(self.energy_display), True, RED)
        self.WIN.blit(energy_display_img, (self.info_bar.x + 5, self.info_bar.y + 5))

        pg.display.flip()

    def draw_map(self):
        self.WIN.fill(BLACK)
        pg.draw.rect(self.WIN, (238, 238, 238), self.info_bar)
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                color = WHITE
                if self.map[row][col] in (1, 'o'): # dùng chung cho map.txt
                    color = BLACK

                elif self.map[row][col] == '_':
                    color = GREY

                elif self.map[row][col] == 'e':
                    color = GREEN
                
                pg.draw.rect(self.WIN,
                            color,
                            [EPSILON * col + BORDER,
                            EPSILON * row + BORDER,
                            EPSILON - BORDER,
                            EPSILON - BORDER])
    
    def index2coordinate(self, pos):
        return (EPSILON * pos[1] + EPSILON / 2, EPSILON * pos[0] + EPSILON / 2)
    
    def draw_return_arrow(self, return_matrix):
        color = GREY
        h, w = len(return_matrix), len(return_matrix[0])
        for i in range(h):
            for j in range(w):
                if return_matrix[(i, j)][0] == None: continue
                pg.draw.line(self.WIN, color, self.index2coordinate((i, j)), self.index2coordinate(return_matrix[(i, j)][0]), width=2)

        pg.display.flip()

    def illustrate_regions(self, decomposed, region_count):
        self.WIN.fill(BLACK)
        pg.draw.rect(self.WIN, (238, 238, 238), self.info_bar)
        region_colors = getDistinctColors(region_count)
        random.shuffle(region_colors)

        for row in range(len(decomposed)):
            for col in range(len(decomposed[0])):
                color = BLACK
                region = decomposed[row][col]
                
                if region == -1:
                    color = WHITE
                elif region != 0:
                    color = region_colors[region - 1]

                pg.draw.rect(self.WIN,
                            color,
                            [EPSILON * col + BORDER,
                            EPSILON * row + BORDER,
                            EPSILON - BORDER,
                            EPSILON - BORDER])

        pg.display.flip()

    def draw_path(self, path, color=RED, width=2):
        point_list = [(EPSILON * pos[1] + EPSILON / 2, EPSILON * pos[0] + EPSILON / 2) for pos in path]
        if len(point_list) > 1:
            pg.draw.lines(self.WIN, color, False, point_list, width=3)

    def update_battery_pos(self, pos):
        self.battery_pos = pos
        self.battery_img.x = EPSILON * pos[1] + BORDER
        self.battery_img.y = EPSILON * pos[0] + BORDER
    
    def update_vehicle_pos(self, pos):
        self.vehicle_pos = pos
        self.vehicle_img.x = EPSILON * pos[1] + BORDER
        self.vehicle_img.y = EPSILON * pos[0] + BORDER

    def task(self, pos):
        self.map[pos] = 'e'

    def move_to(self, pos):
        if self.move_status != 0:
            self.trajectories[-1].append(self.vehicle_pos)
        self.move_status = 0
        self.update_vehicle_pos(pos)
        self.trajectories[-1].append(pos)
    
    def move_retreat(self, pos):
        self.move_status = 1
        self.update_vehicle_pos(pos)
        if pos == self.battery_pos:
            self.move_status = 2

    def move_advance(self, pos):
        if self.move_status != 3:
            self.trajectories.append([])
        self.move_status = 3
        self.update_vehicle_pos(pos)
    
    def set_charge_path(self, path):
        self.charge_path_plan = path
    
    def set_energy_display(self, energy):
        self.energy_display = round(energy, 2)

    def check_valid_pos(self, pos):
        row, col = pos
        if row < 0 or row >= self.row_count: return False
        if col < 0 or col >= self.col_count: return False
        return True

ui = Grid_Map()
ui.read_map('map/empty_map.txt')
ENVIRONMENT, battery_pos = ui.edit_map()

ROW_COUNT = len(ENVIRONMENT)
COL_COUNT = len(ENVIRONMENT[0])

# Pre-calculate return path to charge station from each cell in ENVIRONMENT
from optimization import return_path_matrix, get_return_path
return_matrix = return_path_matrix(ENVIRONMENT, battery_pos)

ui.draw()
ui.draw_return_arrow(return_matrix)
pg.image.save(ui.WIN, 'tmp/screenshot.png')
input()
