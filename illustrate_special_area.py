import pygame as pg
from special_area import Boustrophedon_Cellular_Decomposition
from grid_map import Grid_Map

ui = Grid_Map()
ui.read_map('map/real_map/cantwell.txt')
ENVIRONMENT, battery_pos = ui.edit_map()
# ui.save_map('map/empty_map.txt')
# pg.image.save(ui.WIN, 'tmp/screenshot.png')

# TEST - find special area
from optimization import get_special_area
reverse_dir = False
special_areas_id = [region.region_id for region in get_special_area(ENVIRONMENT, reverse_dir)]

if __name__ == "__main__":
    decomposed, region_count, adj_graph = Boustrophedon_Cellular_Decomposition(ENVIRONMENT, reverse_dir=reverse_dir)

    # TEST 1 - find special area
    # for i, row in enumerate(decomposed):
    #     for j, val in enumerate(row):
    #         if val != 0 and val not in special_areas_id:
    #             decomposed[i, j] = -1

    # ui.illustrate_regions(decomposed, region_count)

    # TEST 2 - find special area + inner special area
    special_areas = get_special_area(ENVIRONMENT)
    candidate_areas = get_special_area(ENVIRONMENT, reverse_dir=True)
    inner_special_areas = []
    for parent_region in special_areas:
        for child_region in candidate_areas:
            if set(child_region.cell_list) <= set(parent_region.cell_list): inner_special_areas.append(child_region)
    ui.illustrate_inner_special_regions(special_areas, inner_special_areas)

    # pg.image.save(ui.WIN, 'tmp/screenshot.png')
    input('Press enter to escape...')
