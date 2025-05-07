import numpy as np
from PIL import Image

def translate_to_gridmap(map_img_path, cell_size, output_file):
    # Binarize the image: 0 free space / 1 obstacle
    binary_map = Image.open(map_img_path)
    binary_map = np.array(binary_map)
    binary_map[binary_map > 0] = 1
    binary_map = 1 - binary_map
    
    # Grid map size
    rows, cols = binary_map.shape
    grid_map = np.zeros((rows//cell_size, cols//cell_size), dtype=np.uint8)
    
    # Convert binary map to the grid map
    for i in range(rows//cell_size):
        for j in range(cols//cell_size):
            grid_cell = binary_map[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            if np.sum(grid_cell) > 0:
                grid_map[i, j] = 1
    
    if len(grid_map) > len(grid_map[0]):
        grid_map = np.rot90(grid_map)

    while True:
        if np.all(grid_map[0, :]):
            grid_map = np.delete(grid_map, (0), axis=0)
        else: break
    while True:
        if np.all(grid_map[-1, :]):
            grid_map = np.delete(grid_map, (-1), axis=0)
        else: break
    while True:
        if np.all(grid_map[:, 0]):
            grid_map = np.delete(grid_map, (0), axis=1)
        else: break
    while True:
        if np.all(grid_map[:, -1]):
            grid_map = np.delete(grid_map, (-1), axis=1)
        else: break
                
    # Save to txt file
    with open(output_file, "w") as f:
        col_count, row_count = len(grid_map[0]), len(grid_map)
        f.write(str(col_count) + ' ' + str(row_count) + '\n')
        for row in grid_map:
            line = [str(value) for value in row]
            line = " ".join(line)
            f.write(line +'\n')

if __name__ == '__main__':
    data_file = 'Eastville'
    translate_to_gridmap(f'real_map_dataset/{data_file}.png', 8, f'map/{data_file.lower()}.txt')
