import pygame as pg
import numpy as np
import os
from special_area import Boustrophedon_Cellular_Decomposition
from optimization import get_special_area

# Constants
GRID_SIZE = 16  # 16x16 grid
CELL_SIZE = 40
BORDER = 2
FONT_SIZE = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)
LIGHT_ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Display modes
MODE_NORMAL = 0
MODE_WEIGHTS = 1
MODE_SPECIAL_AREAS = 2
MODE_RECONSTRUCTED_WEIGHTS = 3


class EnhancedWeightMapVisualizer:
    def __init__(self):
        pg.init()

        # Calculate window size
        self.window_width = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * BORDER
        self.window_height = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * BORDER + 120

        self.screen = pg.display.set_mode((self.window_width, self.window_height))
        pg.display.set_caption("BWave Framework Visualizer - Click: Toggle Wall, Tab: Cycle Modes, Space: Screenshot")

        # Initialize grid (0 = free, 1 = obstacle)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.weight_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.reconstructed_weight_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.reconstructed_cells = set()  # Track which cells were reconstructed

        # Special areas data
        self.special_areas = []
        self.inner_special_areas = []
        self.decomposed = None

        # State
        self.display_mode = MODE_NORMAL

        # Font
        self.font = pg.font.Font(None, FONT_SIZE)
        self.small_font = pg.font.Font(None, 16)
        self.large_font = pg.font.Font(None, FONT_SIZE * 2)

        self.calculate_weights()
        self.calculate_special_areas()
        self.reconstruct_special_weights()

    def calculate_weights(self):
        """Calculate weight map: W[i,j] = col_count - j"""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] == 1:  # Obstacle
                    self.weight_map[i, j] = -1
                else:  # Free space
                    self.weight_map[i, j] = GRID_SIZE - j

    def calculate_special_areas(self):
        """Calculate special areas using BWave framework"""
        try:
            # Convert grid format for BWave (0=free, 1=obstacle)
            environment = self.grid.copy()

            # Find special areas
            self.special_areas = get_special_area(environment, reverse_dir=False)
            candidate_areas = get_special_area(environment, reverse_dir=True)

            # Find inner special areas
            self.inner_special_areas = []
            for parent_region in self.special_areas:
                for child_region in candidate_areas:
                    if set(child_region.cell_list) <= set(parent_region.cell_list):
                        self.inner_special_areas.append(child_region)

            # Get decomposition for visualization
            self.decomposed, region_count, adj_graph = Boustrophedon_Cellular_Decomposition(
                environment, reverse_dir=False
            )

        except Exception as e:
            print(f"Error calculating special areas: {e}")
            self.special_areas = []
            self.inner_special_areas = []
            self.decomposed = None

    def reconstruct_special_weights(self):
        """Reconstruct weight map for special areas - Section 4.2.2"""
        # Start with original weight map
        self.reconstructed_weight_map = self.weight_map.copy()
        self.reconstructed_cells.clear()

        col_count = GRID_SIZE

        # Apply weight reconstruction for special areas
        for region in self.special_areas:
            if hasattr(region, 'cell_list') and hasattr(region, 'min_y'):
                for x, y in region.cell_list:
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        # BWave formula: W[x,y] = col_count + 3 - region.min_y + (y - region.min_y)
                        new_weight = col_count + 3 - region.min_y + (y - region.min_y)
                        self.reconstructed_weight_map[x, y] = new_weight
                        self.reconstructed_cells.add((x, y))

        print(f"Reconstructed weights for {len(self.reconstructed_cells)} cells in special areas")

    def get_gradient_color(self, weight_value, is_reconstructed=False):
        """Get gradient color based on weight value"""
        if weight_value == -1:
            return BLACK  # Obstacles remain black

        # For reconstructed cells, use yellow background
        if is_reconstructed:
            return YELLOW

        # Normalize weight to 0-1 range (higher weight = darker)
        max_weight = max(GRID_SIZE, weight_value) if weight_value > GRID_SIZE else GRID_SIZE
        min_weight = 1

        # Clamp weight to valid range
        if weight_value < min_weight:
            normalized = 0
        else:
            normalized = (weight_value - min_weight) / (max_weight - min_weight)
            normalized = max(0, min(1, normalized))

        # Create gradient from light blue (low weight) to dark blue (high weight)
        light_intensity = int(255 - (normalized * 120))  # 255 to 135

        # Light blue gradient
        color = (light_intensity, light_intensity + 20, 255)

        # Ensure values are in valid range
        color = tuple(max(0, min(255, c)) for c in color)

        return color

    def get_cell_from_pos(self, pos):
        """Convert mouse position to grid coordinates"""
        x, y = pos
        if x < BORDER or y < BORDER:
            return None, None

        col = (x - BORDER) // (CELL_SIZE + BORDER)
        row = (y - BORDER) // (CELL_SIZE + BORDER)

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return row, col
        return None, None

    def toggle_obstacle(self, row, col):
        """Toggle obstacle at given position"""
        if self.grid[row, col] == 0:
            self.grid[row, col] = 1  # Make obstacle
        else:
            self.grid[row, col] = 0  # Make free
        self.calculate_weights()
        self.calculate_special_areas()
        self.reconstruct_special_weights()

    def is_in_special_area(self, row, col):
        """Check if cell is in special area"""
        for region in self.special_areas:
            if (row, col) in region.cell_list:
                return True
        return False

    def is_in_inner_special_area(self, row, col):
        """Check if cell is in inner special area"""
        for region in self.inner_special_areas:
            if (row, col) in region.cell_list:
                return True
        return False

    def draw_grid(self):
        """Draw the grid with current display mode"""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Calculate cell position
                x = j * (CELL_SIZE + BORDER) + BORDER
                y = i * (CELL_SIZE + BORDER) + BORDER

                # Select weight map based on mode
                if self.display_mode == MODE_RECONSTRUCTED_WEIGHTS:
                    weight_value = self.reconstructed_weight_map[i, j]
                    is_reconstructed = (i, j) in self.reconstructed_cells
                else:
                    weight_value = self.weight_map[i, j]
                    is_reconstructed = False

                # Determine cell color based on display mode
                if self.grid[i, j] == 1:  # Obstacle
                    color = BLACK
                elif self.display_mode == MODE_NORMAL:
                    color = WHITE
                elif self.display_mode == MODE_WEIGHTS:
                    color = self.get_gradient_color(weight_value)
                elif self.display_mode == MODE_SPECIAL_AREAS:
                    if self.is_in_inner_special_area(i, j):
                        color = LIGHT_ORANGE  # Inner special areas
                    elif self.is_in_special_area(i, j):
                        color = LIGHT_BLUE  # Special areas
                    else:
                        color = WHITE  # Normal areas
                elif self.display_mode == MODE_RECONSTRUCTED_WEIGHTS:
                    color = self.get_gradient_color(weight_value, is_reconstructed)

                # Draw cell
                cell_rect = pg.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pg.draw.rect(self.screen, color, cell_rect)
                pg.draw.rect(self.screen, GRAY, cell_rect, 1)  # Border

                # Draw values if appropriate
                if self.display_mode in [MODE_WEIGHTS, MODE_RECONSTRUCTED_WEIGHTS]:
                    # Show weight values
                    if self.grid[i, j] == 1:  # Obstacle - RED text for -1
                        text_color = RED
                    elif is_reconstructed:  # Reconstructed cells - RED text on yellow
                        text_color = RED
                    else:  # Free space - black text on blue gradient
                        text_color = BLACK

                    # Render text with appropriate size
                    if abs(weight_value) >= 10:
                        text = self.small_font.render(str(weight_value), True, text_color)
                    else:
                        text = self.font.render(str(weight_value), True, text_color)

                    # Center text in cell
                    text_rect = text.get_rect()
                    text_rect.center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
                    self.screen.blit(text, text_rect)

    def draw_legend(self, show_instructions=True):
        """Draw legend at bottom"""
        legend_y = GRID_SIZE * (CELL_SIZE + BORDER) + BORDER + 20

        # Mode-specific legend
        if self.display_mode == MODE_NORMAL:
            self._draw_normal_legend(legend_y)
        elif self.display_mode == MODE_WEIGHTS:
            self._draw_weights_legend(legend_y)
        elif self.display_mode == MODE_SPECIAL_AREAS:
            self._draw_special_areas_legend(legend_y)
        elif self.display_mode == MODE_RECONSTRUCTED_WEIGHTS:
            self._draw_reconstructed_legend(legend_y)

        # Show current mode
        mode_names = ["NORMAL", "WEIGHTS", "SPECIAL AREAS", "RECONSTRUCTED"]
        mode_text = self.large_font.render(f"Mode: {mode_names[self.display_mode]}", True, BLUE)
        self.screen.blit(mode_text, (400, legend_y - 5))

        # Instructions
        if show_instructions:
            instruction_text = self.small_font.render("TAB: Cycle modes, SPACE: Screenshot, Click: Toggle walls", True,
                                                      RED)
            self.screen.blit(instruction_text, (20, legend_y + 35))

    def _draw_normal_legend(self, y):
        """Draw legend for normal mode"""
        # Wall
        wall_rect = pg.Rect(20, y, 20, 20)
        pg.draw.rect(self.screen, BLACK, wall_rect)
        wall_text = self.large_font.render("Wall", True, BLACK)
        self.screen.blit(wall_text, (50, y - 5))

        # Free space
        free_rect = pg.Rect(150, y, 20, 20)
        pg.draw.rect(self.screen, WHITE, free_rect)
        pg.draw.rect(self.screen, GRAY, free_rect, 1)
        free_text = self.large_font.render("Free", True, BLACK)
        self.screen.blit(free_text, (180, y - 5))

    def _draw_weights_legend(self, y):
        """Draw legend for weights mode"""
        # Wall
        wall_rect = pg.Rect(20, y, 20, 20)
        pg.draw.rect(self.screen, BLACK, wall_rect)
        wall_text = self.large_font.render("Wall(-1)", True, BLACK)
        self.screen.blit(wall_text, (50, y - 5))

        # High weight
        high_rect = pg.Rect(180, y, 20, 20)
        pg.draw.rect(self.screen, self.get_gradient_color(GRID_SIZE), high_rect)
        high_text = self.large_font.render("High Weight", True, BLACK)
        self.screen.blit(high_text, (210, y - 5))

    def _draw_special_areas_legend(self, y):
        """Draw legend for special areas mode"""
        # Wall
        wall_rect = pg.Rect(20, y, 20, 20)
        pg.draw.rect(self.screen, BLACK, wall_rect)
        wall_text = self.large_font.render("Wall", True, BLACK)
        self.screen.blit(wall_text, (50, y - 5))

        # Special area
        special_rect = pg.Rect(120, y, 20, 20)
        pg.draw.rect(self.screen, LIGHT_BLUE, special_rect)
        pg.draw.rect(self.screen, GRAY, special_rect, 1)
        special_text = self.large_font.render("Trap", True, BLACK)
        self.screen.blit(special_text, (150, y - 5))

        # Inner special area
        inner_rect = pg.Rect(220, y, 20, 20)
        pg.draw.rect(self.screen, LIGHT_ORANGE, inner_rect)
        pg.draw.rect(self.screen, GRAY, inner_rect, 1)
        inner_text = self.large_font.render("Inner", True, BLACK)
        self.screen.blit(inner_text, (250, y - 5))

    def _draw_reconstructed_legend(self, y):
        """Draw legend for reconstructed weights mode"""
        # Wall
        wall_rect = pg.Rect(20, y, 20, 20)
        pg.draw.rect(self.screen, BLACK, wall_rect)
        wall_text = self.large_font.render("Wall(-1)", True, BLACK)
        self.screen.blit(wall_text, (50, y - 5))

        # Reconstructed weight
        recon_rect = pg.Rect(160, y, 20, 20)
        pg.draw.rect(self.screen, YELLOW, recon_rect)
        pg.draw.rect(self.screen, GRAY, recon_rect, 1)
        recon_text = self.large_font.render("Reconstructed", True, BLACK)
        self.screen.blit(recon_text, (190, y - 5))

    def cycle_display_mode(self):
        """Cycle through display modes"""
        self.display_mode = (self.display_mode + 1) % 4
        mode_names = ["NORMAL", "WEIGHTS", "SPECIAL AREAS", "RECONSTRUCTED"]
        print(f"Display mode: {mode_names[self.display_mode]}")

    def save_screenshot(self):
        """Save screenshot without instructions"""
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')

        mode_names = ["normal", "weights", "special", "reconstructed"]
        mode_suffix = mode_names[self.display_mode]

        i = 1
        while os.path.exists(f'screenshots/bwave_{mode_suffix}_{i:03d}.png'):
            i += 1
        filename = f'screenshots/bwave_{mode_suffix}_{i:03d}.png'

        # Create clean surface
        temp_surface = pg.Surface((self.window_width, self.window_height))
        temp_surface.fill(WHITE)

        # Temporarily switch to clean surface
        original_screen = self.screen
        self.screen = temp_surface

        # Draw clean version
        self.draw_grid()
        self.draw_legend(show_instructions=False)

        # Save and restore
        pg.image.save(temp_surface, filename)
        self.screen = original_screen

        print(f"Screenshot saved as {filename}")

    def run(self):
        """Main game loop"""
        clock = pg.time.Clock()
        running = True

        print("BWave Framework Visualizer")
        print("Controls:")
        print("- Left Click: Toggle Wall")
        print("- TAB: Cycle display modes (Normal → Weights → Special Areas → Reconstructed)")
        print("- SPACE: Save screenshot")
        print("- ESC: Exit")

        while running:
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        row, col = self.get_cell_from_pos(event.pos)
                        if row is not None and col is not None:
                            self.toggle_obstacle(row, col)

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_TAB:
                        self.cycle_display_mode()
                    elif event.key == pg.K_SPACE:
                        self.save_screenshot()
                    elif event.key == pg.K_ESCAPE:
                        running = False

            # Clear screen and redraw everything
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_legend(show_instructions=True)

            # Update display
            pg.display.flip()
            clock.tick(60)

        pg.quit()


if __name__ == "__main__":
    visualizer = EnhancedWeightMapVisualizer()
    visualizer.run()