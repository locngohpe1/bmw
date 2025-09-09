import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Common parameters
robot_pos = np.array([2, 2])
obstacle_pos = np.array([6, 4])
alpha_angle = np.pi/8  # Small angle as requested

# Calculate collision point and vectors
d_current = np.linalg.norm(obstacle_pos - robot_pos)

# Robot velocity (direction toward collision)
vr_direction = np.array([np.cos(alpha_angle/2), np.sin(alpha_angle/2)])
vr_magnitude = 1.5
vr = vr_direction * vr_magnitude

# Obstacle velocity (slightly different angle)
vk_direction = np.array([np.cos(-alpha_angle/2), np.sin(-alpha_angle/2)])
vk_magnitude = 1.2
vk = vk_direction * vk_magnitude

# Calculate collision point C (approximate)
t0_approx = 2.0  # For visualization
collision_point = robot_pos + vr * t0_approx

# Subfigure (a): Collision Prediction Geometry
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Draw robot
robot_circle = plt.Circle(robot_pos, 0.3, color='blue', fill=True, alpha=0.8)
ax1.add_patch(robot_circle)
ax1.annotate('A', robot_pos, xytext=(robot_pos[0]-0.7, robot_pos[1]-0.5),
             fontsize=12, fontweight='bold', ha='center')
ax1.annotate(r'$(x_r, y_r)$', robot_pos, xytext=(robot_pos[0]-0.5, robot_pos[1]-1.0),
             fontsize=10, ha='center')

# Draw obstacle
obstacle_rect = patches.Rectangle(obstacle_pos-0.3, 0.6, 0.6,
                                 color='red', fill=True, alpha=0.8)
ax1.add_patch(obstacle_rect)
ax1.annotate('B', obstacle_pos, xytext=(obstacle_pos[0]+0.5, obstacle_pos[1]+0.5),
             fontsize=12, fontweight='bold', ha='center')
ax1.annotate(r'$(x_k, y_k)$', obstacle_pos, xytext=(obstacle_pos[0]+0.5, obstacle_pos[1]-0.7),
             fontsize=10, ha='center')

# Draw collision point
ax1.plot(collision_point[0], collision_point[1], '*', color='gold', markersize=15)
ax1.annotate('C', collision_point, xytext=(collision_point[0]+0.3, collision_point[1]+0.3),
             fontsize=12, fontweight='bold', ha='center')

# Draw velocity vectors
arrow_vr = FancyArrowPatch(robot_pos, robot_pos + vr,
                          arrowstyle='->', color='blue', linewidth=2.5)
ax1.add_patch(arrow_vr)
ax1.annotate(r'$\vec{v_r}$', robot_pos + vr/2, xytext=(robot_pos[0]+1.2, robot_pos[1]+1.2),
             fontsize=12, color='blue', fontweight='bold')

arrow_vk = FancyArrowPatch(obstacle_pos, obstacle_pos + vk,
                          arrowstyle='->', color='red', linewidth=2.5)
ax1.add_patch(arrow_vk)
ax1.annotate(r'$\vec{v_k}$', obstacle_pos + vk/2, xytext=(obstacle_pos[0]+0.8, obstacle_pos[1]+0.8),
             fontsize=12, color='red', fontweight='bold')

# Draw distance d_current
ax1.plot([robot_pos[0], obstacle_pos[0]], [robot_pos[1], obstacle_pos[1]],
         'k--', linewidth=1.5, alpha=0.7)
mid_point = (robot_pos + obstacle_pos) / 2
ax1.annotate(r'$d_{current}$', mid_point, xytext=(mid_point[0], mid_point[1]-0.5),
             fontsize=12, ha='center', fontweight='bold')

# Draw angle α
angle_arc = Arc(robot_pos, 1.5, 1.5, angle=0, theta1=0, theta2=np.degrees(alpha_angle),
               color='green', linewidth=2)
ax1.add_patch(angle_arc)
ax1.annotate(r'$\alpha$', robot_pos + np.array([0.8, 0.3]), fontsize=12,
             color='green', fontweight='bold')

# Draw collision triangle (dashed)
triangle_points = np.array([robot_pos, obstacle_pos, collision_point, robot_pos])
ax1.plot(triangle_points[:, 0], triangle_points[:, 1], 'g:', linewidth=1.5, alpha=0.6)

ax1.set_xlabel('X Position (cells)', fontsize=12)
ax1.set_ylabel('Y Position (cells)', fontsize=12)
ax1.set_title('(a) Collision Prediction Geometry', fontsize=14, fontweight='bold')

# Subfigure (b): Triangle Constraint Verification
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Same basic elements
robot_circle2 = plt.Circle(robot_pos, 0.3, color='blue', fill=True, alpha=0.8)
ax2.add_patch(robot_circle2)
obstacle_rect2 = patches.Rectangle(obstacle_pos-0.3, 0.6, 0.6,
                                  color='red', fill=True, alpha=0.8)
ax2.add_patch(obstacle_rect2)
ax2.plot(collision_point[0], collision_point[1], '*', color='gold', markersize=15)

# Labels
ax2.annotate('A', robot_pos, xytext=(robot_pos[0]-0.7, robot_pos[1]-0.5),
             fontsize=12, fontweight='bold', ha='center')
ax2.annotate('B', obstacle_pos, xytext=(obstacle_pos[0]+0.5, obstacle_pos[1]+0.5),
             fontsize=12, fontweight='bold', ha='center')
ax2.annotate('C', collision_point, xytext=(collision_point[0]+0.3, collision_point[1]+0.3),
             fontsize=12, fontweight='bold', ha='center')

# Highlight triangle sides with different colors and thickness
# Side 1: Robot to collision point
ax2.plot([robot_pos[0], collision_point[0]], [robot_pos[1], collision_point[1]],
         'b-', linewidth=4, alpha=0.8, label=r'$\|\vec{v_r}\| \times t_0$')

# Side 2: Obstacle to collision point
ax2.plot([obstacle_pos[0], collision_point[0]], [obstacle_pos[1], collision_point[1]],
         'r-', linewidth=4, alpha=0.8, label=r'$\|\vec{v_k}\| \times t_0$')

# Side 3: Robot to obstacle (initial distance)
ax2.plot([robot_pos[0], obstacle_pos[0]], [robot_pos[1], obstacle_pos[1]],
         'k-', linewidth=4, alpha=0.8, label=r'$d_{current}$')

# Triangle inequality constraints as text box
constraint_text = r'Triangle Constraints:' + '\n' + \
                  r'$\|\vec{v_r}\| \times t_0 + \|\vec{v_k}\| \times t_0 > d_{current}$' + '\n' + \
                  r'$\|\vec{v_r}\| \times t_0 + d_{current} > \|\vec{v_k}\| \times t_0$' + '\n' + \
                  r'$\|\vec{v_k}\| \times t_0 + d_{current} > \|\vec{v_r}\| \times t_0$'

ax2.text(0.5, 6.5, constraint_text, fontsize=11,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
         verticalalignment='top')

ax2.set_xlabel('X Position (cells)', fontsize=12)
ax2.set_ylabel('Y Position (cells)', fontsize=12)
ax2.set_title('(b) Triangle Constraint Verification', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# Additional figure for stopping distance visualization
fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))

ax3.set_xlim(0, 12)
ax3.set_ylim(0, 8)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

# Robot trajectory
trajectory_x = np.linspace(robot_pos[0], collision_point[0], 50)
trajectory_y = np.linspace(robot_pos[1], collision_point[1], 50)

# Stopping position
stop_ratio = 0.7  # Visualize Sr distance
stop_idx = int(len(trajectory_x) * stop_ratio)
stop_pos = np.array([trajectory_x[stop_idx], trajectory_y[stop_idx]])

# Draw elements
robot_circle3 = plt.Circle(robot_pos, 0.3, color='blue', fill=True, alpha=0.8)
ax3.add_patch(robot_circle3)
ax3.plot(collision_point[0], collision_point[1], '*', color='gold', markersize=15)
ax3.plot(stop_pos[0], stop_pos[1], 'o', color='green', markersize=10)

# Trajectory line
ax3.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, alpha=0.7)

# Distance annotations
ax3.annotate('', xy=stop_pos, xytext=robot_pos,
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.annotate(r'$S_r$', (robot_pos + stop_pos)/2, xytext=((robot_pos[0] + stop_pos[0])/2,
             (robot_pos[1] + stop_pos[1])/2 - 0.4), fontsize=12, ha='center',
             color='green', fontweight='bold')

ax3.annotate('', xy=collision_point, xytext=stop_pos,
             arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
ax3.annotate(r'$L_r$', (stop_pos + collision_point)/2,
             xytext=((stop_pos[0] + collision_point[0])/2 + 0.3,
             (stop_pos[1] + collision_point[1])/2 + 0.3), fontsize=12, ha='center',
             color='orange', fontweight='bold')

# Labels
ax3.annotate('Robot Start', robot_pos, xytext=(robot_pos[0]-0.5, robot_pos[1]-0.8),
             fontsize=11, ha='center')
ax3.annotate('Collision Point C', collision_point, xytext=(collision_point[0]+0.5, collision_point[1]+0.5),
             fontsize=11, ha='center')
ax3.annotate('Stop Position', stop_pos, xytext=(stop_pos[0]+0.5, stop_pos[1]-0.8),
             fontsize=11, ha='center', color='green')

# Formula
formula_text = r'$S_r = \|\vec{v_r}\| \times t_0 - L_r$'
ax3.text(8, 6, formula_text, fontsize=14,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.9),
         fontweight='bold')

ax3.set_xlabel('X Position (cells)', fontsize=12)
ax3.set_ylabel('Y Position (cells)', fontsize=12)
ax3.set_title('Stopping Distance Calculation', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()