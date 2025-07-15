import math
import numpy as np
import pygame as pg
import time
import threading
import contextlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import cv2
import logging

# Import Project A components (keep environment)
from grid_map import Grid_Map, EPSILON
from dynamic_obstacles_manager import DynamicObstaclesManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse arguments
parser = argparse.ArgumentParser(description='RPPO Active SLAM with Project A Environment')
parser.add_argument('--map', type=str, default='map/real_map/denmark.txt', help='Path to map file')
parser.add_argument('--speed', type=float, default=0.1, help='Speed of dynamic obstacles')
parser.add_argument('--energy', type=float, default=1000, help='Energy capacity')
args = parser.parse_args()

ENERGY_CAPACITY = args.energy
FPS = 40

# Global variables from Project A
total_travel_length = 0
coverage_length, retreat_length, advance_length = 0, 0, 0
return_charge_count = 1
count_cell_go_through = 1
execute_time = time.time()


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Convolution from Project C"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class RelationalNetwork(nn.Module):
    """Relational Network component for RPPO"""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Multi-head attention components
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Multi-head dot-product attention
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention weights
        attention_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim), dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, V)

        # Residual connection and layer norm
        attended = self.layer_norm(attended + self.mlp(attended))

        return attended


class RPPONetwork(nn.Module):
    """RPPO Network with Relational Network and Depthwise Separable Conv"""

    def __init__(self, input_channels=3, action_dim=2):
        super().__init__()

        # Convolutional layers with depthwise separable conv
        self.conv1 = nn.Conv2d(input_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.sep_conv = DepthwiseSeparableConv2d(64, 64, 3, 1, 1)

        # Calculate feature size after conv layers (for 84x84 input)
        self.feature_size = 64 * 9 * 9

        # Relational network
        self.relational = RelationalNetwork(input_dim=64, hidden_dim=128)

        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_size, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )

        # Actor head
        self.actor_mean = nn.Linear(200, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.5)

        # Critic head
        self.critic = nn.Linear(200, 1)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.sep_conv(x))

        # Flatten for relational processing
        batch_size = x.size(0)
        x_flat = x.view(batch_size, 64, -1).transpose(1, 2)  # (batch, seq_len, 64)

        # Apply relational network
        x_rel = self.relational(x_flat)
        x_rel = x_rel.mean(dim=1)  # Global average pooling

        # Reconstruct feature vector
        x = x.view(batch_size, -1)  # Original flattened features

        # Shared processing
        shared = self.shared_fc(x)

        # Actor output
        action_mean = torch.tanh(self.actor_mean(shared))
        action_std = F.softplus(self.actor_std.expand_as(action_mean)) + 1e-3

        # Critic output
        value = self.critic(shared)

        return action_mean, action_std, value


class RPPOAgent:
    """RPPO Agent with Data Batch Processing"""

    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.exploration_noise = 0.5

        # Networks
        self.policy = RPPONetwork().to(device)
        self.policy_old = RPPONetwork().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer with scheduling
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        # Data batch processing buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_mean, action_std, value = self.policy_old(state)

        dist = Normal(action_mean, action_std)
        action = dist.sample()

        # Add exploration noise during training
        if self.policy.training:
            noise = torch.randn_like(action) * self.exploration_noise * 0.1
            action = action + noise

        log_prob = dist.log_prob(action).sum(-1)

        # Apply action bounds smoothly
        action[0, 0] = torch.sigmoid(action[0, 0]) * 0.4 + 0.1  # Linear velocity [0.1, 0.5]
        action[0, 1] = torch.tanh(action[0, 1]) * (math.pi / 3)  # Angular velocity [-π/3, π/3]

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0, 0]

    def store(self, state, action, reward, log_prob, value, done):
        """Store experience for batch processing"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        """Update policy using PPO with data batch processing"""
        if len(self.states) < 32:
            return 0.0, 0.0

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        values = torch.FloatTensor(self.values).to(device)
        dones = torch.FloatTensor(self.dones).to(device)

        # Calculate returns and advantages with GAE
        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] * (1 - dones[t + 1])

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t])
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Data batch processing: shuffle and repeat
        dataset_size = len(states)
        indices = torch.randperm(dataset_size).to(device)

        # PPO update with multiple epochs
        total_policy_loss = 0
        total_value_loss = 0

        for epoch in range(self.k_epochs):
            # Shuffle data for each epoch
            shuffled_indices = indices[torch.randperm(dataset_size)]

            # Process in batches
            batch_size = 64
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = shuffled_indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                action_mean, action_std, state_values = self.policy(batch_states)

                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # PPO loss calculation
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(state_values.squeeze(), batch_returns)

                # Total loss with entropy bonus
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Update learning rate
        self.scheduler.step()

        # Decay exploration noise
        self.exploration_noise = max(0.1, self.exploration_noise * 0.995)

        # Clear buffer
        self.clear_buffer()

        total_batches = (len(states) // 64 + 1) * self.k_epochs
        return total_policy_loss / total_batches, total_value_loss / total_batches

    def clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


class Robot:
    """Robot class integrating RPPO with Project A environment"""

    def __init__(self, battery_pos, map_row_count, map_col_count):
        self.current_pos = battery_pos
        self.battery_pos = battery_pos
        self.energy = ENERGY_CAPACITY

        # RPPO Agent
        self.agent = RPPOAgent(lr=3e-4)

        # Environment state
        self.map = None
        self.angle = math.pi / 2
        self.move_status = 0  # 0: normal, 1: retreat, 2: charge, 3: advance

        # Metrics
        self.total_moves = 0
        self.visited_cells = set()

        # Episode management
        self.episode_steps = 0
        self.max_episode_steps = 500
        self.episode_reward = 0

    def set_map(self, environment):
        """Set environment map"""
        row_count, col_count = len(environment), len(environment[0])
        self.map = np.full((row_count, col_count), 'u')

        for x in range(len(environment)):
            for y in range(len(environment[0])):
                if environment[x, y] == 1:
                    self.map[x, y] = 'o'

    def get_rgb_observation(self):
        """Generate RGB observation for RPPO"""
        # Create 84x84 RGB image
        img = np.ones((84, 84, 3), dtype=np.float32) * 0.5  # Gray background

        # Get map dimensions
        if self.map is not None:
            map_h, map_w = self.map.shape

            # Draw explored areas in green
            for cell in self.visited_cells:
                cx, cy = cell[0] * 10, cell[1] * 10
                x1, y1 = int(cx * 84 / (map_w * 10)), int(cy * 84 / (map_h * 10))
                x2, y2 = int((cx + 10) * 84 / (map_w * 10)), int((cy + 10) * 84 / (map_h * 10))
                if 0 <= x1 < 84 and 0 <= y1 < 84:
                    img[y1:min(y2, 84), x1:min(x2, 84)] = [0, 1, 0]  # Green

            # Draw obstacles
            for x in range(map_h):
                for y in range(map_w):
                    if self.map[x, y] in ('o', 1):  # Static obstacles
                        px = int(y * 84 / map_w)
                        py = int(x * 84 / map_h)
                        if 0 <= px < 84 and 0 <= py < 84:
                            img[py, px] = [1, 0, 0]  # Red
                    elif self.map[x, y] == 'd':  # Dynamic obstacles
                        px = int(y * 84 / map_w)
                        py = int(x * 84 / map_h)
                        if 0 <= px < 84 and 0 <= py < 84:
                            img[py, px] = [1, 0.5, 0]  # Orange

        # Draw robot
        if hasattr(self, 'current_pos'):
            rx = int(self.current_pos[1] * 84 / (map_w * 10)) if 'map_w' in locals() else 42
            ry = int(self.current_pos[0] * 84 / (map_h * 10)) if 'map_h' in locals() else 42
            if 0 <= rx < 84 and 0 <= ry < 84:
                img[max(0, ry - 2):min(84, ry + 3), max(0, rx - 2):min(84, rx + 3)] = [0, 0, 1]  # Blue

        return img.transpose(2, 0, 1)  # CHW format for PyTorch

    def calculate_reward(self, action, collision, energy_depleted):
        """Calculate reward for RPPO training"""
        if collision:
            return -5.0

        if energy_depleted:
            return -3.0

        linear_vel, angular_vel = action

        # Base survival reward
        reward = 0.5

        # Movement reward
        if linear_vel > 0.2:
            reward += linear_vel * 3.0

        # Exploration bonus
        current_cell = (int(self.current_pos[0] // 10), int(self.current_pos[1] // 10))
        if current_cell not in self.visited_cells:
            self.visited_cells.add(current_cell)
            reward += 4.0  # Exploration bonus

        # Energy efficiency
        if self.energy > ENERGY_CAPACITY * 0.8:
            reward += 0.5
        elif self.energy < ENERGY_CAPACITY * 0.2:
            reward -= 1.0

        # Penalize excessive rotation without movement
        if abs(angular_vel) > 0.5 and linear_vel < 0.15:
            reward -= 0.3

        # Step penalty for efficiency
        reward -= 0.01

        return reward

    def execute_action(self, action):
        """Execute action and return environment feedback"""
        global total_travel_length, coverage_length, count_cell_go_through

        linear_vel, angular_vel = action
        dt = 0.1

        # Update robot orientation
        self.angle += angular_vel * dt

        # Calculate movement
        dx = linear_vel * np.cos(self.angle) * dt * 10
        dy = linear_vel * np.sin(self.angle) * dt * 10

        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)

        # Check collisions and bounds
        collision = False
        row_count, col_count = len(self.map), len(self.map[0])

        if (0 <= new_pos[0] < row_count and 0 <= new_pos[1] < col_count):
            if self.map[int(new_pos[0]), int(new_pos[1])] not in ('o', 1, 'd'):
                # Valid move
                dist = math.sqrt(dx ** 2 + dy ** 2)
                energy_cost = dist * 1.0  # 1 unit per distance unit

                if self.energy >= energy_cost:
                    self.current_pos = new_pos
                    self.energy -= energy_cost
                    total_travel_length += dist
                    coverage_length += dist
                    count_cell_go_through += 1

                    # Mark cell as visited
                    self.map[int(new_pos[0]), int(new_pos[1])] = 'e'
                else:
                    # Energy depleted, need to charge
                    return True, True  # collision=True, energy_depleted=True
            else:
                collision = True
        else:
            collision = True

        # Check if energy is critically low
        energy_depleted = self.energy < ENERGY_CAPACITY * 0.1

        return collision, energy_depleted

    def run(self):
        """Main RPPO Active SLAM loop"""
        global execute_time
        clock = pg.time.Clock()
        run = True
        pause = False

        # Training parameters
        episode = 0
        max_episodes = 1000

        # Episode management
        episode_start_time = time.time()
        last_time = time.time()

        logger.info("Starting RPPO Active SLAM training...")

        while run and episode < max_episodes:
            # Delta time for dynamic obstacles
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Update dynamic obstacles
            if 'dynamic_obstacles' in globals():
                dynamic_obstacles.update(delta_time)

            # Get current observation
            state = self.get_rgb_observation()

            # Select action using RPPO
            action, log_prob, value = self.agent.select_action(state)

            # Execute action
            collision, energy_depleted = self.execute_action(action)

            # Calculate reward
            reward = self.calculate_reward(action, collision, energy_depleted)
            self.episode_reward += reward

            # Check episode termination
            done = collision or energy_depleted or self.episode_steps >= self.max_episode_steps

            # Store experience
            self.agent.store(state, action, reward, log_prob, value, done)

            self.episode_steps += 1
            self.total_moves += 1

            # Update UI
            ui.update_vehicle_pos((int(self.current_pos[0]), int(self.current_pos[1])))
            ui.set_energy_display(self.energy)
            ui.draw()

            if 'dynamic_obstacles' in globals():
                dynamic_obstacles.draw(ui.WIN)

            # Display training info
            if self.episode_steps % 50 == 0:
                training_text = f"Episode: {episode}, Step: {self.episode_steps}, Reward: {self.episode_reward:.2f}, Energy: {self.energy:.1f}"
                font = pg.font.SysFont(None, 24)
                text_img = font.render(training_text, True, (255, 255, 255))
                ui.WIN.blit(text_img, (10, 10))

            pg.display.flip()
            clock.tick(FPS)

            # Handle events
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        pause = not pause
                    elif event.key == pg.K_s:  # Save model
                        torch.save(self.agent.policy.state_dict(), 'rppo_active_slam_model.pth')
                        logger.info("Model saved!")
                if event.type == pg.QUIT:
                    run = False

            if pause:
                continue

            # Episode reset
            if done:
                # Update RPPO policy
                if len(self.agent.states) >= 64:
                    policy_loss, value_loss = self.agent.update()
                    if episode % 10 == 0:
                        logger.info(f"Episode {episode}: Reward={self.episode_reward:.2f}, "
                                    f"Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}, "
                                    f"Explored={len(self.visited_cells)} cells")

                # Reset for next episode
                episode += 1
                self.current_pos = self.battery_pos
                self.energy = ENERGY_CAPACITY
                self.episode_steps = 0
                self.episode_reward = 0
                self.angle = math.pi / 2

                # Reset map exploration state
                if self.map is not None:
                    for x in range(len(self.map)):
                        for y in range(len(self.map[0])):
                            if self.map[x, y] == 'e':
                                self.map[x, y] = 'u'

                self.visited_cells.clear()

        execute_time = time.time() - execute_time
        logger.info(f"Training completed! Total time: {execute_time:.2f}s")

        # Final model save
        torch.save({
            'policy_state_dict': self.agent.policy.state_dict(),
            'total_episodes': episode,
            'total_moves': self.total_moves
        }, 'final_rppo_active_slam_model.pth')
        logger.info("Final model saved!")


def main():
    """Main function"""
    global ui, dynamic_obstacles, execute_time

    # Initialize UI (Project A environment)
    ui = Grid_Map()
    ui.read_map(args.map)
    ENVIRONMENT, battery_pos = ui.edit_map()

    ROW_COUNT = len(ENVIRONMENT)
    COL_COUNT = len(ENVIRONMENT[0])

    # Initialize robot with RPPO
    robot = Robot(battery_pos, ROW_COUNT, COL_COUNT)
    robot.set_map(ENVIRONMENT)

    # Initialize dynamic obstacles manager (Project A)
    dynamic_obstacles = DynamicObstaclesManager(ui, num_obstacles=0, speed_factor=args.speed)

    # Initialize manual obstacles if any
    if hasattr(ui, 'dynamic_obstacles') and ui.dynamic_obstacles:
        dynamic_obstacles.initialize_obstacles()
        logger.info(f"Initialized {len(ui.dynamic_obstacles)} manual dynamic obstacles")

    logger.info("Using RPPO Active SLAM with Project A Environment")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Start training
    execute_time = time.time()
    robot.run()

    # Print final statistics
    print('\n=== RPPO Active SLAM Results ===')
    print(f'Total distance traveled: {total_travel_length:.2f}')
    print(f'Total explored cells: {len(robot.visited_cells)}')
    print(f'Total training time: {execute_time:.2f}s')
    print(f'Average moves per second: {robot.total_moves / execute_time:.2f}')


if __name__ == "__main__":
    main()