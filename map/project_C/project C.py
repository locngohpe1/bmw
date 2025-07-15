import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import cv2
import matplotlib.pyplot as plt
from collections import deque
import random
import math
import copy
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class DepthwiseSeparableConv2d(nn.Module):
    """Deep Separable Convolution implementation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiHeadDotProductAttention(nn.Module):
    """Multi-Head Dot-Product Attention for Relational Network"""

    def __init__(self, d_model, num_heads=8):
        super(MultiHeadDotProductAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention calculation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        return self.output(attention)


class RelationalNetwork(nn.Module):
    """Relational Network with MHDPA"""

    def __init__(self, input_dim, hidden_dim=128):
        super(RelationalNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Multi-Head Attention
        self.attention = MultiHeadDotProductAttention(input_dim)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: (batch_size, n_entities, feature_dim)
        # Apply attention
        attention_out = self.attention(x)

        # Apply MLP
        mlp_out = self.mlp(attention_out)

        # Residual connection + Layer normalization
        output = self.layer_norm(x + mlp_out)

        return output


class RPPONetwork(nn.Module):
    """RPPO Network with Deep Separable Convolution and Relational Network"""

    def __init__(self, input_channels=3, action_dim=2):
        super(RPPONetwork, self).__init__()

        # Conventional convolution layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)

        # Deep separable convolution
        self.sep_conv = DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate conv output size (assuming 84x84 input)
        self.conv_output_size = self._get_conv_output_size()

        # Relational network
        self.relational_net = RelationalNetwork(64, hidden_dim=128)

        # Actor network (policy)
        self.actor_fc1 = nn.Linear(64, 400)
        self.actor_fc2 = nn.Linear(400, 200)
        self.actor_mean = nn.Linear(200, action_dim)
        self.actor_std = nn.Linear(200, action_dim)

        # Critic network (value function)
        self.critic_fc1 = nn.Linear(64, 400)
        self.critic_fc2 = nn.Linear(400, 200)
        self.critic_value = nn.Linear(200, 1)
        # Move model to device
        self.to(device)

    def _get_conv_output_size(self):
        """Calculate the output size after convolution layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 84, 84)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.sep_conv(x))
            return x.numel()

    def forward(self, state):
        # Convolutional layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.sep_conv(x))

        # Reshape for relational network
        batch_size = x.size(0)
        channels = x.size(1)
        height = x.size(2)
        width = x.size(3)

        # Flatten spatial dimensions and treat as entities
        x = x.view(batch_size, channels, height * width).transpose(1, 2)  # (batch, h*w, channels)

        # Apply relational network
        x = self.relational_net(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, channels)

        # Actor network
        actor_x = F.relu(self.actor_fc1(x))
        actor_x = F.relu(self.actor_fc2(actor_x))
        action_mean = torch.tanh(self.actor_mean(actor_x))
        action_std = F.softplus(self.actor_std(actor_x)) + 1e-5

        # Critic network
        critic_x = F.relu(self.critic_fc1(x))
        critic_x = F.relu(self.critic_fc2(critic_x))
        state_value = self.critic_value(critic_x)

        return action_mean, action_std, state_value


class DataBatch:
    """Data batch processing for handling sample correlation"""

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.log_probs = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))

        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_rewards = [self.rewards[i] for i in indices]
        batch_next_states = [self.next_states[i] for i in indices]
        batch_dones = [self.dones[i] for i in indices]
        batch_log_probs = [self.log_probs[i] for i in indices]
        batch_values = [self.values[i] for i in indices]

        return (batch_states, batch_actions, batch_rewards,
                batch_next_states, batch_dones, batch_log_probs, batch_values)

    def shuffle_and_repeat(self, repeat_factor=2):
        """Shuffle and repeat data to reduce correlation"""
        all_data = list(zip(self.states, self.actions, self.rewards,
                            self.next_states, self.dones, self.log_probs, self.values))

        # Repeat data
        repeated_data = all_data * repeat_factor

        # Shuffle
        random.shuffle(repeated_data)

        # Update buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

        for data in repeated_data:
            self.states.append(data[0])
            self.actions.append(data[1])
            self.rewards.append(data[2])
            self.next_states.append(data[3])
            self.dones.append(data[4])
            self.log_probs.append(data[5])
            self.values.append(data[6])

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class RPPOAgent:
    """RPPO Agent implementation"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2,
                 k_epochs=4, c1=0.5, c2=0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.c1 = c1  # Value function loss coefficient
        self.c2 = c2  # Entropy coefficient

        # Networks
        self.policy = RPPONetwork(input_channels=state_dim[0], action_dim=action_dim).to(device)
        self.policy_old = RPPONetwork(input_channels=state_dim[0], action_dim=action_dim).to(device)

        # Copy parameters to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Data batch
        self.data_batch = DataBatch()

        # Action bounds
        self.action_bounds = {
            'linear_vel': [0.1, 0.5],
            'angular_vel': [-math.pi / 3, math.pi / 3]
        }

    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_mean, action_std, state_value = self.policy_old(state)

        # Create normal distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Apply action bounds
        action_bounded = self._apply_action_bounds(action)

        return action_bounded.squeeze().cpu().numpy(), log_prob.cpu().numpy(), state_value.squeeze().cpu().numpy()

    def _apply_action_bounds(self, action):
        """Apply action bounds to ensure valid actions"""
        action = action.clone()
        # Linear velocity bounds [0.1, 0.5]
        action[:, 0] = torch.clamp(action[:, 0], self.action_bounds['linear_vel'][0],
                                   self.action_bounds['linear_vel'][1])
        # Angular velocity bounds [-π/3, π/3]
        action[:, 1] = torch.clamp(action[:, 1], self.action_bounds['angular_vel'][0],
                                   self.action_bounds['angular_vel'][1])
        return action

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in data batch"""
        self.data_batch.add(state, action, reward, next_state, done, log_prob, value)

    def update(self):
        """Update policy using PPO algorithm"""
        if len(self.data_batch) < 64:  # Minimum batch size
            return 0.0, 0.0, 0.0  # Return default values instead of None

        # Shuffle and repeat data
        self.data_batch.shuffle_and_repeat()

        # Convert to tensors
        states = torch.FloatTensor(np.array(list(self.data_batch.states))).to(device)
        actions = torch.FloatTensor(np.array(list(self.data_batch.actions))).to(device)
        old_log_probs = torch.FloatTensor(np.array(list(self.data_batch.log_probs))).to(device)
        rewards = torch.FloatTensor(np.array(list(self.data_batch.rewards))).to(device)

        # Calculate advantages and target values
        advantages, target_values = self._calculate_advantages_and_targets(rewards)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0

        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_mean, action_std, state_values = self.policy(states)

            # Calculate new log probabilities
            dist = Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            critic_loss = F.mse_loss(state_values.squeeze(), target_values)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            total_loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss

            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Accumulate losses
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()

        # Average losses over epochs
        avg_actor_loss = total_actor_loss / self.k_epochs
        avg_critic_loss = total_critic_loss / self.k_epochs
        avg_entropy_loss = total_entropy_loss / self.k_epochs

        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear data batch
        self.data_batch.clear()

        return avg_actor_loss, avg_critic_loss, avg_entropy_loss

    def _calculate_advantages_and_targets(self, rewards):
        """Calculate advantages and target values using GAE"""
        values = torch.FloatTensor(np.array(list(self.data_batch.values))).to(device)
        dones = torch.BoolTensor(np.array(list(self.data_batch.dones))).to(device)

        advantages = torch.zeros_like(rewards).to(device)
        target_values = torch.zeros_like(rewards).to(device)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            if dones[t]:
                next_value = 0
                gae = 0  # Reset GAE at episode end

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae
            advantages[t] = gae
            target_values[t] = advantages[t] + values[t]

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, target_values


class RobotEnvironment:
    """Simulated robot environment for Active SLAM"""

    def __init__(self, width=500, height=500):
        self.width = width
        self.height = height
        self.robot_pos = np.array([width // 2, height // 2], dtype=float)
        self.robot_angle = 0.0
        self.robot_radius = 10

        # Obstacles
        self.static_obstacles = []
        self.dynamic_obstacles = []

        # Map
        self.map = np.ones((height, width), dtype=np.uint8) * 255  # White background
        self.explored_map = np.zeros((height, width), dtype=np.uint8)

        # SLAM components
        self.trajectory = [self.robot_pos.copy()]
        self.laser_data = []

        # Initialize obstacles
        self._initialize_obstacles()

    def _initialize_obstacles(self):
        """Initialize static and dynamic obstacles"""
        # Static obstacles (rectangles and circles)
        self.static_obstacles = [
            {'type': 'rect', 'pos': [100, 100], 'size': [80, 60], 'color': [0, 0, 255]},  # Red rectangle
            {'type': 'rect', 'pos': [300, 200], 'size': [60, 80], 'color': [0, 255, 0]},  # Green rectangle
            {'type': 'circle', 'pos': [200, 350], 'radius': 40, 'color': [255, 0, 0]},  # Blue circle
            {'type': 'circle', 'pos': [400, 100], 'radius': 35, 'color': [0, 255, 255]},  # Yellow circle
        ]

        # Dynamic obstacles
        self.dynamic_obstacles = [
            {'type': 'circle', 'pos': [150, 250], 'radius': 25, 'velocity': [1, 0.5], 'color': [128, 128, 128]}
        ]

    def reset(self):
        """Reset environment"""
        self.robot_pos = np.array([self.width // 2, self.height // 2], dtype=float)
        self.robot_angle = 0.0
        self.trajectory = [self.robot_pos.copy()]
        self.explored_map = np.zeros((self.height, self.width), dtype=np.uint8)
        return self._get_observation()

    def step(self, action):
        """Execute action and return next state, reward, done"""
        linear_vel, angular_vel = action
        dt = 0.1  # Time step

        # Update robot position
        self.robot_angle += angular_vel * dt
        self.robot_pos[0] += linear_vel * np.cos(self.robot_angle) * dt * 50
        self.robot_pos[1] += linear_vel * np.sin(self.robot_angle) * dt * 50

        # Keep robot in bounds
        self.robot_pos[0] = np.clip(self.robot_pos[0], self.robot_radius, self.width - self.robot_radius)
        self.robot_pos[1] = np.clip(self.robot_pos[1], self.robot_radius, self.height - self.robot_radius)

        # Update dynamic obstacles
        self._update_dynamic_obstacles()

        # Check collision
        collision = self._check_collision()

        # Calculate reward
        reward = self._calculate_reward(action, collision)

        # Update trajectory and exploration
        self.trajectory.append(self.robot_pos.copy())
        self._update_exploration()

        # Get observation
        obs = self._get_observation()

        done = collision or len(self.trajectory) > 1000

        return obs, reward, done

    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles"""
        for obs in self.dynamic_obstacles:
            obs['pos'][0] += obs['velocity'][0]
            obs['pos'][1] += obs['velocity'][1]

            # Bounce off walls
            if obs['pos'][0] <= obs['radius'] or obs['pos'][0] >= self.width - obs['radius']:
                obs['velocity'][0] *= -1
            if obs['pos'][1] <= obs['radius'] or obs['pos'][1] >= self.height - obs['radius']:
                obs['velocity'][1] *= -1

    def _check_collision(self):
        """Check if robot collides with obstacles"""
        robot_x, robot_y = self.robot_pos

        # Check static obstacles
        for obs in self.static_obstacles:
            if obs['type'] == 'rect':
                x, y = obs['pos']
                w, h = obs['size']
                if (x <= robot_x <= x + w) and (y <= robot_y <= y + h):
                    return True
            elif obs['type'] == 'circle':
                x, y = obs['pos']
                r = obs['radius']
                if np.linalg.norm([robot_x - x, robot_y - y]) <= r + self.robot_radius:
                    return True

        # Check dynamic obstacles
        for obs in self.dynamic_obstacles:
            x, y = obs['pos']
            r = obs['radius']
            if np.linalg.norm([robot_x - x, robot_y - y]) <= r + self.robot_radius:
                return True

        return False

    def _calculate_reward(self, action, collision):
        """Calculate reward based on action and environment state"""
        linear_vel, angular_vel = action

        # Collision penalty
        if collision:
            return -10.0

        # Simple rotation penalty
        if abs(angular_vel) > 0.8 and linear_vel < 0.2:
            return -1.0

        # Exploration reward (higher linear velocity is better)
        alpha = 0.1  # Time step factor
        reward = alpha * linear_vel * np.cos(angular_vel)

        return reward

    def _update_exploration(self):
        """Update explored areas"""
        x, y = int(self.robot_pos[0]), int(self.robot_pos[1])
        cv2.circle(self.explored_map, (x, y), 20, 255, -1)

    def _get_observation(self):
        """Get RGB observation of environment"""
        # Create RGB image
        rgb_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb_image[:, :] = [255, 255, 255]  # White background

        # Draw obstacles
        for obs in self.static_obstacles:
            if obs['type'] == 'rect':
                x, y = obs['pos']
                w, h = obs['size']
                cv2.rectangle(rgb_image, (x, y), (x + w, y + h), obs['color'], -1)
            elif obs['type'] == 'circle':
                x, y = obs['pos']
                r = obs['radius']
                cv2.circle(rgb_image, (x, y), r, obs['color'], -1)

        for obs in self.dynamic_obstacles:
            x, y = int(obs['pos'][0]), int(obs['pos'][1])
            r = obs['radius']
            cv2.circle(rgb_image, (x, y), r, obs['color'], -1)

        # Draw robot
        robot_x, robot_y = int(self.robot_pos[0]), int(self.robot_pos[1])
        cv2.circle(rgb_image, (robot_x, robot_y), self.robot_radius, [0, 0, 0], -1)

        # Draw robot direction
        end_x = int(robot_x + 20 * np.cos(self.robot_angle))
        end_y = int(robot_y + 20 * np.sin(self.robot_angle))
        cv2.line(rgb_image, (robot_x, robot_y), (end_x, end_y), [0, 0, 0], 2)

        # Resize to 84x84 for network input
        rgb_image_resized = cv2.resize(rgb_image, (84, 84))

        # Normalize and transpose for PyTorch (C, H, W)
        rgb_image_norm = rgb_image_resized.astype(np.float32) / 255.0
        rgb_image_transposed = np.transpose(rgb_image_norm, (2, 0, 1))

        return rgb_image_transposed

    def render(self):
        """Render current environment state"""
        # Create display image
        display_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        display_img[:, :] = [255, 255, 255]

        # Draw explored areas
        explored_colored = cv2.applyColorMap(self.explored_map, cv2.COLORMAP_COOL)
        mask = self.explored_map > 0
        display_img[mask] = explored_colored[mask]

        # Draw obstacles
        for obs in self.static_obstacles:
            if obs['type'] == 'rect':
                x, y = obs['pos']
                w, h = obs['size']
                cv2.rectangle(display_img, (x, y), (x + w, y + h), obs['color'], -1)
            elif obs['type'] == 'circle':
                x, y = obs['pos']
                r = obs['radius']
                cv2.circle(display_img, (x, y), r, obs['color'], -1)

        for obs in self.dynamic_obstacles:
            x, y = int(obs['pos'][0]), int(obs['pos'][1])
            r = obs['radius']
            cv2.circle(display_img, (x, y), r, obs['color'], -1)

        # Draw trajectory
        for i in range(1, len(self.trajectory)):
            pt1 = tuple(map(int, self.trajectory[i - 1]))
            pt2 = tuple(map(int, self.trajectory[i]))
            cv2.line(display_img, pt1, pt2, [255, 0, 0], 2)

        # Draw robot
        robot_x, robot_y = int(self.robot_pos[0]), int(self.robot_pos[1])
        cv2.circle(display_img, (robot_x, robot_y), self.robot_radius, [0, 0, 0], -1)

        return display_img


class TransferLearning:
    """Transfer Learning implementation"""

    def __init__(self, source_agent, target_agent):
        self.source_agent = source_agent
        self.target_agent = target_agent

    def transfer_weights(self, transfer_ratio=0.8):
        """Transfer weights from source to target network"""
        source_state_dict = self.source_agent.policy.state_dict()
        target_state_dict = self.target_agent.policy.state_dict()

        # Transfer convolutional layers and some fully connected layers
        layers_to_transfer = [
            'conv1.weight', 'conv1.bias',
            'conv2.weight', 'conv2.bias',
            'sep_conv.depthwise.weight', 'sep_conv.pointwise.weight',
            'relational_net.attention.query.weight', 'relational_net.attention.query.bias',
            'relational_net.attention.key.weight', 'relational_net.attention.key.bias',
            'relational_net.attention.value.weight', 'relational_net.attention.value.bias'
        ]

        for layer_name in layers_to_transfer:
            if layer_name in source_state_dict and layer_name in target_state_dict:
                # Weighted transfer
                target_state_dict[layer_name] = (
                        transfer_ratio * source_state_dict[layer_name] +
                        (1 - transfer_ratio) * target_state_dict[layer_name]
                )

        # Load transferred weights
        self.target_agent.policy.load_state_dict(target_state_dict)
        self.target_agent.policy_old.load_state_dict(target_state_dict)

        logger.info("Transfer learning completed!")


class ActiveSLAMTrainer:
    """Main trainer for Active SLAM with RPPO"""

    def __init__(self, env, agent, max_episodes=1000, max_steps=500):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []

    def train(self):
        """Train the RPPO agent"""
        logger.info("Starting Active SLAM training...")

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(self.max_steps):
                # Select action
                action, log_prob, value = self.agent.select_action(state)

                # Execute action
                next_state, reward, done = self.env.step(action)

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done, log_prob, value)

                # Update metrics
                episode_reward += reward
                episode_length += 1

                state = next_state

                if done:
                    break

            # Update agent when we have enough data
            if len(self.agent.data_batch) >= 64:
                actor_loss, critic_loss, entropy_loss = self.agent.update()
                if actor_loss != 0.0:  # Only record if actual update occurred
                    self.actor_losses.append(actor_loss)
                    self.critic_losses.append(critic_loss)

            # Record episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Log progress
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(
                    self.episode_rewards)
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Episode Length: {episode_length}")

        logger.info("Training completed!")

    def test(self, num_episodes=10):
        """Test the trained agent"""
        logger.info("Testing trained agent...")

        test_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                action, _, _ = self.agent.select_action(state)
                state, reward, done = self.env.step(action)
                episode_reward += reward

                if done:
                    break

            test_rewards.append(episode_reward)
            logger.info(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")

        avg_test_reward = np.mean(test_rewards)
        logger.info(f"Average Test Reward: {avg_test_reward:.2f}")

        return avg_test_reward

    def plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')

        # Actor losses
        if self.actor_losses:
            axes[1, 0].plot(self.actor_losses)
            axes[1, 0].set_title('Actor Losses')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')

        # Critic losses
        if self.critic_losses:
            axes[1, 1].plot(self.critic_losses)
            axes[1, 1].set_title('Critic Losses')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')

        plt.tight_layout()
        plt.show()


def main():
    """Main function to run Active SLAM training"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Environment setup
    env = RobotEnvironment(width=500, height=500)
    state_dim = (3, 84, 84)  # RGB image
    action_dim = 2  # Linear and angular velocity

    # Create agents
    source_agent = RPPOAgent(state_dim, action_dim, lr=3e-4)
    target_agent = RPPOAgent(state_dim, action_dim, lr=3e-4)

    # Train source agent (simple environment)
    logger.info("Training source agent in simple environment...")
    source_trainer = ActiveSLAMTrainer(env, source_agent, max_episodes=500)
    source_trainer.train()

    # Transfer learning
    logger.info("Applying transfer learning...")
    transfer_learning = TransferLearning(source_agent, target_agent)
    transfer_learning.transfer_weights()

    # Train target agent (complex environment with transferred weights)
    logger.info("Training target agent with transfer learning...")
    target_trainer = ActiveSLAMTrainer(env, target_agent, max_episodes=300)
    target_trainer.train()

    # Test final agent
    logger.info("Testing final agent...")
    target_trainer.test()

    # Plot results
    target_trainer.plot_training_progress()

    logger.info("Active SLAM training with RPPO completed successfully!")


if __name__ == "__main__":
    main()