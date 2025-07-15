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
import logging

# Configure logging and device
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class RPPONetwork(nn.Module):
    def __init__(self, input_channels=3, action_dim=2):
        super().__init__()
        # Simplified CNN
        self.conv1 = nn.Conv2d(input_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.sep_conv = DepthwiseSeparableConv2d(64, 64, 3, 1, 1)

        # Calculate feature size
        self.feature_size = 64 * 9 * 9  # After conv layers on 84x84 input

        # Shared layers
        self.shared_fc = nn.Linear(self.feature_size, 256)

        # Actor head
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.5)

        # Critic head
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.sep_conv(x))
        x = x.view(x.size(0), -1)

        shared = F.relu(self.shared_fc(x))

        # Actor output
        action_mean = torch.tanh(self.actor_mean(shared))
        action_std = F.softplus(self.actor_std.expand_as(action_mean)) + 1e-3

        # Critic output
        value = self.critic(shared)

        return action_mean, action_std, value

class RPPOAgent:
    def __init__(self, lr=1e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):  # Reduced learning rate
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.exploration_noise = 0.5  # Add exploration noise

        self.policy = RPPONetwork().to(device)
        self.policy_old = RPPONetwork().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        # Buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
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

        # Apply bounds more smoothly
        action[0, 0] = torch.sigmoid(action[0, 0]) * 0.4 + 0.1  # [0.1, 0.5]
        action[0, 1] = torch.tanh(action[0, 1]) * (math.pi / 3)  # [-π/3, π/3]

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0, 0]

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        if len(self.states) < 32:
            return 0.0, 0.0

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        values = torch.FloatTensor(self.values).to(device)
        dones = torch.FloatTensor(self.dones).to(device)

        # Calculate returns and advantages
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

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.k_epochs):
            action_mean, action_std, state_values = self.policy(states)

            dist = Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(state_values.squeeze(), returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

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

        return total_policy_loss / self.k_epochs, total_value_loss / self.k_epochs

    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

class RobotEnvironment:
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        self.width = width
        self.height = height

        # Static obstacles
        self.obstacles = [
            {'pos': [50, 50], 'size': [30, 40]},
            {'pos': [120, 80], 'size': [25, 25]},
            {'pos': [80, 140], 'size': [35, 20]},
        ]

        # Initialize state variables
        self.robot_pos = np.array([30.0, 30.0])
        self.robot_angle = 0.0
        self.steps = 0
        self.visited_cells = set()
        self.previous_pos = np.array([30.0, 30.0])

        self.reset()

    def reset(self):
        self.robot_pos = np.array([30.0, 30.0])
        self.robot_angle = 0.0
        self.steps = 0
        self.visited_cells = set()
        self.previous_pos = self.robot_pos.copy()
        return self._get_obs()

    def step(self, action):
        linear_vel, angular_vel = action
        dt = 0.1

        # Update robot
        self.robot_angle += angular_vel * dt
        dx = linear_vel * np.cos(self.robot_angle) * dt * 20
        dy = linear_vel * np.sin(self.robot_angle) * dt * 20

        new_pos = self.robot_pos + [dx, dy]
        collision = False
        # Check bounds
        if 10 <= new_pos[0] <= self.width - 10 and 10 <= new_pos[1] <= self.height - 10:
            # Check obstacles
            for obs in self.obstacles:
                ox, oy = obs['pos']
                ow, oh = obs['size']
                if (ox <= new_pos[0] <= ox + ow) and (oy <= new_pos[1] <= oy + oh):
                    collision = True
                    break

            if not collision:
                self.robot_pos = new_pos
        else:
            collision = True  # Hit boundary

        # Update exploration
        cell = (int(self.robot_pos[0] // 10), int(self.robot_pos[1] // 10))
        self.visited_cells.add(cell)
        self.previous_pos = self.robot_pos.copy()

        self.steps += 1

        # Calculate reward
        reward = self._calculate_reward(action, collision)
        done = collision or self.steps >= 200

        return self._get_obs(), reward, done

    def _calculate_reward(self, action, collision):
        if collision:
            return -3.0  # Reduced collision penalty

        linear_vel, angular_vel = action

        # Base survival reward
        reward = 0.5

        # Strong reward for forward movement
        if linear_vel > 0.2:
            reward += linear_vel * 5.0  # Increased movement reward

        # Reduced penalty for turning
        if abs(angular_vel) > 0.5 and linear_vel < 0.15:
            reward -= 0.2

        # Exploration bonus
        current_cell = (int(self.robot_pos[0] // 15), int(self.robot_pos[1] // 15))  # Larger cells
        if current_cell not in self.visited_cells:
            reward += 3.0  # Big exploration bonus

        # Distance traveled reward
        if hasattr(self, 'previous_pos'):
            distance_moved = np.linalg.norm(self.robot_pos - self.previous_pos)
            reward += distance_moved * 0.1

        # Step penalty to encourage efficiency
        reward -= 0.05

        return reward

    def _get_obs(self):
        # Create simple RGB observation
        img = np.ones((84, 84, 3), dtype=np.float32) * 0.5  # Gray background

        # Draw visited areas in green
        for cell in self.visited_cells:
            cx, cy = cell[0] * 10, cell[1] * 10
            x1, y1 = int(cx * 84 / 200), int(cy * 84 / 200)
            x2, y2 = int((cx + 10) * 84 / 200), int((cy + 10) * 84 / 200)
            img[y1:y2, x1:x2] = [0, 1, 0]  # Green for explored

        # Draw obstacles (scaled)
        for obs in self.obstacles:
            x, y = obs['pos']
            w, h = obs['size']
            x1, y1 = int(x * 84/200), int(y * 84/200)
            x2, y2 = int((x+w) * 84/200), int((y+h) * 84/200)
            img[y1:y2, x1:x2] = [1, 0, 0]  # Red obstacles

        # Draw robot
        rx, ry = int(self.robot_pos[0] * 84/200), int(self.robot_pos[1] * 84/200)
        img[max(0,ry-2):ry+3, max(0,rx-2):rx+3] = [0, 0, 1]  # Blue robot

        return img.transpose(2, 0, 1)  # CHW format

class ActiveSLAMTrainer:
    def __init__(self, agent, env, episodes=300):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.rewards = []

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(300):  # Increased max steps
                action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)

                self.agent.store(state, action, reward, log_prob, value, done)

                episode_reward += reward
                state = next_state

                if done:
                    break

                # Update every episode if enough data
            if len(self.agent.states) >= 64:  # Increased batch size
                policy_loss, value_loss = self.agent.update()

            self.rewards.append(episode_reward)

            if episode % 25 == 0:  # More frequent logging
                avg_reward = np.mean(self.rewards[-25:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Exploration: {self.agent.exploration_noise:.3f}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    env = RobotEnvironment()

    # Train source agent
    source_agent = RPPOAgent(lr=3e-4)
    source_trainer = ActiveSLAMTrainer(source_agent, env, episodes=400)  # More episodes
    logger.info("Training source agent...")
    source_trainer.train()

    # Transfer learning
    target_agent = RPPOAgent(lr=1e-4)  # Lower learning rate for fine-tuning
    target_agent.policy.load_state_dict(source_agent.policy.state_dict())
    target_agent.policy_old.load_state_dict(source_agent.policy.state_dict())

    # Train target agent
    target_trainer = ActiveSLAMTrainer(target_agent, env, episodes=200)  # More episodes
    logger.info("Training target agent with transfer learning...")
    target_trainer.train()

    # Test
    logger.info("Testing...")
    test_rewards = []
    for i in range(5):
        state = env.reset()
        episode_reward = 0
        for _ in range(200):
            action, _, _ = target_agent.select_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            if done:
                break
        test_rewards.append(episode_reward)
        logger.info(f"Test {i+1}: {episode_reward:.2f}")
    logger.info(f"Average test reward: {np.mean(test_rewards):.2f}")
    # Save trained model
    torch.save({
        'policy_state_dict': target_agent.policy.state_dict(),
        'optimizer_state_dict': target_agent.optimizer.state_dict(),
        'test_reward': np.mean(test_rewards)
    }, 'best_active_slam_model.pth')
    logger.info("Model saved successfully!")

    # Visualize final episode
    logger.info("Running visualization episode...")
    state = env.reset()
    episode_reward = 0
    target_agent.policy.eval()

    for step in range(300):
        action, _, _ = target_agent.select_action(state)
        state, reward, done = env.step(action)
        episode_reward += reward

        if step % 50 == 0:
            logger.info(f"Step {step}, Reward: {episode_reward:.2f}, Explored cells: {len(env.visited_cells)}")

        if done:
            break

    logger.info(f"Final visualization - Total reward: {episode_reward:.2f}, Cells explored: {len(env.visited_cells)}")
if __name__ == "__main__":
    main()