import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TransferLearningManager:
    """
    Transfer Learning component for RPPO Active SLAM
    Implements the transfer learning framework from Paper 3
    """

    def __init__(self, source_model_path=None, target_model_path=None):
        self.source_model_path = source_model_path
        self.target_model_path = target_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_source_domain_features(self, source_agent, simple_env_episodes=500):
        """
        Train agent in simple environment (source domain)
        Extract learned features for transfer
        """
        logger.info("=== SOURCE DOMAIN TRAINING ===")
        logger.info(f"Training in simple environment for {simple_env_episodes} episodes...")

        # Train in simple environment (fewer obstacles, static only)
        source_rewards = []

        for episode in range(simple_env_episodes):
            episode_reward = 0
            state = simple_env.reset()  # Assumed simple environment

            for step in range(300):
                action, log_prob, value = source_agent.select_action(state)
                next_state, reward, done = simple_env.step(action)

                source_agent.store(state, action, reward, log_prob, value, done)
                episode_reward += reward
                state = next_state

                if done:
                    break

            # Update policy
            if len(source_agent.states) >= 64:
                policy_loss, value_loss = source_agent.update()

            source_rewards.append(episode_reward)

            if episode % 50 == 0:
                avg_reward = np.mean(source_rewards[-50:])
                logger.info(f"Source Episode {episode}, Avg Reward: {avg_reward:.2f}")

        # Save source domain model
        if self.source_model_path:
            torch.save({
                'policy_state_dict': source_agent.policy.state_dict(),
                'policy_old_state_dict': source_agent.policy_old.state_dict(),
                'optimizer_state_dict': source_agent.optimizer.state_dict(),
                'episode_rewards': source_rewards,
                'total_episodes': simple_env_episodes
            }, self.source_model_path)
            logger.info(f"Source domain model saved to {self.source_model_path}")

        return source_agent

    def transfer_knowledge(self, source_agent, target_agent, transfer_strategy='full_network'):
        """
        Transfer learned knowledge from source to target domain

        Args:
            source_agent: Trained agent from source domain
            target_agent: New agent for target domain
            transfer_strategy: 'full_network', 'feature_layers', 'conv_only'
        """
        logger.info("=== KNOWLEDGE TRANSFER ===")
        logger.info(f"Transfer strategy: {transfer_strategy}")

        source_state_dict = source_agent.policy.state_dict()
        target_state_dict = target_agent.policy.state_dict()

        if transfer_strategy == 'full_network':
            # Transfer all network parameters
            target_agent.policy.load_state_dict(source_state_dict)
            target_agent.policy_old.load_state_dict(source_state_dict)
            logger.info("Transferred full network parameters")

        elif transfer_strategy == 'feature_layers':
            # Transfer only convolutional and shared layers
            transfer_layers = [
                'conv1.weight', 'conv1.bias',
                'conv2.weight', 'conv2.bias',
                'sep_conv.depthwise.weight', 'sep_conv.depthwise.bias',
                'sep_conv.pointwise.weight', 'sep_conv.pointwise.bias',
                'shared_fc.0.weight', 'shared_fc.0.bias',
                'shared_fc.2.weight', 'shared_fc.2.bias'
            ]

            for layer_name in transfer_layers:
                if layer_name in source_state_dict and layer_name in target_state_dict:
                    target_state_dict[layer_name] = source_state_dict[layer_name]
                    logger.info(f"Transferred layer: {layer_name}")

            target_agent.policy.load_state_dict(target_state_dict)
            target_agent.policy_old.load_state_dict(target_state_dict)

        elif transfer_strategy == 'conv_only':
            # Transfer only convolutional feature extractors
            conv_layers = [
                'conv1.weight', 'conv1.bias',
                'conv2.weight', 'conv2.bias',
                'sep_conv.depthwise.weight', 'sep_conv.depthwise.bias',
                'sep_conv.pointwise.weight', 'sep_conv.pointwise.bias'
            ]

            for layer_name in conv_layers:
                if layer_name in source_state_dict and layer_name in target_state_dict:
                    target_state_dict[layer_name] = source_state_dict[layer_name]
                    logger.info(f"Transferred conv layer: {layer_name}")

            target_agent.policy.load_state_dict(target_state_dict)
            target_agent.policy_old.load_state_dict(target_state_dict)

        # Reduce learning rate for fine-tuning
        for param_group in target_agent.optimizer.param_groups:
            param_group['lr'] *= 0.1  # 10x smaller learning rate

        logger.info("Knowledge transfer completed!")
        return target_agent

    def fine_tune_target_domain(self, target_agent, complex_env, target_episodes=300):
        """
        Fine-tune transferred model in target domain (complex environment)

        Args:
            target_agent: Agent with transferred knowledge
            complex_env: Complex environment with dynamic obstacles
            target_episodes: Number of episodes for fine-tuning
        """
        logger.info("=== TARGET DOMAIN FINE-TUNING ===")
        logger.info(f"Fine-tuning in complex environment for {target_episodes} episodes...")

        target_rewards = []

        for episode in range(target_episodes):
            episode_reward = 0
            state = complex_env.reset()

            for step in range(400):  # Longer episodes for complex environment
                action, log_prob, value = target_agent.select_action(state)
                next_state, reward, done = complex_env.step(action)

                target_agent.store(state, action, reward, log_prob, value, done)
                episode_reward += reward
                state = next_state

                if done:
                    break

            # Update policy with smaller batches for stability
            if len(target_agent.states) >= 32:
                policy_loss, value_loss = target_agent.update()

            target_rewards.append(episode_reward)

            if episode % 25 == 0:
                avg_reward = np.mean(target_rewards[-25:])
                logger.info(f"Target Episode {episode}, Avg Reward: {avg_reward:.2f}")

        # Save fine-tuned model
        if self.target_model_path:
            torch.save({
                'policy_state_dict': target_agent.policy.state_dict(),
                'policy_old_state_dict': target_agent.policy_old.state_dict(),
                'optimizer_state_dict': target_agent.optimizer.state_dict(),
                'episode_rewards': target_rewards,
                'total_episodes': target_episodes,
                'transfer_completed': True
            }, self.target_model_path)
            logger.info(f"Target domain model saved to {self.target_model_path}")

        return target_agent

    def load_pretrained_model(self, agent, model_path):
        """Load pretrained model for transfer learning or inference"""
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)

            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            agent.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Loaded pretrained model from {model_path}")

            if 'total_episodes' in checkpoint:
                logger.info(f"Model trained for {checkpoint['total_episodes']} episodes")

            return agent
        else:
            logger.warning(f"Model file {model_path} not found!")
            return agent

    def evaluate_transfer_effectiveness(self, source_agent, target_agent, test_env, num_tests=10):
        """
        Evaluate the effectiveness of transfer learning
        Compare performance with and without transfer
        """
        logger.info("=== TRANSFER LEARNING EVALUATION ===")

        # Test source agent
        source_agent.policy.eval()
        source_rewards = []

        for test in range(num_tests):
            state = test_env.reset()
            episode_reward = 0

            for _ in range(300):
                action, _, _ = source_agent.select_action(state)
                state, reward, done = test_env.step(action)
                episode_reward += reward
                if done:
                    break

            source_rewards.append(episode_reward)

        # Test target agent
        target_agent.policy.eval()
        target_rewards = []

        for test in range(num_tests):
            state = test_env.reset()
            episode_reward = 0

            for _ in range(300):
                action, _, _ = target_agent.select_action(state)
                state, reward, done = test_env.step(action)
                episode_reward += reward
                if done:
                    break

            target_rewards.append(episode_reward)

        # Calculate improvement metrics
        source_avg = np.mean(source_rewards)
        target_avg = np.mean(target_rewards)
        improvement = ((target_avg - source_avg) / abs(source_avg)) * 100

        logger.info(f"Source domain performance: {source_avg:.2f} ± {np.std(source_rewards):.2f}")
        logger.info(f"Target domain performance: {target_avg:.2f} ± {np.std(target_rewards):.2f}")
        logger.info(f"Transfer learning improvement: {improvement:.1f}%")

        return {
            'source_rewards': source_rewards,
            'target_rewards': target_rewards,
            'improvement_percentage': improvement,
            'source_avg': source_avg,
            'target_avg': target_avg
        }


class AdaptiveTransferStrategy:
    """
    Adaptive transfer learning strategy that selects optimal transfer approach
    based on environment similarity and agent performance
    """

    def __init__(self):
        self.environment_complexity_metrics = {}

    def analyze_environment_complexity(self, environment):
        """Analyze environment complexity to determine transfer strategy"""

        # Count obstacles
        static_obstacles = np.sum(environment == 1)
        total_cells = environment.size
        obstacle_density = static_obstacles / total_cells

        # Calculate environment connectivity (free space connectivity)
        free_cells = np.sum(environment == 0)
        connectivity_ratio = free_cells / total_cells

        # Estimate navigation difficulty
        difficulty_score = obstacle_density * 2 + (1 - connectivity_ratio)

        complexity = {
            'obstacle_density': obstacle_density,
            'connectivity_ratio': connectivity_ratio,
            'difficulty_score': difficulty_score,
            'complexity_level': 'simple' if difficulty_score < 0.3 else 'moderate' if difficulty_score < 0.6 else 'complex'
        }

        return complexity

    def select_transfer_strategy(self, source_complexity, target_complexity):
        """Select optimal transfer strategy based on environment analysis"""

        complexity_diff = target_complexity['difficulty_score'] - source_complexity['difficulty_score']

        if complexity_diff < 0.1:
            # Very similar environments - transfer everything
            return 'full_network'
        elif complexity_diff < 0.3:
            # Moderately different - transfer features only
            return 'feature_layers'
        else:
            # Very different - transfer low-level features only
            return 'conv_only'

    def progressive_transfer_learning(self, source_agent, target_agent, target_env):
        """
        Progressive transfer learning with multiple stages
        """
        logger.info("=== PROGRESSIVE TRANSFER LEARNING ===")

        # Stage 1: Transfer convolutional features
        self.transfer_conv_features(source_agent, target_agent)

        # Stage 2: Light fine-tuning with frozen conv layers
        self.fine_tune_with_frozen_features(target_agent, target_env, episodes=100)

        # Stage 3: Full fine-tuning
        self.full_fine_tuning(target_agent, target_env, episodes=200)

        return target_agent

    def transfer_conv_features(self, source_agent, target_agent):
        """Transfer only convolutional features"""
        source_dict = source_agent.policy.state_dict()
        target_dict = target_agent.policy.state_dict()

        conv_keys = [k for k in source_dict.keys() if 'conv' in k]

        for key in conv_keys:
            target_dict[key] = source_dict[key]

        target_agent.policy.load_state_dict(target_dict)
        target_agent.policy_old.load_state_dict(target_dict)

        logger.info("Transferred convolutional features")

    def fine_tune_with_frozen_features(self, agent, env, episodes=100):
        """Fine-tune with frozen feature layers"""
        # Freeze convolutional layers
        for name, param in agent.policy.named_parameters():
            if 'conv' in name:
                param.requires_grad = False

        logger.info("Fine-tuning with frozen conv layers...")

        # Fine-tune for specified episodes
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(200):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done = env.step(action)

                agent.store(state, action, reward, log_prob, value, done)
                episode_reward += reward
                state = next_state

                if done:
                    break

            if len(agent.states) >= 32:
                agent.update()

        logger.info("Frozen feature fine-tuning completed")

    def full_fine_tuning(self, agent, env, episodes=200):
        """Full fine-tuning with all layers unfrozen"""
        # Unfreeze all layers
        for param in agent.policy.parameters():
            param.requires_grad = True

        # Reduce learning rate for stability
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] *= 0.5

        logger.info("Full fine-tuning...")

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(300):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done = env.step(action)

                agent.store(state, action, reward, log_prob, value, done)
                episode_reward += reward
                state = next_state

                if done:
                    break

            if len(agent.states) >= 32:
                agent.update()

        logger.info("Full fine-tuning completed")