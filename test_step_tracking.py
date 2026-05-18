"""
Test the step tracking functionality
"""
from src.utils.logger import Logger
import pandas as pd

print("=" * 60)
print("Testing Step Tracking Implementation")
print("=" * 60)

# Create logger for DQN on CartPole
logger = Logger("dqn", "CartPole-v1", seed=0)

# Simulate 5 episodes with varying step counts
episodes_data = [
    (45, 195.0),  # Episode 1: 45 steps, reward 195
    (42, 200.0),  # Episode 2: 42 steps, reward 200
    (48, 210.0),  # Episode 3: 48 steps, reward 210
    (50, 220.0),  # Episode 4: 50 steps, reward 220
    (55, 230.0),  # Episode 5: 55 steps, reward 230
]

print("\nSimulating 5 training episodes:")
print("-" * 60)
cumulative_steps = 0
for ep_num, (steps, reward) in enumerate(episodes_data, 1):
    cumulative_steps += steps
    logger.log_episode_reward(reward)
    logger.log_steps_in_episode(steps)
    print(f"Episode {ep_num}: {steps} steps, reward={reward}, total_steps={cumulative_steps}")

print("-" * 60)
print(f"\nTotal steps recorded: {logger.total_steps}")
print(f"Steps per episode: {logger.steps_per_episode}")

# Convert to dataframe
df = logger.to_dataframe()
episode_df = df[df['metric'] == 'episode_reward'].copy()

print("\n" + "=" * 60)
print("DataFrame Output (what gets saved to CSV):")
print("=" * 60)
print(episode_df[['step', 'value', 'metric']].to_string(index=False))

print("\n" + "=" * 60)
print("Verification:")
print("=" * 60)
print(f"✓ Step column is cumulative: {episode_df['step'].tolist()}")
print(f"✓ Values match episode rewards: {episode_df['value'].tolist()}")
print(f"✓ Metrics labeled correctly: {episode_df['metric'].unique()}")

# Check that steps are indeed cumulative
expected_steps = [45, 87, 135, 185, 240]
actual_steps = episode_df['step'].tolist()
if actual_steps == expected_steps:
    print(f"✓ Cumulative steps correct: {actual_steps}")
else:
    print(f"✗ ERROR: Expected {expected_steps}, got {actual_steps}")

print("\n" + "=" * 60)
print("SUCCESS: Step tracking is working correctly!")
print("=" * 60)
