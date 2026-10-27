# This is a test case where we run 100 episodes of inventory management. The model will order a fixed [20, 20, 20] units regardless of demand. This is just to have as a measureable baseline for the RL model. 

import numpy as np
from or_gym.envs.supply_chain.inventory_management import (
    InvManagementBacklogEnv,
)

NUM_EPISODES = 100

env = InvManagementBacklogEnv()
episode_rewards = []
episode_backlogs = []

for episode in range(NUM_EPISODES):

    state = env.reset()
    total_reward = 0
    backlog_days = 0

    for day in range(env.num_periods):

        action = np.array([20, 20, 20], dtype=np.int16)

        state, reward, done, info = env.step(action)

        total_reward += reward

        if env.B[day, 0] > 0:
            backlog_days += 1

        if done:
            break

    episode_rewards.append(total_reward)
    episode_backlogs.append(backlog_days)

    print(
        f"Episode {episode + 1:3d}: "
        f"Reward = {total_reward:.2f}"
    )

print("\n===== BASELINE RESULTS =====")

print(f"Episodes: {NUM_EPISODES}")
print(f"Average reward: {np.mean(episode_rewards):.2f}")
print(f"Standard deviation: {np.std(episode_rewards):.2f}")
print(f"Minimum reward: {np.min(episode_rewards):.2f}")
print(f"Maximum reward: {np.max(episode_rewards):.2f}")
print(f"Average backlog days: {np.mean(episode_backlogs):.2f}")

env.close()
