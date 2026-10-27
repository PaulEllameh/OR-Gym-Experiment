# This is a test case where we run 100 episodes of inventory management. The model will order a fixed [20, 20, 20] units regardless of demand. This is just to have as a measureable baseline for the RL model. 

import numpy as np
from or_gym.envs.supply_chain.inventory_management import (
    InvManagementBacklogEnv,
)
from pathlib import Path
import csv
import matplotlib.pyplot as plt

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


results_directory = Path("results")
results_directory.mkdir(exist_ok=True)

csv_path = results_directory / "hund_episode_results.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Total Reward", "Backlog Days"])

    for i, (reward, backlog) in enumerate(
        zip(episode_rewards, episode_backlogs), start=1
    ):
        writer.writerow([i, reward, backlog])

plt.figure(figsize=(10, 6))

plt.hist(
    episode_rewards,
    bins=10,
    edgecolor="black"
)

mean_reward = np.mean(episode_rewards)

plt.axvline(
    mean_reward,
    linestyle="--",
    linewidth=2,
    label=f"Average: {mean_reward:.2f}"
)

plt.title("Distribution of Total Rewards for Fixed Ordering Policy (100 Episodes)")
plt.xlabel("Total Episode Reward")
plt.ylabel("Number of Episodes")
plt.yticks(range(0, 21, 2))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(results_directory / "hund_episode_rewards.png", dpi=300)
plt.close()

print("\nResults saved successfully!")
