import numpy as np
from stable_baselines3 import PPO

from or_gym.envs.supply_chain.inventory_management import (
    InvManagementBacklogEnv
)

from train_ppo import ScaleActions, FloatObservations


NUM_EPISODES = 100
SEED = 2026


def evaluate(model=None):

    raw_env = InvManagementBacklogEnv()

    if model is not None:
        env = FloatObservations(ScaleActions(raw_env))
    else:
        env = raw_env

    np.random.seed(SEED)

    rewards = []
    demands = []

    for episode in range(NUM_EPISODES):

        state = env.reset()
        total_reward = 0.0

        for day in range(raw_env.num_periods):

            if model is None:
                action = np.array(
                    [20, 20, 20],
                    dtype=np.int16
                )
            else:
                action, _ = model.predict(
                    state,
                    deterministic=True
                )

            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        demands.append(raw_env.D.copy())

    env.close()

    return np.array(rewards), np.array(demands)


def main():

    print("Evaluating OR-Gym policies...\n")

    model = PPO.load(
        "models/ppo_inventory_second_try",
        device="cpu"
    )

    fixed_rewards, fixed_demands = evaluate()

    ppo_rewards, ppo_demands = evaluate(model)

    np.testing.assert_array_equal(
        fixed_demands,
        ppo_demands
    )

    print("===== 100 EPISODE COMPARISON =====")

    print("\nFixed Ordering Policy:")
    print(f"Mean reward: {fixed_rewards.mean():.2f}")
    print(f"Std deviation: {fixed_rewards.std():.2f}")

    print("\nPPO Policy:")
    print(f"Mean reward: {ppo_rewards.mean():.2f}")
    print(f"Std deviation: {ppo_rewards.std():.2f}")

    difference = ppo_rewards.mean() - fixed_rewards.mean()

    print(f"\nPPO vs Fixed: {difference:+.2f}")


if __name__ == "__main__":
    main()
