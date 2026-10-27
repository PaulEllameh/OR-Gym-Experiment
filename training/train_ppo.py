
from pathlib import Path

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from or_gym.envs.supply_chain.inventory_management import (
    InvManagementBacklogEnv,
)

class ScaleActions(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.capacities = env.action_space.high.astype(
            np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

    def action(self, action):
        action = np.clip(action, -1.0, 1.0)

        orders = (
            (action + 1.0) / 2.0
        ) * self.capacities

        return np.rint(orders).astype(np.int16)


class FloatObservations(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.astype(
                np.float32
            ) / 100.0,
            high=env.observation_space.high.astype(
                np.float32
            ) / 100.0,
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(
            observation, dtype=np.float32
        ) / 100.0


def main():

    print("Starting OR-Gym PPO Training")

    env = InvManagementBacklogEnv()

    env = ScaleActions(env)
    env = FloatObservations(env)

    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        policy_kwargs={
            "net_arch": dict(
                pi=[128, 128],
                vf=[128, 128]
            )
        },
        verbose=1,
        seed=42,
        device="cpu",
    )

    print("\nTraining PPO...")
    
    model.learn(total_timesteps=5120)

    Path("models").mkdir(exist_ok=True)

    model.save("models/ppo_inventory_first_try")

    print("\nTraining completed!")
    print("Model saved successfully.")

    observation = env.reset()
    episode_reward = 0.0

    for day in range(30):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, done, info = env.step(
            action
        )

        episode_reward += reward

        if done:
            break

    print(
        f"Test episode reward: {episode_reward:.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()
