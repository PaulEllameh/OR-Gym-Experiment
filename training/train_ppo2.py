
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from or_gym.envs.supply_chain.inventory_management import (
    InvManagementBacklogEnv
)

from train_ppo import ScaleActions, FloatObservations


def main():

    print("Continuing OR-Gym PPO training...")

    env = InvManagementBacklogEnv()
    env = ScaleActions(env)
    env = FloatObservations(env)
    env = Monitor(env)

    model = PPO.load(
        "models/ppo_inventory_first_try",
        env=env,
        device="cpu"
    )

    print("Previous model loaded successfully.")

    # Train for another 50,000 timesteps.
    model.learn(
        total_timesteps=50000,
        reset_num_timesteps=False
    )

    Path("models").mkdir(exist_ok=True)

    model.save("models/ppo_inventory_second_try")

    print("\nAdditional training completed!")
    print("Extended PPO model saved.")

    env.close()


if __name__ == "__main__":
    main()
