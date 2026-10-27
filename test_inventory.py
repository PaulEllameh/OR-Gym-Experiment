
import numpy as np
from or_gym.envs.supply_chain.inventory_management import (
    InvManagementBacklogEnv,
)

def main():
    env = InvManagementBacklogEnv()

    state = env.reset()

    total_reward = 0

    print("OR-Gym Inventory Management Simulation")
    print("---------------------------------------")

    print(f"Episode length: {env.num_periods} days")
    print(f"Initial inventory: {env.I[0]}")
    print(f"Lead times: {env.lead_time}")
    print(f"Observation shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print()

    for day in range(env.num_periods):

        action = np.array([20, 20, 20], dtype=np.int16)

        state, reward, done, info = env.step(action)

        total_reward += reward

        demand = env.D[day]
        inventory = env.I[env.period]
        retailer_sales = env.S[day, 0]
        retailer_backlog = env.B[day, 0]

        print(f"Day {day + 1}")
        print(f"  Demand: {demand:.0f}")
        print(f"  Orders: {action}")
        print(f"  Inventory: {inventory}")
        print(f"  Retailer sales: {retailer_sales:.0f}")
        print(f"  Retailer backlog: {retailer_backlog:.0f}")
        print(f"  Reward: {reward:.2f}")
        print()

        if done:
            break

    print("---------------------------------------")
    print("Simulation completed successfully!")
    print(f"Total episode reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
