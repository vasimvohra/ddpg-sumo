from sumo_bridge import SUMOBridge
from Environment_Platoon import Environ
from ddpg_torch import Agent
import numpy as np


def train_platoon_control():
    # Initialize environment with your original parameters
    down_lanes = [0, 1, 2, 3, 4]
    up_lanes = [0, 1, 2, 3, 4]
    left_lanes = [0, 1, 2, 3, 4]
    right_lanes = [0, 1, 2, 3, 4]
    width = 750
    height = 1299
    n_veh = 12
    size_platoon = 3
    n_RB = 3
    V2I_min = 540
    bandwidth = 180000
    V2V_size = 4000 * 8
    Gap = 25

    # Calculate number of platoons
    n_platoon = int(n_veh / size_platoon)

    # Create environment
    env = Environ(down_lanes, up_lanes, left_lanes, right_lanes,
                  width, height, n_veh, size_platoon, n_RB,
                  V2I_min, bandwidth, V2V_size, Gap)

    # Initialize game
    env.new_random_game()

    # Create SUMO bridge
    sumo = SUMOBridge(env)

    # Initialize MADDPG agent
    n_input = len(get_state(env=env, idx=0))
    n_output = 3  # channel selection, mode selection, power

    agent = Agent(
        alpha=0.0001,
        beta=0.001,
        input_dims=n_input,
        tau=0.005,
        n_actions=n_output,
        gamma=0.99,
        max_size=100000,
        C_fc1_dims=1024,
        C_fc2_dims=512,
        C_fc3_dims=256,
        A_fc1_dims=1024,
        A_fc2_dims=512,
        batch_size=64,
        n_agents=n_platoon
    )

    # Start SUMO visualization
    try:
        sumo.start_sumo()

        n_episode = 500
        n_step_per_episode = int(env.time_slow / env.time_fast)
        batch_size = 64

        # Training loop
        for i_episode in range(n_episode):
            print(f"\nEpisode {i_episode + 1}")

            # Reset environment state
            env.V2V_demand = env.V2V_demand_size * np.ones(n_platoon, dtype=np.float16)
            env.individual_time_limit = env.time_slow * np.ones(n_platoon, dtype=np.float16)
            env.active_links = np.ones((n_platoon), dtype='bool')

            if i_episode == 0:
                env.AoI = np.ones(n_platoon) * 100

            # Update environment every 20 episodes
            if i_episode % 20 == 0:
                env.renew_positions()
                env.renew_channel(n_veh, size_platoon)
                env.renew_channels_fastfading()

            # Get initial states
            state_old_all = []
            for i in range(n_platoon):
                state = get_state(env=env, idx=i)
                state_old_all.append(state)

            # Episode loop
            for i_step in range(n_step_per_episode):
                # Get actions from MADDPG
                action = agent.choose_action(np.asarray(state_old_all).flatten())
                action = np.clip(action, -0.999, 0.999)

                # Process actions for each platoon
                action_all_training = np.zeros([n_platoon, n_output], dtype=int)
                for i in range(n_platoon):
                    action_all_training[i, 0] = ((action[0 + i * n_output] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, 1] = ((action[1 + i * n_output] + 1) / 2) * 2  # Inter/Intra platoon mode
                    action_all_training[i, 2] = np.round(
                        np.clip(((action[2 + i * n_output] + 1) / 2) * 30, 1, 30))  # power

                # Step environment with SUMO visualization
                per_user_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand, V2V_success = \
                    sumo.step(action_all_training)

                # Get new states
                state_new_all = []
                for i in range(n_platoon):
                    state_new = get_state(env, i)
                    state_new_all.append(state_new)

                # Store transition
                agent.remember(
                    np.asarray(state_old_all).flatten(),
                    action,
                    global_reward,
                    np.asarray(state_new_all).flatten(),
                    i_step == n_step_per_episode - 1
                )

                # Learn
                if agent.memory.mem_cntr > batch_size:
                    agent.learn()

                # Update states
                state_old_all = state_new_all

                # Print progress
                if i_step % 10 == 0:
                    print(f"Step {i_step}, Reward: {global_reward:.2f}")
                    for p in range(n_platoon):
                        print(f"Platoon {p}: V2I: {C_rate[p]:.2f}, V2V: {V_rate[p]:.2f}, AoI: {platoon_AoI[p]:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        sumo.close()


def get_state(env, idx):
    """ Get state from the environment """
    V2I_abs = (env.V2I_channels_abs[idx * env.size_platoon] - 60) / 60.0
    V2V_abs = (env.V2V_channels_abs[idx * env.size_platoon,
                                    idx * env.size_platoon + (1 + np.arange(env.size_platoon - 1))] - 60) / 60.0
    V2I_fast = (env.V2I_channels_with_fastfading[idx * env.size_platoon, :] -
                env.V2I_channels_abs[idx * env.size_platoon] + 10) / 35
    V2V_fast = (env.V2V_channels_with_fastfading[idx * env.size_platoon,
                idx * env.size_platoon + (1 + np.arange(env.size_platoon - 1)), :] -
                env.V2V_channels_abs[idx * env.size_platoon,
                                     idx * env.size_platoon + (1 + np.arange(env.size_platoon - 1))].reshape(
                    env.size_platoon - 1, 1) + 10) / 35
    Interference = (-env.Interference_all[idx] - 60) / 60
    AoI_levels = env.AoI[idx] / (int(env.time_slow / env.time_fast))
    V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])

    return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_abs, -1),
                           np.reshape(V2V_fast, -1), np.reshape(Interference, -1), np.reshape(AoI_levels, -1),
                           V2V_load_remaining), axis=0)


if __name__ == "__main__":
    train_platoon_control()