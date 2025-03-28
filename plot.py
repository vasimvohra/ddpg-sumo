from Classes.Environment_Platoon_simple_env import SUMOPlatoonEnv

env = SUMOPlatoonEnv(gui=False)
obs = env.reset()
print("\n================================")
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    print(f"fffff sfsdfasdfdsf#{step + 1}")
    for i in range(env.num_agents):
        vehicle_id = env.vehicle_ids[i]
        position = obs[2 * i]
        speed = obs[2 * i + 1]
        print(f"  Vehicle {vehicle_id}: Position = {position:.2f}, Speed = {speed:.2f}")

    if done:
        break


env.close()
