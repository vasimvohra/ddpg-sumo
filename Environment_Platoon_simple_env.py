import os
import sys
import numpy as np
import traci
import sumolib
from gym import Env
from gym.spaces import Box
import os
os.environ["SUMO_HOME"] = "C:/Program Files (x86)/Eclipse/Sumo"



class SUMOPlatoonEnv(Env):
    def __init__(self,
                 net_file="highway.net.xml",
                 route_file="highway.rou.xml",
                 num_agents=3,
                 sumo_binary="sumo",
                 gui=False):

        self.net_file = net_file
        self.route_file = route_file
        self.num_agents = num_agents
        self.gui = gui
        self.sumo_binary = sumo_binary

        self.max_steps = 500
        self.step_count = 0

        self.vehicle_ids = [f"veh{i}" for i in range(self.num_agents)]

        # Observation = [x1, v1, x2, v2, ...] (positions & speeds)
        self.observation_space = Box(low=0, high=1e3, shape=(2 * self.num_agents,), dtype=np.float32)

        # Action = acceleration for each agent
        self.action_space = Box(low=-5, high=5, shape=(self.num_agents,), dtype=np.float32)

    def start_sumo(self):
        if 'SUMO_HOME' not in os.environ:
            raise EnvironmentError("Please set the SUMO_HOME environment variable")

        sumo_cmd = [
            os.path.join(os.environ['SUMO_HOME'], "bin", "sumo-gui" if self.gui else "sumo"),
            "-n", self.net_file,
            "-r", self.route_file,
            "--start",
            "--step-length", "0.1",
            #"--delay", "100",
            "--duration-log.disable", "true"
        ]

        traci.start(sumo_cmd)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self.start_sumo()

        self.step_count = 0

        # Wait until vehicles are inserted into the simulation
        while True:
            traci.simulationStep()
            current_vehicles = traci.vehicle.getIDList()
            if all(vid in current_vehicles for vid in self.vehicle_ids):
                break

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        obs = []
        for vid in self.vehicle_ids:
            pos = traci.vehicle.getPosition(vid)[0]  # x position
            speed = traci.vehicle.getSpeed(vid)
            obs.extend([pos, speed])
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        self.step_count += 1

        for i, vid in enumerate(self.vehicle_ids):
            speed = traci.vehicle.getSpeed(vid)
            accel = np.clip(actions[i], -5, 5)
            new_speed = max(0, speed + accel * 0.1)
            traci.vehicle.setSpeed(vid, new_speed)

        traci.simulationStep()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = self.step_count >= self.max_steps

        # 📍 Print vehicle debug info here
        #print(f"Step {self.step_count}")
        for vid in self.vehicle_ids:
            pos = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
           # print(f"Vehicle {vid} → Pos: {pos}, Speed: {speed:.2f} m/s")
        #print("inside vehicle")
        return obs, reward, done, {}

    def _get_reward(self, obs):
        # Example: negative of speed variance + penalty for spacing violations
        speeds = obs[1::2]
        reward = -np.var(speeds)

        positions = obs[0::2]
        spacing = np.diff(positions)
        spacing_penalty = np.sum(spacing < 5.0) * -10.0

        return reward + spacing_penalty

    def close(self):
        if traci.isLoaded():
            traci.close()
