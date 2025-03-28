import os
import sys
import traci
import numpy as np
from typing import List, Dict, Tuple
import os
os.environ["SUMO_HOME"] = "C:/Program Files (x86)/Eclipse/Sumo"

import os
import sys
import traci
import numpy as np
from typing import List, Dict, Tuple


class SUMOPlatoonEnv:
    class SUMOPlatoonEnv:
        def __init__(self, config_file: str, n_platoon: int, size_platoon: int):
            """SUMO Environment wrapper for platoon control

            Args:
                config_file (str): Path to SUMO config file
                n_platoon (int): Number of platoons
                size_platoon (int): Size of each platoon
            """
            self.config_file = config_file
            self.n_platoon = n_platoon
            self.size_platoon = size_platoon

            # Store vehicle IDs
            self.vehicles = []
            for p in range(n_platoon):
                platoon = []
                platoon.append(f"leader_{p}")  # Leader ID
                for i in range(1, size_platoon):
                    platoon.append(f"follower{i}_{p}")  # Follower IDs
                self.vehicles.append(platoon)

            # Flatten vehicle list for convenience
            self.all_vehicle_ids = [vid for platoon in self.vehicles for vid in platoon]

            # SUMO setup
            if 'SUMO_HOME' in os.environ:
                tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
                sys.path.append(tools)
            else:
                sys.exit("Please declare environment variable 'SUMO_HOME'")

            # Communication parameters
            self.n_RB = 3  # number of resource blocks
            self.V2I_min = 540  # minimum required data rate for V2I
            self.bandwidth = 180000  # bandwidth per RB
            self.V2V_size = 32000  # V2V payload size (4000 bytes * 8)
            self.gap = 25  # desired gap between vehicles in meters

            # State/Action space setup
            self.n_actions = 3  # [channel selection, mode selection, power]
            self.state_dim = self._get_state_dim()

        def _get_state_dim(self) -> int:
            """Calculate state dimension based on features"""
            # Features per platoon:
            # - V2I channel info (abs + fast fading)
            # - V2V channel info (abs + fast fading) for each follower
            # - Interference level
            # - AoI (Age of Information)
            # - V2V load remaining
            return 2 + 2 * (self.size_platoon - 1) + 1 + 1 + 1

    def start(self):
        """Start SUMO simulation"""
        sumo_cmd = [
            "sumo-gui",
            "-c", self.config_file,
            "--step-length", "0.1",
            "--start"
        ]
        traci.start(sumo_cmd)

    def get_state(self) -> np.ndarray:
        """Get current state of the platoon"""
        # Get vehicle data
        leader_data = self._get_vehicle_data(self.leader_id)
        follower_data = [self._get_vehicle_data(f_id) for f_id in self.follower_ids]

        # Calculate V2V and V2I channel characteristics
        v2i_abs = (leader_data['distance_to_bs'] - 60) / 60.0
        v2i_fast = (leader_data['speed'] - 10) / 35.0

        v2v_abs = []
        v2v_fast = []
        for f_data in follower_data:
            dist = f_data['distance_to_leader']
            v2v_abs.append((dist - 60) / 60.0)
            v2v_fast.append((f_data['speed'] - leader_data['speed']) / 35.0)

        # Calculate interference and AoI
        interference = self._calculate_interference(self.leader_id)
        aoi = self._get_age_of_information()

        # Calculate remaining V2V demand
        v2v_remaining = self._get_v2v_demand()

        # Combine all features
        state = np.concatenate([
            [v2i_abs, v2i_fast],
            v2v_abs,
            v2v_fast,
            [interference],
            [aoi],
            [v2v_remaining]
        ])

        return state

    def _get_vehicle_data(self, vehicle_id: str) -> Dict:
        """Get vehicle telemetry data"""
        try:
            pos = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)

            # Get distance to leader for followers
            if vehicle_id in self.follower_ids:
                leader_info = traci.vehicle.getLeader(vehicle_id)
                distance_to_leader = leader_info[1] if leader_info else self.gap
            else:
                distance_to_leader = 0

            return {
                'speed': speed,
                'position': pos,
                'distance_to_leader': distance_to_leader,
                'distance_to_bs': self._calculate_distance_to_bs(pos),
            }
        except traci.exceptions.TraCIException as e:
            print(f"Error getting data for vehicle {vehicle_id}: {e}")
            return {
                'speed': 0,
                'position': (0, 0),
                'distance_to_leader': self.gap,
                'distance_to_bs': 100
            }

    def _calculate_distance_to_bs(self, position: Tuple[float, float]) -> float:
        """Calculate distance to base station"""
        bs_position = (750 / 2, 1299 / 2)  # Base station position
        return np.hypot(position[0] - bs_position[0], position[1] - bs_position[1])

    def _calculate_interference(self, vehicle_id: str) -> float:
        """Calculate interference for a vehicle"""
        interference = -60  # Base interference
        try:
            pos1 = traci.vehicle.getPosition(vehicle_id)
            for v_id in self.all_vehicle_ids:
                if v_id != vehicle_id:
                    pos2 = traci.vehicle.getPosition(v_id)
                    dist = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                    if dist < 100:  # Only consider nearby vehicles
                        interference += (100 - dist) * 0.1
        except traci.exceptions.TraCIException as e:
            print(f"Error calculating interference: {e}")
        return (interference - 60) / 60.0

    def _get_age_of_information(self) -> float:
        """Get current AoI"""
        # Simplified AoI calculation
        return 1.0

    def _get_v2v_demand(self) -> float:
        """Get remaining V2V demand"""
        # Simplified V2V demand
        return 1.0

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one simulation step"""
        # Extract actions
        rb_selection = int(((actions[0] + 1) / 2) * self.n_RB)
        mode = int(((actions[1] + 1) / 2) * 2)  # 0: V2I, 1: V2V
        power = np.clip(((actions[2] + 1) / 2) * 30, 1, 30)  # 1-30 dBm

        # Apply actions
        self._apply_actions(rb_selection, mode, power)

        # Advance SUMO simulation
        try:
            traci.simulationStep()
        except traci.exceptions.TraCIException as e:
            print(f"Error in simulation step: {e}")
            return self.get_state(), -1, True, {}

        # Calculate reward components
        v2i_rate = self._calculate_v2i_rate()
        v2v_rate = self._calculate_v2v_rate()
        aoi = self._get_age_of_information()

        # Calculate reward
        reward = self._calculate_reward(v2i_rate, v2v_rate, aoi)

        # Get new state
        new_state = self.get_state()

        # Check if simulation is done
        done = self._is_done()

        info = {
            'V2I_rate': v2i_rate,
            'V2V_rate': v2v_rate,
            'AoI': aoi
        }

        return new_state, reward, done, info

    def _apply_actions(self, rb: int, mode: int, power: float):
        """Apply actions to vehicles"""
        try:
            if mode == 0:  # V2I mode
                # Set speed to maintain platoon
                self._maintain_platoon()
                # Set communication power for V2I
                traci.vehicle.setParameter(self.leader_id, "device.btreceiver.power", str(power))
            else:  # V2V mode
                # Adjust speeds for better V2V communication
                self._adjust_platoon_formation()
                # Set communication power for V2V
                for vid in self.all_vehicle_ids:
                    traci.vehicle.setParameter(vid, "device.btreceiver.power", str(power))
        except traci.exceptions.TraCIException as e:
            print(f"Error applying actions: {e}")

    def _maintain_platoon(self):
        """Maintain platoon formation"""
        try:
            leader_speed = traci.vehicle.getSpeed(self.leader_id)

            for follower_id in self.follower_ids:
                # Get distance to leader
                leader_data = traci.vehicle.getLeader(follower_id)
                if leader_data:
                    dist = leader_data[1]
                    # Adjust speed based on distance
                    if dist < self.gap:
                        traci.vehicle.setSpeed(follower_id, leader_speed * 0.9)
                    elif dist > self.gap:
                        traci.vehicle.setSpeed(follower_id, leader_speed * 1.1)
                    else:
                        traci.vehicle.setSpeed(follower_id, leader_speed)
        except traci.exceptions.TraCIException as e:
            print(f"Error maintaining platoon: {e}")

    def _adjust_platoon_formation(self):
        """Adjust platoon formation for V2V communication"""
        self._maintain_platoon()  # Use same logic as maintain_platoon for now

    def _calculate_v2i_rate(self) -> float:
        """Calculate V2I communication rate"""
        return 1.0  # Simplified calculation

    def _calculate_v2v_rate(self) -> float:
        """Calculate V2V communication rate"""
        return 1.0  # Simplified calculation

    def _calculate_reward(self, v2i_rate: float, v2v_rate: float, aoi: float) -> float:
        """Calculate reward based on communication rates and AoI"""
        return v2i_rate * 0.4 + v2v_rate * 0.4 - aoi * 0.2

    def _is_done(self) -> bool:
        """Check if simulation is done"""
        return traci.simulation.getMinExpectedNumber() <= 0

    def close(self):
        """Close SUMO simulation"""
        traci.close()