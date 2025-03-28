import os
import sys
import traci
import numpy as np
from datetime import datetime
from Classes.Environment_Platoon import Environ, Vehicle


class SUMOBridge:
    def __init__(self, env: Environ):
        """Bridge between SUMO and existing Environment_Platoon

        Args:
            env (Environ): Instance of Environment_Platoon class
        """
        self.env = env
        self.log_file = "sumo_bridge.log"
        self._setup_logging()

        # Ensure SUMO_HOME is set
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
            self.log_message("SUMO_HOME found and tools path added")
        else:
            error_msg = "Please declare environment variable 'SUMO_HOME'"
            self.log_message(error_msg, level="ERROR")
            sys.exit(error_msg)

    def _setup_logging(self):
        """Setup logging to file"""
        with open(self.log_file, 'w') as f:
            f.write(f"SUMO Bridge Log - Started at {datetime.utcnow()}\n")

    def log_message(self, message, level="INFO"):
        """Log message to file with timestamp

        Args:
            message (str): Message to log
            level (str): Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} [{level}] {message}\n")
        if level == "ERROR":
            print(f"Error: {message}")

    def start_sumo(self, config_file="platoon.sumocfg"):
        """Start SUMO with GUI and initialize vehicles

        Args:
            config_file (str): Path to SUMO configuration file
        """
        sumo_cmd = [
            "sumo-gui",
            "-c", config_file,
            "--step-length", str(self.env.time_fast),
            "--start",
            "--quit-on-end",
            "--error-log", "sumo_errors.log",
            "--no-warnings", "true",
            "--gui-settings-file", "gui-settings.cfg"
        ]

        cmd_str = " ".join(sumo_cmd)
        self.log_message(f"Starting SUMO with command: {cmd_str}")
        print(f"Starting SUMO with command: {cmd_str}")

        try:
            traci.start(sumo_cmd)
            self.log_message("SUMO started successfully")

            # Set simulation parameters
            traci.simulation.setParam("time-to-teleport", "-1")
            traci.simulation.setParam("collision.action", "none")

            # Initialize vehicles
            self._init_vehicles()

        except traci.exceptions.FatalTraCIError as e:
            error_msg = f"Error starting SUMO: {e}"
            self.log_message(error_msg, level="ERROR")
            if os.path.exists("sumo_errors.log"):
                with open("sumo_errors.log", "r") as f:
                    error_log = f.read()
                    self.log_message(f"SUMO Error Log:\n{error_log}", level="ERROR")
                    print("SUMO Error Log:")
                    print(error_log)
            raise

    def _init_vehicles(self):
        """Initialize vehicles in SUMO based on Environment_Platoon state"""
        # First remove any existing vehicles
        for vid in traci.vehicle.getIDList():
            traci.vehicle.remove(vid)
            self.log_message(f"Removed existing vehicle: {vid}")

        # Add new vehicles
        for i, vehicle in enumerate(self.env.vehicles):
            veh_id = f"leader_{i // self.env.size_platoon}" if i % self.env.size_platoon == 0 else f"follower{i % self.env.size_platoon}_{i // self.env.size_platoon}"

            try:
                # Add vehicle to SUMO
                traci.vehicle.add(
                    vehID=veh_id,
                    routeID="route_0",
                    typeID="default_vehicle",
                    departPos="0",
                    departLane="0",
                    departSpeed=str(vehicle.velocity)
                )

                # Set vehicle color based on role (leader/follower)
                if i % self.env.size_platoon == 0:
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))  # Red for leaders
                else:
                    traci.vehicle.setColor(veh_id, (0, 255, 0, 255))  # Green for followers

                # Set initial position
                traci.vehicle.moveToXY(
                    vehID=veh_id,
                    edgeID="E0",  # Changed from gneE0
                    lane=0,
                    x=vehicle.position[0],
                    y=vehicle.position[1],
                    angle=self._get_angle(vehicle.direction),
                    keepRoute=2
                )

                # Set vehicle parameters
                traci.vehicle.setMinGap(veh_id, self.env.gap)
                traci.vehicle.setSpeedMode(veh_id, 31)  # Disable all speed checks
                traci.vehicle.setLaneChangeMode(veh_id, 0)  # Disable lane changes

                self.log_message(f"Vehicle {veh_id} initialized at position {vehicle.position}")

            except traci.exceptions.TraCIException as e:
                self.log_message(f"Error initializing vehicle {veh_id}: {e}", level="ERROR")

    def _get_angle(self, direction):
        """Convert direction to angle in degrees for SUMO

        Args:
            direction (str): Direction ('u', 'd', 'l', 'r')

        Returns:
            float: Angle in degrees
        """
        angles = {
            'u': 90,  # up = 90 degrees
            'd': 270,  # down = 270 degrees
            'l': 180,  # left = 180 degrees
            'r': 0  # right = 0 degrees
        }
        return angles.get(direction, 0)

    def step(self, actions):
        """Execute one simulation step

        Args:
            actions: Actions from MADDPG agent

        Returns:
            tuple: (per_user_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand, V2V_success)
        """
        try:
            # First apply actions in Environment_Platoon
            per_user_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand, V2V_success = \
                self.env.act_for_training(actions)

            # Then update SUMO with new positions
            self._update_sumo_positions()

            # Execute SUMO step
            traci.simulationStep()

            # Log step results
            self.log_message(f"Step completed - Global reward: {global_reward}")

            # Return results from Environment_Platoon
            return per_user_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand, V2V_success

        except traci.exceptions.TraCIException as e:
            self.log_message(f"Error in SUMO step: {e}", level="ERROR")
            raise

    def _update_sumo_positions(self):
        """Update SUMO vehicle positions based on Environment_Platoon"""
        for i, vehicle in enumerate(self.env.vehicles):
            veh_id = f"leader_{i // self.env.size_platoon}" if i % self.env.size_platoon == 0 else f"follower{i % self.env.size_platoon}_{i // self.env.size_platoon}"

            try:
                # Determine edge ID based on position
                edge_id = "E0" if vehicle.position[1] < 1299 else "E1"  # Changed from gneE0/gneE1

                # Update position
                traci.vehicle.moveToXY(
                    vehID=veh_id,
                    edgeID=edge_id,
                    lane=0,
                    x=vehicle.position[0],
                    y=vehicle.position[1],
                    angle=self._get_angle(vehicle.direction),
                    keepRoute=2
                )

                # Update speed
                traci.vehicle.setSpeed(veh_id, vehicle.velocity)

            except traci.exceptions.TraCIException as e:
                self.log_message(f"Error updating vehicle {veh_id}: {e}", level="ERROR")

    def get_vehicle_data(self):
        """Get current vehicle data from SUMO

        Returns:
            dict: Dictionary of vehicle data
        """
        data = {}
        try:
            for veh_id in traci.vehicle.getIDList():
                data[veh_id] = {
                    'position': traci.vehicle.getPosition(veh_id),
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'angle': traci.vehicle.getAngle(veh_id),
                    'lane': traci.vehicle.getLaneID(veh_id)
                }
        except traci.exceptions.TraCIException as e:
            self.log_message(f"Error getting vehicle data: {e}", level="ERROR")
        return data

    def close(self):
        """Close SUMO connection and cleanup"""
        try:
            traci.close()
            self.log_message("SUMO connection closed")
        except:
            self.log_message("Error closing SUMO connection", level="WARNING")