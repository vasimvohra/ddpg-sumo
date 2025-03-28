import os
import sys
import traci


class SUMOConnector:
    """Bridge between SUMO and existing MADDPG implementation"""

    def __init__(self):
        # Ensure SUMO_HOME is set
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")

    def start_sumo(self, config_file="platoon.sumocfg"):
        """Start SUMO with GUI and connect via TraCI"""
        sumo_cmd = [
            "sumo-gui",
            "-c", config_file,
            "--step-length", "0.1",
            "--start",  # Auto-start simulation
            "--quit-on-end"  # Quit when simulation ends
        ]
        traci.start(sumo_cmd)

    def step(self, actions):
        """Execute one simulation step"""
        # Apply actions to vehicles
        for veh_id in traci.vehicle.getIDList():
            if veh_id.startswith('leader'):
                # Apply leader actions
                speed = traci.vehicle.getSpeed(veh_id)
                traci.vehicle.setSpeed(veh_id, speed)  # Maintain speed for now
            elif veh_id.startswith('follower'):
                # Apply follower actions
                leader = traci.vehicle.getLeader(veh_id)
                if leader:
                    # Adjust speed based on gap
                    gap = leader[1]
                    if gap < 25:  # Desired gap
                        traci.vehicle.setSpeed(veh_id, traci.vehicle.getSpeed(veh_id) * 0.9)
                    elif gap > 25:
                        traci.vehicle.setSpeed(veh_id, traci.vehicle.getSpeed(veh_id) * 1.1)

        # Execute simulation step
        traci.simulationStep()

        # Get state information
        state = self.get_state()
        return state

    def get_state(self):
        """Get current state of all vehicles"""
        state = {}
        for veh_id in traci.vehicle.getIDList():
            state[veh_id] = {
                'position': traci.vehicle.getPosition(veh_id),
                'speed': traci.vehicle.getSpeed(veh_id),
                'acceleration': traci.vehicle.getAcceleration(veh_id)
            }
            if veh_id.startswith('follower'):
                leader = traci.vehicle.getLeader(veh_id)
                if leader:
                    state[veh_id]['gap'] = leader[1]
        return state

    def close(self):
        """Close SUMO connection"""
        traci.close()