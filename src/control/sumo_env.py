import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import os
import sys

# Import our existing tools
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from acquisition.detector import TrafficDetector
from acquisition.fusion import DataFusion

class SumoTrafficEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    It wraps the SUMO simulation so the RL agent can interact with it.
    """
    def __init__(self, sumo_binary, config_path):
        super(SumoTrafficEnv, self).__init__()
        
        self.sumo_binary = sumo_binary
        self.config_path = config_path
        self.sim_steps = 0
        
        # --- ACTIONS ---
        # 0: Switch to (or keep) North-South Green
        # 1: Switch to (or keep) East-West Green
        self.action_space = spaces.Discrete(2)
        
        # --- OBSERVATION (STATE) ---
        # Vector of size 5:
        # [Queue_N, Queue_S, Queue_E, Queue_W, Current_Phase_Index]
        self.observation_space = spaces.Box(low=0, high=999, shape=(5,), dtype=np.float32)
        
        # Tools
        self.detector = TrafficDetector()
        self.fusion = DataFusion()
        
        # Traffic Light ID in map.net.xml
        self.tls_id = "J1"
        
        # Phase mapping from map.net.xml
        # 0: NS Green, 1: NS Yellow, 2: EW Green, 3: EW Yellow
        self.PHASE_NS_GREEN = 0
        self.PHASE_NS_YELLOW = 1
        self.PHASE_EW_GREEN = 2
        self.PHASE_EW_YELLOW = 3
        
        self.yellow_duration = 4
        self.green_duration = 10 # Min duration between switches

    def reset(self, seed=None, options=None):
        """
        Restarts the simulation for a new training episode.
        """
        super().reset(seed=seed)
        
        # Close existing if open
        try:
            traci.close()
        except:
            pass
            
        # Start SUMO
        # We use a random port to prevent conflicts during parallel training
        port = 8000 + np.random.randint(0, 1000)
        
        # --- FIX FOR TRACI CWD ERROR ---
        sim_dir = os.path.dirname(self.config_path)
        config_name = os.path.basename(self.config_path)
        
        # We must run the command using just the filename, 
        # but we have to be INSIDE the directory for it to work.
        sumo_cmd = [self.sumo_binary, "-c", config_name]
        
        original_dir = os.getcwd()
        try:
            # 1. Switch to simulation directory
            os.chdir(sim_dir)
            
            # 2. Start TraCI (without 'cwd' argument)
            traci.start(sumo_cmd, port=port)
            
        finally:
            # 3. Always switch back to original directory
            # This is crucial so the RL agent saves the model in the right place
            os.chdir(original_dir)
        
        self.sim_steps = 0
        return self._get_state(), {}

    def step(self, action):
        """
        The Agent takes an action (0 or 1).
        We execute it in SUMO and return the new state and reward.
        """
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        
        # --- ACTION LOGIC ---
        # If action is 0 (Want NS Green)
        if action == 0:
            if current_phase == self.PHASE_EW_GREEN:
                # We are currently EW, need to switch
                self._set_phase(self.PHASE_EW_YELLOW) # Yellow
                self._run_steps(self.yellow_duration)
                self._set_phase(self.PHASE_NS_GREEN) # Green
            # Else: Already NS, just stay green
            
        # If action is 1 (Want EW Green)
        elif action == 1:
            if current_phase == self.PHASE_NS_GREEN:
                # We are currently NS, need to switch
                self._set_phase(self.PHASE_NS_YELLOW) # Yellow
                self._run_steps(self.yellow_duration)
                self._set_phase(self.PHASE_EW_GREEN) # Green
            # Else: Already EW, just stay green

        # Run Green phase for a minimum duration (prevents flickering)
        self._run_steps(self.green_duration)
        
        # --- GET DATA ---
        state = self._get_state()
        
        # --- CALCULATE REWARD ---
        # Reward = Negative Total Queue Length
        # The Agent wants to maximize reward, so it must minimize the queue.
        total_queue = state[0] + state[1] + state[2] + state[3]
        reward = -1 * total_queue
        
        # Check if simulation ended (1 hour)
        terminated = self.sim_steps >= 3600
        truncated = False
        
        return state, reward, terminated, truncated, {}

    def _get_state(self):
        """
        Gathers data using our Detector and Fusion modules.
        """
        sensor_data = self.detector.get_induction_loop_data()
        # We simplify Fusion for the RL agent to just raw queue numbers
        # N, S, E, W queues
        q_n = sensor_data.get("N_to_J1_0", {}).get("occupancy", 0) + \
              sensor_data.get("N_to_J1_1", {}).get("occupancy", 0)
        q_s = sensor_data.get("S_to_J1_0", {}).get("occupancy", 0) + \
              sensor_data.get("S_to_J1_1", {}).get("occupancy", 0)
        q_e = sensor_data.get("E_to_J1_0", {}).get("occupancy", 0) + \
              sensor_data.get("E_to_J1_1", {}).get("occupancy", 0)
        q_w = sensor_data.get("W_to_J1_0", {}).get("occupancy", 0) + \
              sensor_data.get("W_to_J1_1", {}).get("occupancy", 0)
              
        phase = traci.trafficlight.getPhase(self.tls_id)
        
        return np.array([q_n, q_s, q_e, q_w, phase], dtype=np.float32)

    def _set_phase(self, phase_index):
        traci.trafficlight.setPhase(self.tls_id, phase_index)

    def _run_steps(self, steps):
        for _ in range(steps):
            traci.simulationStep()
            self.sim_steps += 1