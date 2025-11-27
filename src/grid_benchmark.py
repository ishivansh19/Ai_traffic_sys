import os
import sys
import traci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
STEPS_TO_RUN = 2000 
VISUAL_MODE = True

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- SMART DETECTOR ---
class SmartDetector:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.incoming_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))

    def get_state(self):
        queues = []
        for lane in self.incoming_lanes:
            try:
                occ = traci.lane.getLastStepVehicleNumber(lane)
                queues.append(occ)
            except:
                queues.append(0)
        
        total = len(queues)
        if total == 0: return [0,0,0,0]
        
        inputs = [0,0,0,0]
        chunk = int(np.ceil(total/4))
        for i in range(4):
            start = i*chunk
            end = min((i+1)*chunk, total)
            if start < total: inputs[i] = sum(queues[start:end])
        return inputs

def get_simulation_config():
    if VISUAL_MODE:
        if sys.platform == "win32":
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
        else:
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
    else:
        if sys.platform == "win32":
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe')
        else:
            sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    simulation_dir = os.path.join(root_dir, 'simulation_grid')
    config_name = "grid.sumocfg"
    
    sumo_cmd = [
        sumo_binary, 
        "-c", config_name, 
        "--start", 
        "--quit-on-end",
        "--no-step-log", "true", 
        "--duration-log.disable", "true"
    ]
    
    return sumo_cmd, simulation_dir, root_dir

def get_network_stats():
    total_wait = 0
    total_speed = 0
    vehs = traci.vehicle.getIDList()
    veh_count = len(vehs)
    
    if veh_count == 0: return 0, 0
        
    for veh in vehs:
        total_wait += traci.vehicle.getWaitingTime(veh)
        total_speed += traci.vehicle.getSpeed(veh)
        
    return total_wait, (total_speed / veh_count)

def discover_phases(tls_id):
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    logic = logics[0]
    
    ns_green = -1
    ew_green = -1
    
    for i, phase in enumerate(logic.phases):
        state = phase.state
        mid = len(state) // 2
        if 'G' in state[:mid] and 'G' not in state[mid:]:
            ns_green = i
        elif 'G' not in state[:mid] and 'G' in state[mid:]:
            ew_green = i
            
    if ns_green == -1: ns_green = 0
    if ew_green == -1: ew_green = 2
    
    return ns_green, ew_green

def run_grid_episode(mode, sumo_cmd, simulation_dir, root_dir):
    print(f"\n--- Running {mode.upper()} Simulation (2x2 Grid) ---")
    
    wait_metrics = []
    speed_metrics = []
    
    dqn_agent = None
    if mode == 'ai':
        model_path = os.path.join(root_dir, 'models', 'rl_agent', 'dqn_traffic')
        try:
            dqn_agent = DQN.load(model_path)
            print("AI Model Loaded.")
        except:
            print("Error: Model not found.")
            return [], []

    original_dir = os.getcwd()
    os.chdir(simulation_dir)
    
    traci.start(sumo_cmd, port=9000 + (1 if mode=='ai' else 0))
    
    # REMOVED ZOOM LOGIC HERE
    
    tls_ids = traci.trafficlight.getIDList()
    detectors = {tls: SmartDetector(tls) for tls in tls_ids}
    last_actions = {tls: 0 for tls in tls_ids}
    
    phase_map = {}
    for tls in tls_ids:
        ns, ew = discover_phases(tls)
        phase_map[tls] = {'NS': ns, 'EW': ew}
        print(f"Light {tls} -> NS Green Index: {ns}, EW Green Index: {ew}")

    try:
        for step in range(STEPS_TO_RUN):
            traci.simulationStep()
            
            wait, speed = get_network_stats()
            wait_metrics.append(wait)
            speed_metrics.append(speed)
            
            if mode == 'ai':
                for tls in tls_ids:
                    inputs = detectors[tls].get_state()
                    current_phase = traci.trafficlight.getPhase(tls)
                    
                    mapped_phase_input = 0
                    if current_phase == phase_map[tls]['EW']: mapped_phase_input = 2
                    
                    rl_state = np.array(inputs + [mapped_phase_input], dtype=np.float32)
                    action, _ = dqn_agent.predict(rl_state, deterministic=True)
                    
                    ns_idx = phase_map[tls]['NS']
                    ew_idx = phase_map[tls]['EW']
                    
                    if step - last_actions[tls] > 10:
                        if action == 0: 
                            if current_phase != ns_idx:
                                traci.trafficlight.setPhase(tls, ns_idx)
                                last_actions[tls] = step
                        elif action == 1: 
                            if current_phase != ew_idx:
                                traci.trafficlight.setPhase(tls, ew_idx)
                                last_actions[tls] = step

            if step % 500 == 0:
                print(f"Step {step}/{STEPS_TO_RUN} | Total Network Wait: {wait:.0f}s")
                
    finally:
        traci.close()
        os.chdir(original_dir)
        
    return wait_metrics, speed_metrics

def analyze_grid_results(base_wait, ai_wait, base_speed, ai_speed, root_dir):
    print("\n--- 2x2 GRID RESULTS ANALYSIS ---")
    
    avg_base_wait = np.mean(base_wait)
    avg_ai_wait = np.mean(ai_wait)
    avg_base_speed = np.mean(base_speed)
    avg_ai_speed = np.mean(ai_speed)
    
    imp_wait = 0.0
    if avg_base_wait > 0:
        imp_wait = ((avg_base_wait - avg_ai_wait) / avg_base_wait) * 100
        
    imp_speed = 0.0
    if avg_base_speed > 0:
        imp_speed = ((avg_ai_speed - avg_base_speed) / avg_base_speed) * 100
    
    print(f"Avg Wait Time: Baseline={avg_base_wait:.0f}s | AI={avg_ai_wait:.0f}s | Improvement={imp_wait:.2f}%")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(base_wait, label='Fixed Timers', color='red', alpha=0.6)
    ax1.plot(ai_wait, label='AI Grid Control', color='green', linewidth=2)
    ax1.set_title("Total Network Congestion")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Waiting Time (s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(base_speed, label='Fixed Timers', color='red', alpha=0.6)
    ax2.plot(ai_speed, label='AI Grid Control', color='green', linewidth=2)
    ax2.set_title("Average Traffic Speed")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plot_path = os.path.join(root_dir, 'grid_benchmark_results.png')
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")

if __name__ == "__main__":
    sumo_cmd, sim_dir, root_dir = get_simulation_config()
    b_wait, b_speed = run_grid_episode('baseline', sumo_cmd, sim_dir, root_dir)
    a_wait, a_speed = run_grid_episode('ai', sumo_cmd, sim_dir, root_dir)
    if b_wait and a_wait:
        analyze_grid_results(b_wait, a_wait, b_speed, a_speed, root_dir)