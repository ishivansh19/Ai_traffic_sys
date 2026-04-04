import os
import sys
import traci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
STEPS_TO_RUN = 3600  

# >>> WEATHER TOGGLE <<<
# Set to True to test Rain/Fog. Set to False for Sunny weather.
ADVERSE_WEATHER = True 

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from acquisition.detector import TrafficDetector

def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        if tools not in sys.path:
            sys.path.append(tools)
    else:
        print("Error: SUMO_HOME not set.")
        sys.exit(1)

def get_simulation_config():
    # Force GUI mode so you can see the blue cars!
    if sys.platform == "win32":
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
    else:
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
    
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    simulation_dir = os.path.join(root_dir, 'simulation')
    config_name = "config.sumocfg"
    sumo_cmd = [sumo_binary, "-c", config_name, "--start", "--quit-on-end"]
    
    return sumo_cmd, simulation_dir, root_dir

def apply_weather_physics():
    """Dynamically alters the physics of all vehicles via TraCI API."""
    print(" >>> INJECTING ADVERSE WEATHER PHYSICS (RAIN/FOG) <<< ")
    try:
        # Note: TraCI expects maxSpeed in meters per second (m/s).
        # 15 km/h = ~4.16 m/s
        
        # 1. Car Physics
        traci.vehicletype.setMaxSpeed("car", 4.16)
        traci.vehicletype.setAccel("car", 1.2)
        traci.vehicletype.setDecel("car", 2.0)
        traci.vehicletype.setTau("car", 2.0) # Slower reaction time
        traci.vehicletype.setColor("car", (0, 0, 255, 255)) # Turn Blue
        
        # 2. Truck Physics
        traci.vehicletype.setMaxSpeed("truck", 2.77) # 10 km/h
        traci.vehicletype.setAccel("truck", 0.8)
        traci.vehicletype.setDecel("truck", 1.5)
        traci.vehicletype.setTau("truck", 2.5)
        traci.vehicletype.setColor("truck", (0, 0, 255, 255)) # Turn Blue
        
        # 3. Bus Physics
        traci.vehicletype.setMaxSpeed("bus", 2.77)
        traci.vehicletype.setAccel("bus", 0.9)
        traci.vehicletype.setDecel("bus", 1.8)
        traci.vehicletype.setTau("bus", 2.5)
        traci.vehicletype.setColor("bus", (0, 0, 255, 255)) # Turn Blue
        
        # 4. Ambulance Physics (Remains Red for visibility, but slower)
        traci.vehicletype.setMaxSpeed("ambulance", 6.94) # 25 km/h
        traci.vehicletype.setAccel("ambulance", 2.0)
        traci.vehicletype.setDecel("ambulance", 3.0)
        traci.vehicletype.setTau("ambulance", 1.5)
        
    except Exception as e:
        print(f"Warning: Could not apply weather physics. Error: {e}")

def get_total_waiting_time(detector):
    total_wait = 0
    for lane in detector.sensor_lanes:
        try:
            total_wait += traci.lane.getWaitingTime(lane)
        except:
            pass
    return total_wait

def run_episode(mode, sumo_cmd, simulation_dir, root_dir):
    print(f"\n--- Running {mode.upper()} Simulation ---")
    
    metrics = []
    detector = TrafficDetector()
    
    dqn_agent = None
    if mode == 'ai':
        model_path = os.path.join(root_dir, 'models', 'rl_agent', 'dqn_traffic')
        try:
            dqn_agent = DQN.load(model_path)
            print("AI Model Loaded.")
        except:
            print("CRITICAL ERROR: Could not load AI model.")
            return []

    tls_id = "J1"
    last_action_time = 0
    # min_green_time = 10
    min_green_time = 15 if ADVERSE_WEATHER else 10 
    yellow_time = 4
    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2
    PHASE_EW_YELLOW = 3

    original_dir = os.getcwd()
    os.chdir(simulation_dir)
    traci.start(sumo_cmd, port=9000 + (1 if mode=='ai' else 0))
    
    # -> INJECT WEATHER ON STARTUP <-
    if ADVERSE_WEATHER:
        apply_weather_physics()
    
    try:
        for step in range(STEPS_TO_RUN):
            traci.simulationStep()
            
            current_wait = get_total_waiting_time(detector)
            metrics.append(current_wait)
            
            if mode == 'ai':
                time_since_action = step - last_action_time
                if time_since_action > min_green_time:
                    sensor_data = detector.get_induction_loop_data()
                    q_n = sensor_data.get("N_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("N_to_J1_1", {}).get("occupancy", 0)
                    q_s = sensor_data.get("S_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("S_to_J1_1", {}).get("occupancy", 0)
                    q_e = sensor_data.get("E_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("E_to_J1_1", {}).get("occupancy", 0)
                    q_w = sensor_data.get("W_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("W_to_J1_1", {}).get("occupancy", 0)
                    current_phase = traci.trafficlight.getPhase(tls_id)
                    
                    rl_state = np.array([q_n, q_s, q_e, q_w, current_phase], dtype=np.float32)
                    action, _ = dqn_agent.predict(rl_state, deterministic=True)
                    
                    if action == 0: 
                        if current_phase == PHASE_EW_GREEN:
                            traci.trafficlight.setPhase(tls_id, PHASE_EW_YELLOW)
                            last_action_time = step + yellow_time
                        elif current_phase == PHASE_EW_YELLOW and time_since_action > yellow_time:
                                traci.trafficlight.setPhase(tls_id, PHASE_NS_GREEN)
                    elif action == 1: 
                        if current_phase == PHASE_NS_GREEN:
                            traci.trafficlight.setPhase(tls_id, PHASE_NS_YELLOW)
                            last_action_time = step + yellow_time
                        elif current_phase == PHASE_NS_YELLOW and time_since_action > yellow_time:
                            traci.trafficlight.setPhase(tls_id, PHASE_EW_GREEN)

            if step % 500 == 0:
                print(f"Step {step}/{STEPS_TO_RUN} | Current Wait: {current_wait}s")

    finally:
        traci.close()
        os.chdir(original_dir)
        
    return metrics

def analyze_results(baseline_data, ai_data, root_dir):
    print("\n--- ANALYZING RESULTS ---")
    
    avg_base = np.mean(baseline_data)
    avg_ai = np.mean(ai_data)
    if avg_base > 0:
        improvement = ((avg_base - avg_ai) / avg_base) * 100
    else:
        improvement = 0.0
    
    print(f"Average Waiting Time (Baseline): {avg_base:.2f} seconds")
    print(f"Average Waiting Time (AI Agent): {avg_ai:.2f} seconds")
    print(f"IMPROVEMENT: {improvement:.2f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(baseline_data, label='Fixed Timer (Baseline)', color='red', alpha=0.6)
    plt.plot(ai_data, label='AI Controller (DQN)', color='green', linewidth=2)
    
    # Title changes based on weather toggle
    title_prefix = "ADVERSE WEATHER (Rain/Fog):" if ADVERSE_WEATHER else "NORMAL WEATHER:"
    plt.title(f"{title_prefix} Traffic Control Efficiency\nImprovement: {improvement:.1f}%")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Accumulated Waiting Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save with different names based on weather toggle
    filename = 'weather_performance.png' if ADVERSE_WEATHER else 'benchmark_performance.png'
    plot_path = os.path.join(root_dir, filename)
    plt.savefig(plot_path)
    print(f"Performance Graph saved to: {plot_path}")

if __name__ == "__main__":
    check_sumo_home()
    sumo_cmd, sim_dir, root_dir = get_simulation_config()
    
    baseline_metrics = run_episode('baseline', sumo_cmd, sim_dir, root_dir)
    ai_metrics = run_episode('ai', sumo_cmd, sim_dir, root_dir)
    
    if baseline_metrics and ai_metrics:
        analyze_results(baseline_metrics, ai_metrics, root_dir)