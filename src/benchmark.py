import os
import sys
import traci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
STEPS_TO_RUN = 3600  # Run for 1 hour of simulation time

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# We are in AI_Traffic_System/src
# We want to add AI_Traffic_System/src to path to find 'acquisition'
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Acquisition (we only need detector for benchmarking logic)
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
    """Returns the sumo command and directory."""
    if sys.platform == "win32":
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe') # HEADLESS for speed
    else:
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    
    # --- FIX IS HERE ---
    # current_dir is ".../AI_Traffic_System/src"
    # We go up ONE level ("..") to get to ".../AI_Traffic_System"
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    simulation_dir = os.path.join(root_dir, 'simulation')
    config_name = "config.sumocfg"
    sumo_cmd = [sumo_binary, "-c", config_name, "--no-step-log", "true", "--duration-log.disable", "true"]
    
    return sumo_cmd, simulation_dir, root_dir

def get_total_waiting_time(detector):
    """Calculates the sum of waiting time (speed < 0.1m/s) for all incoming lanes."""
    total_wait = 0
    for lane in detector.sensor_lanes:
        try:
            total_wait += traci.lane.getWaitingTime(lane)
        except:
            pass
    return total_wait

def run_episode(mode, sumo_cmd, simulation_dir, root_dir):
    """
    Runs a single simulation episode.
    mode: 'baseline' (Fixed Timers) or 'ai' (DQN Agent)
    """
    print(f"\n--- Running {mode.upper()} Simulation ---")
    
    # 1. Setup
    metrics = []
    detector = TrafficDetector()
    
    dqn_agent = None
    if mode == 'ai':
        model_path = os.path.join(root_dir, 'models', 'rl_agent', 'dqn_traffic')
        try:
            dqn_agent = DQN.load(model_path)
            print("AI Model Loaded.")
        except:
            print("CRITICAL ERROR: Could not load AI model. Run training first.")
            return []

    # Control Variables
    tls_id = "J1"
    last_action_time = 0
    min_green_time = 10
    yellow_time = 4
    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2
    PHASE_EW_YELLOW = 3

    # 2. Start SUMO
    original_dir = os.getcwd()
    
    # Ensure simulation directory exists before changing to it
    if not os.path.exists(simulation_dir):
        print(f"CRITICAL ERROR: Simulation directory not found at: {simulation_dir}")
        return []

    os.chdir(simulation_dir)
    traci.start(sumo_cmd, port=9000 + (1 if mode=='ai' else 0))
    
    try:
        for step in range(STEPS_TO_RUN):
            traci.simulationStep()
            
            # --- METRICS COLLECTION ---
            # We record the Total Waiting Time at this exact second
            current_wait = get_total_waiting_time(detector)
            metrics.append(current_wait)
            
            # --- AI CONTROL LOGIC (Only for AI mode) ---
            if mode == 'ai':
                time_since_action = step - last_action_time
                if time_since_action > min_green_time:
                    # Build State
                    sensor_data = detector.get_induction_loop_data()
                    q_n = sensor_data.get("N_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("N_to_J1_1", {}).get("occupancy", 0)
                    q_s = sensor_data.get("S_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("S_to_J1_1", {}).get("occupancy", 0)
                    q_e = sensor_data.get("E_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("E_to_J1_1", {}).get("occupancy", 0)
                    q_w = sensor_data.get("W_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("W_to_J1_1", {}).get("occupancy", 0)
                    current_phase = traci.trafficlight.getPhase(tls_id)
                    
                    rl_state = np.array([q_n, q_s, q_e, q_w, current_phase], dtype=np.float32)
                    
                    # Predict
                    action, _ = dqn_agent.predict(rl_state, deterministic=True)
                    
                    # Act
                    if action == 0: # Want NS Green
                        if current_phase == PHASE_EW_GREEN:
                            traci.trafficlight.setPhase(tls_id, PHASE_EW_YELLOW)
                            last_action_time = step + yellow_time
                        elif current_phase == PHASE_EW_YELLOW and time_since_action > yellow_time:
                                traci.trafficlight.setPhase(tls_id, PHASE_NS_GREEN)
                    elif action == 1: # Want EW Green
                        if current_phase == PHASE_NS_GREEN:
                            traci.trafficlight.setPhase(tls_id, PHASE_NS_YELLOW)
                            last_action_time = step + yellow_time
                        elif current_phase == PHASE_NS_YELLOW and time_since_action > yellow_time:
                            traci.trafficlight.setPhase(tls_id, PHASE_EW_GREEN)

            if step % 500 == 0:
                print(f"Step {step}/{STEPS_TO_RUN} | Current Wait: {current_wait}s")

    finally:
        try:
            traci.close()
        except:
            pass
        os.chdir(original_dir)
        
    return metrics

def analyze_results(baseline_data, ai_data, root_dir):
    print("\n--- ANALYZING RESULTS ---")
    
    # 1. Calculate Statistics
    avg_base = np.mean(baseline_data)
    avg_ai = np.mean(ai_data)
    
    # Avoid division by zero if wait is 0
    if avg_base > 0:
        improvement = ((avg_base - avg_ai) / avg_base) * 100
    else:
        improvement = 0.0
    
    print(f"Average Waiting Time (Baseline): {avg_base:.2f} seconds")
    print(f"Average Waiting Time (AI Agent): {avg_ai:.2f} seconds")
    print(f"IMPROVEMENT: {improvement:.2f}%")
    
    # 2. Save Data
    data_dir = os.path.join(root_dir, 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    df = pd.DataFrame({
        'step': range(len(baseline_data)),
        'baseline_wait': baseline_data,
        'ai_wait': ai_data
    })
    csv_path = os.path.join(data_dir, 'benchmark_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to: {csv_path}")
    
    # 3. Plot Graph
    plt.figure(figsize=(12, 6))
    plt.plot(df['baseline_wait'], label='Fixed Timer (Baseline)', color='red', alpha=0.6)
    plt.plot(df['ai_wait'], label='AI Controller (DQN)', color='green', linewidth=2)
    
    plt.title(f"Traffic Control Efficiency: Baseline vs AI\nImprovement: {improvement:.1f}%")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Accumulated Waiting Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(root_dir, 'benchmark_performance.png')
    plt.savefig(plot_path)
    print(f"Performance Graph saved to: {plot_path}")
    print("You can open this image to visualize the AI's efficiency.")

if __name__ == "__main__":
    check_sumo_home()
    sumo_cmd, sim_dir, root_dir = get_simulation_config()
    
    # Run Comparison
    baseline_metrics = run_episode('baseline', sumo_cmd, sim_dir, root_dir)
    ai_metrics = run_episode('ai', sumo_cmd, sim_dir, root_dir)
    
    if baseline_metrics and ai_metrics:
        analyze_results(baseline_metrics, ai_metrics, root_dir)