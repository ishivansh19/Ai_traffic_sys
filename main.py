import os
import sys
import traci
import torch
import pickle
import numpy as np
from collections import deque

# --- 1. Suppress OneDNN Warning ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import DQN

# --- IMPORT MODULES ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from acquisition.detector import TrafficDetector
from acquisition.fusion import DataFusion
from prediction.lstm_model import TrafficLSTM

# --- HELPER FUNCTIONS ---
def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        print("Error: SUMO_HOME environment variable not set.")
        sys.exit(1)
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)

def launch_sumo():
    if sys.platform == "win32":
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
    else:
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simulation_dir = os.path.join(script_dir, "simulation")
    config_name = "config.sumocfg"
    
    required_files = [
        os.path.join(simulation_dir, config_name),
        os.path.join(simulation_dir, "map.net.xml"),
        os.path.join(simulation_dir, "routes.rou.xml")
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"CRITICAL ERROR: Missing file: {f}")
            sys.exit(1)

    sumo_cmd = [sumo_binary, "-c", config_name]
    return sumo_cmd, simulation_dir

def load_lstm_model():
    """Loads the Traffic Predictor (LSTM)."""
    try:
        model_path = "models/lstm/traffic_model.pth"
        scaler_path = "models/lstm/traffic_scaler.pkl"
        
        model = TrafficLSTM(input_size=4, hidden_size=64, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
        print(" [OK] LSTM Predictor loaded.")
        return model, scaler
    except Exception as e:
        print(f" [WARN] LSTM not loaded: {e}")
        return None, None

def load_dqn_agent():
    """Loads the Traffic Controller (RL Agent)."""
    try:
        # We load the zip file created by train_controller.py
        model_path = "models/rl_agent/dqn_traffic"
        model = DQN.load(model_path)
        print(" [OK] DQN Controller loaded.")
        return model
    except Exception as e:
        print(f" [WARN] DQN Agent not loaded: {e}")
        return None

def get_emergency_phase(lane_id, phase_ns, phase_ew):
    """
    Determines which phase should be active to clear the path for the ambulance.
    Based on the lane ID string (e.g., 'N_to_J1_0').
    """
    if not lane_id:
        return None
        
    # If ambulance is coming from North or South, we need NS Green
    if lane_id.startswith("N_") or lane_id.startswith("S_"):
        return phase_ns
    
    # If ambulance is coming from East or West, we need EW Green
    if lane_id.startswith("E_") or lane_id.startswith("W_"):
        return phase_ew
        
    return None

# --- MAIN LOOP ---
def run_simulation(sumo_cmd, simulation_dir):
    print("Starting Final System Simulation...")
    original_dir = os.getcwd()
    
    # 1. Init Modules
    detector = TrafficDetector()
    fusion_engine = DataFusion()
    
    # 2. Load AI Models
    lstm_model, scaler = load_lstm_model()
    dqn_agent = load_dqn_agent()
    
    # 3. Memory Buffers & Control Variables
    history_buffer = deque(maxlen=10)
    
    # Traffic Light Control Variables
    tls_id = "J1"
    last_action_time = 0
    min_green_time = 10
    yellow_time = 4
    
    # Phase constants matching map.net.xml
    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2
    PHASE_EW_YELLOW = 3
    
    try:
        # Switch to simulation dir to load map files correctly
        os.chdir(simulation_dir)
        traci.start(sumo_cmd, port=8813, numRetries=10)
        
        # Fix Camera
        try:
            traci.gui.setZoom("View #0", 600)
            traci.gui.setOffset("View #0", 0, 0)
        except:
            pass 
        
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # --- A. DATA ACQUISITION ---
            sensor_data = detector.get_induction_loop_data()
            vision_data = detector.get_camera_frame_data()
            state = fusion_engine.fuse_data(sensor_data, vision_data)
            
            # --- B. LSTM PREDICTION ---
            prediction_text = "N/A"
            if lstm_model is not None:
                current_features = [
                    state['total_vehicles'], 
                    state['queue_length'], 
                    state['network_speed'], 
                    1 if state['emergency_alert'] else 0
                ]
                history_buffer.append(current_features)
                
                if len(history_buffer) == 10:
                    input_seq = np.array(history_buffer)
                    input_scaled = scaler.transform(input_seq)
                    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0)
                    
                    with torch.no_grad():
                        prediction_scaled = lstm_model(input_tensor).item()
                    
                    dummy_row = np.zeros(4)
                    dummy_row[1] = prediction_scaled
                    prediction_real = scaler.inverse_transform([dummy_row])[0][1]
                    prediction_text = f"{prediction_real:.1f}"

            # --- C. CONTROL LOGIC (DQN + EMERGENCY OVERRIDE) ---
            current_phase = traci.trafficlight.getPhase(tls_id)
            time_since_action = step - last_action_time
            
            # Priority 1: EMERGENCY OVERRIDE (Objective 4)
            if state['emergency_alert']:
                target_phase = get_emergency_phase(state['emergency_lane'], PHASE_NS_GREEN, PHASE_EW_GREEN)
                
                if target_phase is not None:
                    # If we are not in the correct phase, switch immediately
                    if current_phase != target_phase:
                        # If currently green in wrong direction, go yellow first
                        if current_phase == PHASE_NS_GREEN and target_phase == PHASE_EW_GREEN:
                             traci.trafficlight.setPhase(tls_id, PHASE_NS_YELLOW)
                             last_action_time = step + yellow_time
                        elif current_phase == PHASE_EW_GREEN and target_phase == PHASE_NS_GREEN:
                             traci.trafficlight.setPhase(tls_id, PHASE_EW_YELLOW)
                             last_action_time = step + yellow_time
                        elif current_phase in [PHASE_NS_YELLOW, PHASE_EW_YELLOW]:
                             if time_since_action > yellow_time:
                                 traci.trafficlight.setPhase(tls_id, target_phase)
                    else:
                        # We are already in the correct Green phase. Keep it Green!
                        # By setting last_action_time to current step, we prevent DQN from changing it.
                        last_action_time = step 
                        
                    if step % 50 == 0:
                        print(f" >>> EMERGENCY OVERRIDE: Forcing Green for {state['emergency_lane']} <<<")
            
            # Priority 2: AI CONTROL (DQN)
            elif dqn_agent is not None and time_since_action > min_green_time:
                # Only run AI if NO emergency
                
                # Build State Vector
                q_n = sensor_data.get("N_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("N_to_J1_1", {}).get("occupancy", 0)
                q_s = sensor_data.get("S_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("S_to_J1_1", {}).get("occupancy", 0)
                q_e = sensor_data.get("E_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("E_to_J1_1", {}).get("occupancy", 0)
                q_w = sensor_data.get("W_to_J1_0", {}).get("occupancy", 0) + sensor_data.get("W_to_J1_1", {}).get("occupancy", 0)
                
                rl_state = np.array([q_n, q_s, q_e, q_w, current_phase], dtype=np.float32)
                
                # Ask AI
                action, _ = dqn_agent.predict(rl_state, deterministic=True)
                
                # Execute AI Action
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
            
            # --- D. LOGGING ---
            if step % 50 == 0:
                current_phase_name = "NS Green" if traci.trafficlight.getPhase(tls_id) == 0 else "EW Green"
                if traci.trafficlight.getPhase(tls_id) in [1, 3]: current_phase_name = "Yellow"
                
                print(f"Step {step:4} | Queue: {state['queue_length']:3} | Predicted: {prediction_text:5} | Light: {current_phase_name}")

            step += 1

        print("Simulation finished.")

    except traci.TraCIException as e:
        print(f"TraCI Error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            traci.close()
        except:
            pass
        os.chdir(original_dir)

if __name__ == "__main__":
    check_sumo_home()
    sumo_command, sim_dir = launch_sumo()
    run_simulation(sumo_command, sim_dir)