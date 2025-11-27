import os
import sys
import traci
import csv
import random

# --- PATH SETUP ---
# Add the src directory to the path so we can import acquisition modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import our existing modules
from acquisition.detector import TrafficDetector
from acquisition.fusion import DataFusion

def check_sumo_home():
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME not set.")
        sys.exit(1)
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)

def run_generation(steps=5000):
    check_sumo_home()
    
    # 1. Setup Paths
    # We go up two levels from src/prediction/ to get to root
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    simulation_dir = os.path.join(root_dir, 'simulation')
    data_dir = os.path.join(root_dir, 'data')
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    csv_path = os.path.join(data_dir, 'traffic_history.csv')
    
    # 2. Launch Config (HEADLESS MODE)
    if sys.platform == "win32":
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe')
    else:
        sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        
    sumo_cmd = [sumo_binary, "-c", "config.sumocfg"]
    
    print(f"--- DATA GENERATION STARTED ---")
    print(f"Running {steps} simulation steps...")
    print(f"Saving to: {csv_path}")
    
    # 3. Initialize Modules
    detector = TrafficDetector()
    fusion = DataFusion()
    
    # 4. Open CSV for writing
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: We need sequences of these to train the LSTM
        writer.writerow(['step', 'total_vehicles', 'queue_length', 'network_speed', 'emergency_active'])
        
        # 5. Run Simulation
        original_dir = os.getcwd()
        try:
            os.chdir(simulation_dir)
            traci.start(sumo_cmd, port=8814) # Use a different port just in case
            
            for step in range(steps):
                traci.simulationStep()
                
                # Get Data
                sensor_data = detector.get_induction_loop_data()
                vision_data = detector.get_camera_frame_data()
                state = fusion.fuse_data(sensor_data, vision_data)
                
                # Write to CSV
                writer.writerow([
                    step,
                    state['total_vehicles'],
                    state['queue_length'],
                    state['network_speed'],
                    1 if state['emergency_alert'] else 0
                ])
                
                if step % 500 == 0:
                    print(f"Progress: {step}/{steps} steps completed.")
                    
            print("Data generation complete.")
            
        except Exception as e:
            print(f"Error during generation: {e}")
        finally:
            try:
                traci.close()
            except:
                pass
            os.chdir(original_dir)

if __name__ == "__main__":
    # Run for 10,000 steps to get a good dataset
    run_generation(steps=10000)