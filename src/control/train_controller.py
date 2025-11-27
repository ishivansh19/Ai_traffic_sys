import os
import sys

# --- 1. Suppress OneDNN Warning ---
# This must be done BEFORE importing torch or stable_baselines3
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import DQN
from sumo_env import SumoTrafficEnv

def check_sumo_home():
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME not set.")
        sys.exit(1)
    return os.environ['SUMO_HOME']

def train():
    sumo_home = check_sumo_home()
    
    # Setup Paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(root_dir, 'simulation', 'config.sumocfg')
    model_dir = os.path.join(root_dir, 'models', 'rl_agent')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Identify SUMO Binary (Headless for training)
    if sys.platform == "win32":
        sumo_binary = os.path.join(sumo_home, 'bin', 'sumo.exe')
    else:
        sumo_binary = os.path.join(sumo_home, 'bin', 'sumo')

    print("--- STARTING RL TRAINING ---")
    
    # 1. Create the Environment
    env = SumoTrafficEnv(sumo_binary, config_path)
    
    # 2. Initialize the Agent (DQN)
    # We use MlpPolicy (Multi-Layer Perceptron) which is standard for vector inputs
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=5000)
    
    # 3. Train
    # 20,000 timesteps allows it to see roughly 5-6 full simulation hours
    print("Training Agent... this may take a few minutes.")
    try:
        model.learn(total_timesteps=20000)
        
        # 4. Save
        save_path = os.path.join(model_dir, "dqn_traffic")
        model.save(save_path)
        print(f"Model saved to {save_path}.zip")
        
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        # Close env
        env.close()

if __name__ == "__main__":
    train()