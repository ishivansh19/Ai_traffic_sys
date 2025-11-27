import os
import sys
import subprocess
import time

def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        return os.environ['SUMO_HOME']
    else:
        print("Error: SUMO_HOME environment variable not set.")
        sys.exit(1)

def run_diagnosis():
    sumo_home = check_sumo_home()
    
    # 1. Identify directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(script_dir, "simulation")
    
    print(f"--- DIAGNOSTIC TOOL ---")
    print(f"Checking directory: {sim_dir}")

    # 2. Verify Files Exist
    required_files = ["map.nodes.xml", "map.edges.xml", "map.connections.xml", "routes.rou.xml", "config.sumocfg"]
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(sim_dir, f)):
            missing.append(f)
    
    if missing:
        print(f"CRITICAL ERROR: The following files are missing in 'simulation/':")
        for m in missing:
            print(f" - {m}")
        print("Please create these files before continuing.")
        return

    # 3. Attempt to Compile Network (Re-run build logic)
    print("\n--- STEP 1: COMPILING MAP ---")
    if sys.platform == "win32":
        netconvert = os.path.join(sumo_home, 'bin', 'netconvert.exe')
    else:
        netconvert = os.path.join(sumo_home, 'bin', 'netconvert')

    cmd_build = [
        netconvert,
        "--node-files", os.path.join(sim_dir, "map.nodes.xml"),
        "--edge-files", os.path.join(sim_dir, "map.edges.xml"),
        "--connection-files", os.path.join(sim_dir, "map.connections.xml"),
        "--output-file", os.path.join(sim_dir, "map.net.xml"),
        "--no-turnarounds",
        "--tls.guess"
    ]
    
    try:
        result = subprocess.run(cmd_build, capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: Map Compilation Failed!")
            print(result.stderr)
            return
        else:
            print("SUCCESS: Map compiled. 'map.net.xml' created.")
    except FileNotFoundError:
        print("ERROR: Could not find netconvert tool.")
        return

    # 4. Attempt to Run SUMO Headless
    print("\n--- STEP 2: DRY RUNNING SIMULATION ---")
    print("Running SUMO without GUI to capture errors...")
    
    if sys.platform == "win32":
        sumo_bin = os.path.join(sumo_home, 'bin', 'sumo.exe') # Note: sumo.exe, not sumo-gui.exe
    else:
        sumo_bin = os.path.join(sumo_home, 'bin', 'sumo')

    # We run for just 5 steps to see if it crashes on load
    cmd_run = [
        sumo_bin, 
        "-c", "config.sumocfg",
        "--end", "5" 
    ]
    
    # Switch dir so config paths work
    original_dir = os.getcwd()
    os.chdir(sim_dir)
    
    result = subprocess.run(cmd_run, capture_output=True, text=True)
    
    os.chdir(original_dir)

    if result.returncode != 0:
        print("\n!!! CRASH DETECTED !!!")
        print("Here is the exact error from SUMO:")
        print("---------------------------------------------------")
        print(result.stderr)
        print("---------------------------------------------------")
        print("Fix the error above in your XML files.")
    else:
        print("\nSUCCESS! The simulation runs perfectly in headless mode.")
        print("This means your XML files are valid.")
        print("If main.py still fails, the issue is the TraCI port connection.")

if __name__ == "__main__":
    run_diagnosis()