import os
import sys
import subprocess

def check_sumo_home():
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME not set.")
        sys.exit(1)
    return os.environ['SUMO_HOME']

def build():
    sumo_home = check_sumo_home()
    
    # Locate netconvert executable
    if sys.platform == "win32":
        netconvert = os.path.join(sumo_home, 'bin', 'netconvert.exe')
    else:
        netconvert = os.path.join(sumo_home, 'bin', 'netconvert')

    if not os.path.exists(netconvert):
        print(f"Error: Could not find netconvert at {netconvert}")
        return

    # Define paths
    # We assume this script is in the root and files are in simulation/
    base_dir = os.path.join(os.path.dirname(__file__), "simulation")
    
    nodes = os.path.join(base_dir, "map.nodes.xml")
    edges = os.path.join(base_dir, "map.edges.xml")
    conns = os.path.join(base_dir, "map.connections.xml")
    
    # The Output File (This is what SUMO GUI actually needs)
    output = os.path.join(base_dir, "map.net.xml")

    print("Compiling network...")
    
    # The command to compile the map
    # We add --no-turnarounds to prevent U-turns at the intersection
    # We add --tls.guess to automatically add traffic lights at the intersection
    cmd = [
        netconvert,
        "--node-files", nodes,
        "--edge-files", edges,
        "--connection-files", conns,
        "--output-file", output,
        "--no-turnarounds",
        "--tls.guess" 
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSUCCESS! Compiled network saved to: {output}")
        print("You can now run main.py")
    except subprocess.CalledProcessError as e:
        print("\nCOMPILATION FAILED.")
        print("Please check your XML files for syntax errors.")

if __name__ == "__main__":
    build()