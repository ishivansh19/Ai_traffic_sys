import os
import sys
import subprocess

def check_sumo_home():
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME not set.")
        sys.exit(1)
    return os.environ['SUMO_HOME']

def build_grid():
    sumo_home = check_sumo_home()
    
    # 1. Define New Directory (Sandbox)
    base_dir = os.path.join(os.path.dirname(__file__), "simulation_grid")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    print(f"Creating 2x2 Grid Map in: {base_dir}")

    # 2. Identify Tools
    if sys.platform == "win32":
        netgenerate = os.path.join(sumo_home, 'bin', 'netgenerate.exe')
        randomtrips = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    else:
        netgenerate = os.path.join(sumo_home, 'bin', 'netgenerate')
        randomtrips = os.path.join(sumo_home, 'tools', 'randomTrips.py')

    # 3. Generate 2x2 Grid Network
    net_file = os.path.join(base_dir, "grid.net.xml")
    cmd_net = [
        netgenerate,
        "--grid",
        "--grid.number", "2",       
        "--grid.length", "200",
        "--output-file", net_file,
        "--tls.guess", "true",
        "--default.speed", "13.89",
        "--no-turnarounds"
    ]
    
    print("Generating Grid Network...")
    subprocess.run(cmd_net, check=True)
    
    # 4. Generate Random Traffic Demand
    # CHANGE: Period set to 0.8. This is the "Sweet Spot".
    # Enough traffic to cause queues, but not enough to cause instant gridlock.
    route_file = os.path.join(base_dir, "grid.rou.xml")
    cmd_route = [
        "python", randomtrips,
        "-n", net_file,
        "-o", route_file,
        "--period", "0.8", 
        "--end", "3600"
    ]
    
    print("Generating Random Traffic Demand...")
    subprocess.run(cmd_route, check=True)
    
    # 5. Create Config File
    config_content = f"""<configuration>
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="grid.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <gui_only>
        <delay value="50"/>
    </gui_only>
</configuration>
"""
    with open(os.path.join(base_dir, "grid.sumocfg"), "w") as f:
        f.write(config_content)
        
    print("\nSUCCESS! 2x2 Grid simulation ready.")

if __name__ == "__main__":
    build_grid()