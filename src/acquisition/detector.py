import traci
import numpy as np

class TrafficDetector:
    """
    Simulates the hardware layer. 
    Responsible for retrieving raw data from the environment (SUMO).
    """
    def __init__(self):
        # Define the IDs of the induction loops (detectors) we want to monitor
        # In our map, these are the lane IDs entering the intersection
        self.sensor_lanes = [
            "N_to_J1_0", "N_to_J1_1", "N_to_J1_2",
            "S_to_J1_0", "S_to_J1_1", "S_to_J1_2",
            "E_to_J1_0", "E_to_J1_1", "E_to_J1_2",
            "W_to_J1_0", "W_to_J1_1", "W_to_J1_2"
        ]

    def get_induction_loop_data(self):
        """
        Simulates reading data from magnetic loops embedded in the road.
        Returns: Dictionary of {lane_id: {speed, vehicle_count}}
        """
        data = {}
        for lane in self.sensor_lanes:
            try:
                # Get number of vehicles on the lane in the last step
                count = traci.lane.getLastStepVehicleNumber(lane)
                # Get average speed on the lane (returns -1 if empty)
                speed = traci.lane.getLastStepMeanSpeed(lane)
                
                # Clean the data: if speed is negative (empty), set to 0
                if speed < 0:
                    speed = 0.0
                    
                data[lane] = {
                    "occupancy": count,
                    "avg_speed": speed
                }
            except traci.TraCIException:
                # Handle cases where lane might not exist momentarily
                data[lane] = {"occupancy": 0, "avg_speed": 0.0}
        
        return data

    def get_camera_frame_data(self):
        """
        Simulates a Computer Vision system (like YOLO).
        Instead of processing pixels (which is slow), we query the simulation
        for 'Ground Truth' but format it like a vision detection output.
        
        Returns: List of detected objects with Class and Confidence.
        """
        vision_objects = []
        
        # Scan all lanes for vehicles
        for lane in self.sensor_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            
            for vid in vehicle_ids:
                # Get vehicle type (car, truck, ambulance)
                v_type = traci.vehicle.getTypeID(vid)
                # Get position
                pos = traci.vehicle.getPosition(vid)
                
                # Simulate YOLO Output Format
                detection = {
                    "id": vid,
                    "class": v_type,       # e.g., "ambulance", "car"
                    "confidence": 0.98,    # Simulated high confidence
                    "lane": lane,
                    "location": pos        # (x, y) coords
                }
                vision_objects.append(detection)
                
        return vision_objects