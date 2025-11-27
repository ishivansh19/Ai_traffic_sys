import numpy as np

class DataFusion:
    """
    Responsible for fusing heterogeneous data sources (Vision + Sensors)
    into a unified state representation.
    """
    def __init__(self):
        self.emergency_active = False
        self.emergency_location = None

    def fuse_data(self, sensor_data, vision_data):
        """
        Merges loop sensor data with camera detections.
        
        Args:
            sensor_data: Dict from get_induction_loop_data()
            vision_data: List from get_camera_frame_data()
            
        Returns:
            fused_state: A structured dictionary of the current traffic situation.
        """
        # 1. Reset Emergency Flag
        self.emergency_active = False
        self.emergency_location = None
        
        # 2. Process Vision Data (Check for Ambulances)
        vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "ambulance": 0}
        
        for obj in vision_data:
            v_class = obj['class']
            if v_class in vehicle_counts:
                vehicle_counts[v_class] += 1
            
            # OBJECTIVE 4 TRIGGER: Detect Emergency Vehicle
            if v_class == "ambulance":
                self.emergency_active = True
                self.emergency_location = obj['lane']
                # High confidence detection of EV
        
        # 3. Aggregating Loop Data (Calculate Total Density)
        total_queue = 0
        avg_network_speed = 0.0
        active_lanes = 0
        
        for lane, info in sensor_data.items():
            total_queue += info['occupancy']
            if info['occupancy'] > 0:
                avg_network_speed += info['avg_speed']
                active_lanes += 1
        
        if active_lanes > 0:
            avg_network_speed /= active_lanes

        # 4. Construct the Fused State
        fused_state = {
            "total_vehicles": sum(vehicle_counts.values()),
            "queue_length": total_queue,
            "network_speed": round(avg_network_speed, 2),
            "class_distribution": vehicle_counts,
            "emergency_alert": self.emergency_active,
            "emergency_lane": self.emergency_location
        }
        
        return fused_state