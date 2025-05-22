from dynamic_objects import DynamicObjectDetector
from path_planner import PathPlanner
from visualization import Visualizer
from ros_pipeline import DataLoader

def main():
    rclpy.init()
    data_loader = DataLoader(slam_method="cuvslam")
    detector = DynamicObjectDetector()
    planner = PathPlanner()
    visualizer = Visualizer()

    while rclpy.ok():
        rclpy.spin_once(data_loader)

        if data_loader.rgb_image is not None and data_loader.depth_image is not None and data_loader.pose is not None:
            dynamic_objects = detector.detect_objects(data_loader.depth_image, data_loader.pose)
            safe_path = planner.compute_safe_path(start=np.array([0, 0, 0]), goal=np.array([10, 10, 10]), dynamic_objects=dynamic_objects)
            visualizer.visualize(safe_path, dynamic_objects)

    rclpy.shutdown()

if __name__ == "__main__":
    main()