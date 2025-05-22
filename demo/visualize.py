import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self):
        pass

    def visualize(self, path, dynamic_objects):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot path
        ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Drone Path')

        # Plot dynamic objects
        for obj in dynamic_objects:
            ellipsoid = create_ellipsoid(obj)
            ax.scatter(ellipsoid[:, 0], ellipsoid[:, 1], ellipsoid[:, 2], label='Dynamic Object')

        plt.legend()
        plt.show()