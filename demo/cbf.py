import cvxpy as cp
import numpy as np

class PathPlanner:
    def __init__(self, alpha=5, beta=5):
        self.alpha = alpha
        self.beta = beta

    def compute_safe_path(self, start, goal, dynamic_objects):
        # Define optimization variables
        x = cp.Variable((3, 1))  # Drone position
        u = cp.Variable((3, 1))  # Control input

        # Define constraints
        constraints = []
        for obj in dynamic_objects:
            ellipsoid = create_ellipsoid(obj)
            constraints.append(batch_mahalanobis_distance(x, ellipsoid) >= 1)

        # Define cost function
        cost = cp.norm(u, 2) + self.alpha * cp.norm(x - goal, 2)

        # Solve QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        return x.value
    