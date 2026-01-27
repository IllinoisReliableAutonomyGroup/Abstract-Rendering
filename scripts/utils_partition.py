import os

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

def generate_partition(odd_type, part=None, fine_part=50):
    if odd_type == "cylinder":
        if part is None:
            part = [1,1,1]
        input_min = np.array([0.0, 0.0, 0.0])
        input_max = np.array([1.0, 1.0, 6.28])
        part = np.array(part)  # partitions per dimension

        # Step 1: compute edges per dimension
        edges = [
            np.linspace(input_min[d], input_max[d], part[d] + 1)
            for d in range(3)
        ]

        # Step 2: generate partitions
        centers = []
        lower_bounds = []
        upper_bounds = []

        for i in range(part[0]):
            for j in range(part[1]):
                for k in range(part[2]):
                    lb = np.array([
                        edges[0][i],
                        edges[1][j],
                        edges[2][k],
                    ])
                    ub = np.array([
                        edges[0][i]+(edges[0][i + 1]-edges[0][i])/fine_part,
                        edges[1][j]+(edges[1][j + 1]-edges[1][j])/fine_part,
                        edges[2][k]+(edges[2][k + 1]-edges[2][k])/fine_part,
                    ])
                    center = (lb + ub) * 0.5

                    lower_bounds.append(lb)
                    upper_bounds.append(ub)
                    centers.append(center)

        # Convert to arrays
        lower_bounds = np.stack(lower_bounds)   # (250, 3)
        upper_bounds = np.stack(upper_bounds)   # (250, 3)
        centers = np.stack(centers)             # (250, 3)

        return centers, lower_bounds, upper_bounds

if __name__ == '__main__':
    odd_type = "cylinder"
    part = [1,5,10]
    
    centers, lower_bounds, upper_bounds = partition(odd_type, part)
    print(lower_bounds.shape, upper_bounds.shape, centers.shape)
    # print(centers, lower_bounds, upper_bounds)