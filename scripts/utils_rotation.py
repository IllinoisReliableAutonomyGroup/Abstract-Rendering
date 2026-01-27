import numpy as np
import torch

from scipy.spatial.transform import Rotation 

def quaternion_to_rotation_matrix(quats):
    """
    Converts quaternions to rotation matrices.
    Args:
        quats: Tensor of shape [N, 4], where each quaternion is [w, x, y, z].
    Returns:
        Rotation matrices of shape [N, 3, 3].
    """
    quats = quats / quats.norm(dim=1, keepdim=True)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    N = quats.size(0)
    R = torch.empty(N, 3, 3, device=quats.device, dtype=quats.dtype)

    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)

    return R 

### Convert between direction vector and yaw, pitch, roll
def direction_to_orientation(dx, dy, dz):
    # Normalize forward (world Y-forward convention)
    forward = np.array([dx, dy, dz], dtype=float)
    forward /= np.linalg.norm(forward)

    # World up
    up = np.array([0.0, 0.0, 1.0])

    # Right = forward × up  (IMPORTANT ORDER)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # Recompute orthogonal up
    up_corrected = np.cross(right, forward)

    # Construct rotation matrix
    # Columns = local axes expressed in world frame
    # X_local = right
    # Y_local = forward
    # Z_local = up
    R_matrix = np.stack([right, forward, up_corrected], axis=1)

    # Extract yaw, pitch, roll
    rot = Rotation.from_matrix(R_matrix)
    yaw, pitch, roll = rot.as_euler("ZYX")
    # yaw = 0 means looking along +X axis
    yaw += np.pi / 2

    return yaw, pitch, roll

def orientation_to_direction(yaw, pitch, roll):
    """
    Convert yaw, pitch, roll (ZYX, radians) to a forward direction vector (dx, dy, dz)
    World frame convention:
        X → right
        Y → forward
        Z → up
    """
    # Undo the previous +pi/2 yaw shift
    yaw -= np.pi / 2

    # Construct rotation
    rot = Rotation.from_euler("ZYX", [yaw, pitch, roll])

    # Rotation matrix
    R_matrix = rot.as_matrix()

    # Forward vector is local Y axis in world coordinates
    forward = R_matrix[:, 1]

    # Normalize
    forward /= np.linalg.norm(forward)

    dx, dy, dz = forward
    return dx, dy, dz


if __name__ == '__main__':
    pass