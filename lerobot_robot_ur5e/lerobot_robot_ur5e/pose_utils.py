import numpy as np
from scipy.spatial.transform import Rotation as R


def sample_pose_with_jitter(base_pose: list[float], jitter_range: list[float]) -> list[float]:
    """Sample a TCP pose near ``base_pose`` within ``jitter_range``.

    ``base_pose`` is ``[x, y, z, roll, pitch, yaw]`` in meters/radians (XYZ Euler).
    ``jitter_range`` is ``[x, y, z, roll_deg, pitch_deg, yaw_deg]``: the max random
    offset for position (meters, uniform) and orientation (degrees, uniform).
    Returns a pose as ``[x, y, z, rx, ry, rz]`` with orientation as a rotation vector,
    matching what RTDE's ``moveL``/``servoL`` expect.
    """
    if len(base_pose) != 6:
        raise ValueError(f"base_pose must contain 6 values, got {len(base_pose)}.")
    if len(jitter_range) != 6:
        raise ValueError(f"jitter_range must contain 6 values, got {len(jitter_range)}.")

    target_pose = np.array(base_pose, dtype=float)
    random_range = np.abs(np.array(jitter_range, dtype=float))
    target_pose[:3] += np.random.uniform(-random_range[:3], random_range[:3])

    delta_euler_deg = np.random.uniform(-random_range[3:], random_range[3:])
    target_euler = target_pose[3:] + np.deg2rad(delta_euler_deg)
    # base_pose uses XYZ Euler radians; jitter_range rotation uses degrees.
    # RTDE moveL/servoL expect a rotation vector.
    target_pose[3:] = R.from_euler("xyz", target_euler).as_rotvec()
    return target_pose.tolist()
