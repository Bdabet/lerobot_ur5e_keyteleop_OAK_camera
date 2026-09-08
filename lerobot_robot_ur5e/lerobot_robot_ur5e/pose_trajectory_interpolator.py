import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class PoseTrajectoryInterpolator:
    """Interpolates a trajectory of 6D poses ``[x, y, z, rx, ry, rz]`` (rotvec) over time.

    Position is linearly interpolated; orientation is spherically interpolated
    (Slerp). Sampling before the first or after the last waypoint holds the
    boundary pose.
    """

    def __init__(self, times: list[float], poses: list[list[float]]):
        times = np.asarray(times, dtype=float)
        poses = np.asarray(poses, dtype=float)
        assert len(times) == len(poses)
        assert len(times) > 0

        order = np.argsort(times)
        self.times = times[order]
        self.poses = poses[order]
        self._rotations = R.from_rotvec(self.poses[:, 3:])
        if len(self.times) >= 2:
            self._slerp = Slerp(self.times, self._rotations)
        else:
            self._slerp = None

    @property
    def start_time(self) -> float:
        return float(self.times[0])

    @property
    def end_time(self) -> float:
        return float(self.times[-1])

    def _position_at(self, t: float) -> np.ndarray:
        t_clamped = np.clip(t, self.times[0], self.times[-1])
        return np.array(
            [np.interp(t_clamped, self.times, self.poses[:, i]) for i in range(3)]
        )

    def _rotation_at(self, t: float) -> R:
        if self._slerp is None:
            return self._rotations[0]
        t_clamped = float(np.clip(t, self.times[0], self.times[-1]))
        return self._slerp([t_clamped])[0]

    def __call__(self, t: float) -> np.ndarray:
        position = self._position_at(t)
        rotation = self._rotation_at(t)
        pose = np.zeros(6, dtype=float)
        pose[:3] = position
        pose[3:] = rotation.as_rotvec()
        return pose

    def trim(self, curr_time: float) -> "PoseTrajectoryInterpolator":
        """Return a trajectory truncated to start at ``curr_time`` (sampling the
        pose there), keeping only waypoints at or after it."""
        curr_pose = self(curr_time)
        keep = self.times > curr_time
        new_times = [curr_time] + self.times[keep].tolist()
        new_poses = [curr_pose.tolist()] + self.poses[keep].tolist()
        return PoseTrajectoryInterpolator(times=new_times, poses=new_poses)

    def drive_to_waypoint(
        self,
        pose,
        time: float,
        curr_time: float,
        max_pos_speed: float = np.inf,
        max_rot_speed: float = np.inf,
    ) -> "PoseTrajectoryInterpolator":
        """Schedule ``pose`` as a new waypoint at ``time``, trimming the existing
        trajectory to ``curr_time`` first. If ``time`` is too soon to reach
        ``pose`` without exceeding the speed limits, it is pushed out to the
        minimum feasible arrival time."""
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        pose = np.asarray(pose, dtype=float)
        trimmed = self.trim(curr_time)
        start_pose = trimmed(curr_time)

        pos_dist = float(np.linalg.norm(pose[:3] - start_pose[:3]))
        rot_dist = float(
            (R.from_rotvec(pose[3:]) * R.from_rotvec(start_pose[3:]).inv()).magnitude()
        )
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(time - curr_time, pos_min_duration, rot_min_duration)
        time = curr_time + duration

        new_times = trimmed.times.tolist() + [time]
        new_poses = trimmed.poses.tolist() + [pose.tolist()]
        return PoseTrajectoryInterpolator(times=new_times, poses=new_poses)
