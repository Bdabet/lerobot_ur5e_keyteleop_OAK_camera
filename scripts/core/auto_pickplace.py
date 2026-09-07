"""Scripted pick-and-place controller that stands in for a live teleoperator.

Exposes the same duck-typed surface `record_loop`/`run_record` expect from `teleop`
(`connect`, `disconnect`, `get_action`, `reset_step_size`, `set_robot`) plus a
`set_events` hook so it can signal episode completion itself, exactly like a human
pressing the "next episode" key.
"""

import logging
import random
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from lerobot_robot_ur5e import sample_pose_with_jitter

logger = logging.getLogger(__name__)


class AutoPickPlaceConfig:
    def __init__(self, cfg: dict[str, Any]):
        # When register_on_start is true, pick_pose/place_pose are captured by jogging
        # the robot at the start of the run (see scripts/core/pose_registration.py)
        # instead of being required up front.
        self.register_on_start: bool = cfg.get("register_on_start", False)
        self.registration_time_sec: float = cfg.get("registration_time_sec", 300)

        pick_pose = cfg.get("pick_pose", [])
        if not pick_pose:
            if not self.register_on_start:
                raise ValueError(
                    "auto_pick_place.pick_pose is required unless register_on_start is true."
                )
            self.pick_poses: list[list[float]] = []
        else:
            # Accept either a single [x,y,z,roll,pitch,yaw] pose or a list of them.
            self.pick_poses = [pick_pose] if isinstance(pick_pose[0], (int, float)) else list(pick_pose)
        self.pick_pose_range: list[float] = cfg.get("pick_pose_range", [0.0] * 6)

        place_pose = cfg.get("place_pose")
        if place_pose is None and not self.register_on_start:
            raise ValueError("auto_pick_place.place_pose is required unless register_on_start is true.")
        self.place_pose: list[float] | None = place_pose
        self.place_pose_range: list[float] = cfg.get("place_pose_range", [0.0] * 6)

        self.approach_height: float = cfg.get("approach_height", 0.10)
        self.max_step_size: float = cfg.get("max_step_size", 0.01)
        self.max_rot_step_size: float = cfg.get("max_rot_step_size", 0.02)
        self.position_tolerance: float = cfg.get("position_tolerance", 0.003)
        self.rotation_tolerance: float = cfg.get("rotation_tolerance", 0.02)
        self.grasp_close_wait_s: float = cfg.get("grasp_close_wait_s", 0.5)
        self.release_open_wait_s: float = cfg.get("release_open_wait_s", 0.3)

        # Stop the downward pick move early on TCP force contact instead of
        # servoing all the way to the (possibly optimistic) pick_pose depth.
        self.force_stop_enabled: bool = cfg.get("force_stop_enabled", False)
        self.contact_force_threshold: float = cfg.get("contact_force_threshold", 15.0)

        self.shuffle_pick_order: bool = cfg.get("shuffle_pick_order", True)
        self.place_min_separation: float = cfg.get("place_min_separation", 0.05)
        self.place_sampling_max_attempts: int = cfg.get("place_sampling_max_attempts", 200)

        # Human-like imperfections: overshooting the pick/place point before
        # settling, and bowing the transfer path instead of a straight line.
        self.overshoot_enabled: bool = cfg.get("overshoot_enabled", False)
        self.pick_overshoot_distance: float = cfg.get("pick_overshoot_distance", 0.0)
        self.place_overshoot_distance: float = cfg.get("place_overshoot_distance", 0.0)
        self.trajectory_curve_enabled: bool = cfg.get("trajectory_curve_enabled", False)
        self.trajectory_bow_range: float = cfg.get("trajectory_bow_range", 0.0)
        self.trajectory_waypoints: int = cfg.get("trajectory_waypoints", 3)


@dataclass
class Waypoint:
    name: str
    position: np.ndarray
    rotvec: np.ndarray
    gripper: float | None
    dwell_s: float = 0.0 # pause at waypoint before moving to the next one (seconds)
    force_stop: bool = False # if True, an early TCP-force contact ends this waypoint


def _hover(pose_rotvec: list[float], height: float) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(pose_rotvec[:3], dtype=float) + np.array([0.0, 0.0, height])
    rotvec = np.array(pose_rotvec[3:], dtype=float)
    return position, rotvec


class AutoPickPlaceController:
    """Scripted pick-and-place teleop stand-in.

    Every configured pick_pose is picked and placed within a single episode: the
    waypoint list is pre_pick -> pick -> grasp -> lift -> pre_place -> place ->
    release -> retreat, repeated once per object. Deltas are streamed closed-loop
    each tick (clipped to a max step size) toward the current waypoint.

    Between episodes, build_scene_reset_waypoints() builds a mirror-image
    sequence that picks each object back up from where it was just placed and
    sets it down at a fresh randomized spot in its own pick region, so the scene
    is ready for the next episode without human intervention.
    """

    name = "auto_pick_place"

    def __init__(self, config: AutoPickPlaceConfig, use_gripper: bool):
        self.config = config
        self.use_gripper = use_gripper
        self.robot = None
        self.events: dict | None = None
        self._connected = False

        self._waypoints: list[Waypoint] = []
        self._current_idx = 0
        self._dwell_until: float | None = None
        self._episode_complete = False
        self._last_gripper = 1.0
        # (home_pick_pose, actual_place_pose) per object from the most recently
        # built episode, used by build_scene_reset_waypoints() to bring objects
        # back from their goal position to a fresh spot in their pick region.
        self._last_placements: list[tuple[list[float], list[float]]] = []

    # ======= duck-typed teleop interface =======

    def set_robot(self, robot) -> None:
        self.robot = robot

    def set_events(self, events: dict) -> None:
        self.events = events

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def current_waypoint(self) -> "Waypoint | None":
        if self._episode_complete or not self._waypoints:
            return None
        return self._waypoints[self._current_idx]

    @property
    def episode_complete(self) -> bool:
        return self._episode_complete

    def reset_step_size(self) -> None:
        """Per-episode reset hook (called once before each episode starts)."""
        self._build_episode_waypoints()
        self._current_idx = 0
        self._dwell_until = None
        self._episode_complete = False
        self._last_gripper = 1.0

    def build_scene_reset_waypoints(self) -> bool:
        """Build a waypoint sequence that returns each object from where the last
        episode placed it to a freshly randomized position in its own pick region.

        Intended to be run through record_loop with dataset=None (not recorded as
        a demonstration) between episodes, so the scene is ready for the next
        episode's pick-and-place without human intervention. Returns False (no
        waypoints built) if there is no completed episode's placements to reset.
        """
        if not self._last_placements:
            return False

        waypoints: list[Waypoint] = []
        for i, (home_pick_pose, placed_pose) in enumerate(self._last_placements, start=1):
            reset_target = sample_pose_with_jitter(home_pick_pose, self.config.pick_pose_range)
            waypoints.extend(self._pick_place_block(i, placed_pose, reset_target, name_prefix="reset_"))
            logger.info(
                "====== [AUTO] Reset %d/%d: from=%s to=%s ======",
                i,
                len(self._last_placements),
                [round(v, 4) for v in placed_pose],
                [round(v, 4) for v in reset_target],
            )

        self._waypoints = waypoints
        self._current_idx = 0
        self._dwell_until = None
        self._episode_complete = False
        self._last_gripper = 1.0
        self._last_placements = []
        return True

    def get_action(self) -> dict[str, float]:
        if self.robot is None:
            raise ValueError("AutoPickPlaceController requires set_robot() before get_action().")

        if self._episode_complete:
            self._signal_done()
            return self._zero_action()

        current_pose = self.robot.get_ee_pose()
        current_position = np.array(current_pose[:3], dtype=float)
        current_rotation = R.from_rotvec(current_pose[3:])

        waypoint = self._waypoints[self._current_idx]

        if waypoint.force_stop and self.config.force_stop_enabled and self._contact_detected():
            self._freeze_contact_position(self._current_idx, current_position)
            waypoint = self._waypoints[self._current_idx]

        pos_error = waypoint.position - current_position
        rot_error_vec = (R.from_rotvec(waypoint.rotvec) * current_rotation.inv()).as_euler("xyz")

        pos_error_norm = float(np.linalg.norm(pos_error))
        rot_error_norm = float(np.linalg.norm(rot_error_vec))
        within_tolerance = (
            pos_error_norm <= self.config.position_tolerance
            and rot_error_norm <= self.config.rotation_tolerance
        )

        if within_tolerance:
            if waypoint.gripper is not None:
                self._last_gripper = waypoint.gripper
            if self._dwell_until is None:
                self._dwell_until = time.perf_counter() + waypoint.dwell_s
            if time.perf_counter() >= self._dwell_until:
                self._dwell_until = None
                if self._current_idx == len(self._waypoints) - 1:
                    self._episode_complete = True
                    self._signal_done()
                else:
                    self._current_idx += 1
            return self._zero_action()

        delta_position = _clip_to_norm(pos_error, self.config.max_step_size)
        delta_rotation = _clip_to_norm(rot_error_vec, self.config.max_rot_step_size)

        action = {
            "delta_x": float(delta_position[0]),
            "delta_y": float(delta_position[1]),
            "delta_z": float(delta_position[2]),
            "delta_rx": float(delta_rotation[0]),
            "delta_ry": float(delta_rotation[1]),
            "delta_rz": float(delta_rotation[2]),
        }
        if self.use_gripper:
            action["gripper_position"] = self._last_gripper
        return action

    # ======= internals =======

    def _contact_detected(self) -> bool:
        """True if live TCP force exceeds contact_force_threshold, meaning the
        gripper has touched an object/surface during a force_stop waypoint."""
        tcp_force = self.robot.get_tcp_force()
        force_norm = float(np.linalg.norm(tcp_force[:3]))
        if force_norm >= self.config.contact_force_threshold:
            logger.info(
                "====== [AUTO] Contact detected at waypoint '%s': force=%.2fN >= threshold=%.2fN ======",
                self._waypoints[self._current_idx].name,
                force_norm,
                self.config.contact_force_threshold,
            )
            return True
        return False

    def _freeze_contact_position(self, from_idx: int, frozen_position: np.ndarray) -> None:
        """Rewrite from_idx and any immediately following waypoints that target
        the same point (e.g. the grasp/release dwell after pick/place) to the
        actual position where contact was detected, so the robot stops there
        instead of resuming its descent toward the original target depth."""
        original_target = self._waypoints[from_idx].position
        idx = from_idx
        while idx < len(self._waypoints) and np.allclose(self._waypoints[idx].position, original_target):
            self._waypoints[idx] = replace(self._waypoints[idx], position=frozen_position)
            idx += 1

    def _signal_done(self) -> None:
        if self.events is not None:
            self.events["exit_early"] = True

    def _zero_action(self) -> dict[str, float]:
        action = {key: 0.0 for key in ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz")}
        if self.use_gripper:
            action["gripper_position"] = self._last_gripper
        return action

    def _build_episode_waypoints(self) -> None:
        if not self.config.pick_poses or self.config.place_pose is None:
            raise ValueError(
                "AutoPickPlaceController has no pick/place poses. If "
                "register_on_start is true, pose registration must run before "
                "the first episode."
            )

        pick_order = list(self.config.pick_poses)
        if self.config.shuffle_pick_order:
            random.shuffle(pick_order)

        placed_positions: list[np.ndarray] = []
        placements: list[tuple[list[float], list[float]]] = []
        waypoints: list[Waypoint] = []

        for i, pick_pose in enumerate(pick_order, start=1):
            pick_sampled = sample_pose_with_jitter(pick_pose, self.config.pick_pose_range)
            place_sampled = self._sample_place_pose(placed_positions)
            placed_positions.append(np.array(place_sampled[:3], dtype=float))
            placements.append((pick_pose, place_sampled))

            waypoints.extend(self._pick_place_block(i, pick_sampled, place_sampled))
            logger.info(
                "====== [AUTO] Object %d/%d: pick=%s place=%s ======",
                i,
                len(pick_order),
                [round(v, 4) for v in pick_sampled],
                [round(v, 4) for v in place_sampled],
            )

        self._waypoints = waypoints
        self._last_placements = placements

    def _pick_place_block(
        self,
        index: int,
        source_pose: list[float],
        target_pose: list[float],
        name_prefix: str = "",
    ) -> list[Waypoint]:
        """A pre_pick -> pick -> grasp -> lift -> pre_place -> place -> release ->
        retreat waypoint block moving an object from source_pose to target_pose."""
        source_position = np.array(source_pose[:3], dtype=float)
        source_rotvec = np.array(source_pose[3:], dtype=float)
        target_position = np.array(target_pose[:3], dtype=float)
        target_rotvec = np.array(target_pose[3:], dtype=float)
        source_hover_position, source_hover_rotvec = _hover(source_pose, self.config.approach_height)
        target_hover_position, target_hover_rotvec = _hover(target_pose, self.config.approach_height)

        def name(step: str) -> str:
            return f"{name_prefix}{step}_{index}"

        waypoints = [
            Waypoint(name("pre_pick"), source_hover_position, source_hover_rotvec, gripper=1.0),
        ]
        if self.config.overshoot_enabled and self.config.pick_overshoot_distance > 0.0:
            overshoot_position = _sample_overshoot_position(source_position, self.config.pick_overshoot_distance)
            waypoints.append(Waypoint(name("overshoot_pick"), overshoot_position, source_rotvec, gripper=1.0))
        waypoints.append(Waypoint(name("pick"), source_position, source_rotvec, gripper=1.0, force_stop=True))
        waypoints.append(
            Waypoint(
                name("grasp"),
                source_position,
                source_rotvec,
                gripper=0.0,
                dwell_s=self.config.grasp_close_wait_s,
            )
        )
        waypoints.append(Waypoint(name("lift"), source_hover_position, source_hover_rotvec, gripper=None))
        if self.config.trajectory_curve_enabled and self.config.trajectory_bow_range > 0.0:
            waypoints.extend(
                _bow_bezier_waypoints(
                    source_hover_position,
                    source_hover_rotvec,
                    target_hover_position,
                    target_hover_rotvec,
                    self.config.trajectory_bow_range,
                    self.config.trajectory_waypoints,
                    name_prefix=f"{name_prefix}transfer_{index}_",
                )
            )
        waypoints.append(Waypoint(name("pre_place"), target_hover_position, target_hover_rotvec, gripper=None))
        if self.config.overshoot_enabled and self.config.place_overshoot_distance > 0.0:
            overshoot_position = _sample_overshoot_position(target_position, self.config.place_overshoot_distance)
            waypoints.append(Waypoint(name("overshoot_place"), overshoot_position, target_rotvec, gripper=None))
        waypoints.append(Waypoint(name("place"), target_position, target_rotvec, gripper=None))
        waypoints.append(
            Waypoint(
                name("release"),
                target_position,
                target_rotvec,
                gripper=1.0,
                dwell_s=self.config.release_open_wait_s,
            )
        )
        waypoints.append(Waypoint(name("retreat"), target_hover_position, target_hover_rotvec, gripper=None))
        return waypoints

    def _sample_place_pose(self, placed_positions: list[np.ndarray]) -> list[float]:
        """Sample a place pose within place_pose +/- place_pose_range, rejecting
        candidates too close to positions already placed earlier this episode."""
        candidate = sample_pose_with_jitter(self.config.place_pose, self.config.place_pose_range)
        for _ in range(self.config.place_sampling_max_attempts):
            candidate_position = np.array(candidate[:3], dtype=float)
            if all(
                np.linalg.norm(candidate_position - prev) >= self.config.place_min_separation
                for prev in placed_positions
            ):
                return candidate
            candidate = sample_pose_with_jitter(self.config.place_pose, self.config.place_pose_range)

        logger.warning(
            "====== [AUTO] Could not find a place position >= %.3fm from prior placements "
            "after %d attempts; using closest candidate found. ======",
            self.config.place_min_separation,
            self.config.place_sampling_max_attempts,
        )
        return candidate


def _clip_to_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)


def _sample_overshoot_position(base_position: np.ndarray, max_distance: float) -> np.ndarray:
    """A point near base_position, offset sideways by a random amount up to
    max_distance and raised slightly (never lowered, to avoid overshooting
    down into the table/object) to mimic a human reaching a bit too far
    before correcting back to the true pick/place point."""
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    magnitude = np.random.uniform(0.0, max_distance)
    offset = np.array([magnitude * np.cos(angle), magnitude * np.sin(angle), magnitude * 0.2])
    return base_position + offset


def _bow_bezier_waypoints(
    start_position: np.ndarray,
    start_rotvec: np.ndarray,
    end_position: np.ndarray,
    end_rotvec: np.ndarray,
    bow_range: float,
    num_points: int,
    name_prefix: str,
) -> list["Waypoint"]:
    """Intermediate waypoints along a quadratic Bezier curve from start to end,
    bowed sideways/up by a random amount, so the transfer move reads as an
    indirect human reach instead of a straight line. Orientation is slerped
    between the two endpoints."""
    if num_points <= 0:
        return []

    travel = end_position - start_position
    travel_norm = float(np.linalg.norm(travel))
    if travel_norm == 0.0:
        perpendicular = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 0.0, 1.0]) if abs(travel[2]) < travel_norm * 0.9 else np.array([1.0, 0.0, 0.0])
        perpendicular = np.cross(travel, arbitrary)
        perpendicular /= np.linalg.norm(perpendicular)

    bow_magnitude = np.random.uniform(-bow_range, bow_range)
    control_point = (start_position + end_position) / 2.0 + perpendicular * bow_magnitude
    control_point[2] += abs(np.random.uniform(0.0, bow_range))

    rotations = R.from_rotvec([start_rotvec, end_rotvec])
    slerp = Slerp([0.0, 1.0], rotations)

    waypoints = []
    for i in range(1, num_points + 1):
        t = i / (num_points + 1)
        position = (1 - t) ** 2 * start_position + 2 * (1 - t) * t * control_point + t**2 * end_position
        waypoints.append(
            Waypoint(f"{name_prefix}{i}", position, slerp([t])[0].as_rotvec(), gripper=None)
        )
    return waypoints