"""Scripted pick-and-place controller that stands in for a live teleoperator.

Exposes the same duck-typed surface `record_loop`/`run_record` expect from `teleop`
(`connect`, `disconnect`, `get_action`, `reset_step_size`, `set_robot`) plus a
`set_events` hook so it can signal episode completion itself, exactly like a human
pressing the "next episode" key.

Objects are not tied to a single fixed pick/place pose. Instead, `pick_region`
and `place_region` each define an x/y bounding box (plus a fixed orientation)
that positions are sampled from. Before the first episode, `object_loading.py`
drives this same controller through an interactive loading sequence (see
`build_wait_for_object_waypoints`/`build_retreat_after_manual_place_waypoints`/
`register_loaded_object`) that asks the operator how many objects there are;
for each one, the operator hands it to the arm and then manually jogs it down
onto the table wherever they like - the height it's released at becomes that
object's fixed height for the rest of the run.
"""

import logging
import random
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

logger = logging.getLogger(__name__)


class Region:
    """An x/y bounding box (meters, robot base frame) that pick/place
    positions are sampled uniformly from. Orientation is not configured here -
    every sampled pose uses whatever orientation the robot actually has at
    robot.init_pose (see AutoPickPlaceController.set_canonical_orientation_
    from_current_pose()), so it can never drift from init_pose."""

    def __init__(self, cfg: dict[str, Any], name: str):
        if "x_range" not in cfg or "y_range" not in cfg:
            raise ValueError(f"auto_pick_place.{name} requires x_range and y_range.")
        self.x_range: tuple[float, float] = tuple(float(v) for v in cfg["x_range"])
        self.y_range: tuple[float, float] = tuple(float(v) for v in cfg["y_range"])


class AutoPickPlaceConfig:
    def __init__(self, cfg: dict[str, Any]):
        self.pick_region = Region(cfg.get("pick_region", {}), "pick_region")
        self.place_region = Region(cfg.get("place_region", {}), "place_region")

        # How long (seconds) the arm holds at robot.init_pose with the gripper
        # turned on while the operator hands it each object during loading.
        self.object_load_hold_s: float = cfg.get("object_load_hold_s", 3.0)
        # Max time (seconds) the operator has to jog each object down onto the
        # table and release it during loading.
        self.manual_place_time_sec: float = cfg.get("manual_place_time_sec", 300)

        self.approach_height: float = cfg.get("approach_height", 0.10)
        self.max_step_size: float = cfg.get("max_step_size", 0.01)
        self.max_rot_step_size: float = cfg.get("max_rot_step_size", 0.02)
        # Seconds to ramp the commanded step from a standstill up to the full
        # max_step_size/max_rot_step_size cap (and back down), instead of
        # jumping straight to full speed the instant a new waypoint starts.
        # Smaller = snappier but rougher; larger = smoother but mushier.
        self.ramp_time_s: float = cfg.get("ramp_time_s", 0.15)
        self.position_tolerance: float = cfg.get("position_tolerance", 0.003)
        self.rotation_tolerance: float = cfg.get("rotation_tolerance", 0.02)
        self.grasp_close_wait_s: float = cfg.get("grasp_close_wait_s", 0.5)
        self.release_open_wait_s: float = cfg.get("release_open_wait_s", 0.3)

        # Stop the downward pick/place move early on TCP force contact instead
        # of servoing all the way to the (possibly optimistic) target depth.
        self.force_stop_enabled: bool = cfg.get("force_stop_enabled", False)
        self.contact_force_threshold: float = cfg.get("contact_force_threshold", 15.0)
        # Position streaming cap used only while descending on a force_stop
        # waypoint (pick/place), smaller than max_step_size so each tick's
        # overshoot into a rigid object stays small - contact is caught a few
        # millimeters in instead of at whatever distance max_step_size covers
        # in one control period, which is what was tripping the UR's own
        # protective stop (and dropping the RTDE connection) before contact
        # was even detected.
        self.force_stop_approach_step_size: float = cfg.get("force_stop_approach_step_size", 0.005)
        # When force_stop_enabled, right after grasp/release the TCP retreats
        # this many meters straight up from wherever contact actually
        # happened (not an absolute world height - object heights vary),
        # before continuing on to the full lift/retreat waypoint. Guarantees
        # the retreat always moves away from the surface it just touched.
        self.force_stop_clearance_height: float = cfg.get("force_stop_clearance_height", 0.02)

        self.shuffle_pick_order: bool = cfg.get("shuffle_pick_order", True)
        self.pick_min_separation: float = cfg.get("pick_min_separation", 0.05)
        self.place_min_separation: float = cfg.get("place_min_separation", 0.05)
        self.place_sampling_max_attempts: int = cfg.get("place_sampling_max_attempts", 200)

        # Human-like imperfections: overshooting the pick/place point before
        # settling, and bowing the transfer path instead of a straight line.
        self.overshoot_enabled: bool = cfg.get("overshoot_enabled", True)
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
    clearance_after_stop: bool = False # if True, gets re-targeted to actual-contact + clearance height


def _hover(pose_rotvec: list[float], height: float) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(pose_rotvec[:3], dtype=float) + np.array([0.0, 0.0, height])
    rotvec = np.array(pose_rotvec[3:], dtype=float)
    return position, rotvec


def _sample_region_position(region: Region, height: float, canonical_rotvec: np.ndarray) -> list[float]:
    """A random pose [x, y, z, rx, ry, rz] (rotvec) with x/y uniform over the
    region's bounding box, z fixed at height, and the canonical orientation
    (whatever the robot's actual orientation is at robot.init_pose)."""
    x = float(np.random.uniform(region.x_range[0], region.x_range[1]))
    y = float(np.random.uniform(region.y_range[0], region.y_range[1]))
    return [x, y, float(height), *canonical_rotvec.tolist()]


class AutoPickPlaceController:
    """Scripted pick-and-place teleop stand-in.

    Objects are loaded interactively (see object_loading.py) before the first
    episode: each object gets an index, a fixed height, and an initial position
    sampled inside pick_region. Every episode then picks each loaded object up
    from wherever it currently sits and places it at a fresh random spot inside
    place_region; the waypoint list is pre_pick -> pick -> grasp -> lift ->
    pre_place -> place -> release -> retreat, repeated once per object. Deltas
    are streamed closed-loop each tick (clipped to a max step size) toward the
    current waypoint.

    Between episodes, build_scene_reset_waypoints() builds a mirror-image
    sequence that picks each object back up from where it was just placed and
    sets it down at a fresh randomized spot inside pick_region, so the scene is
    ready for the next episode without human intervention.
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
        # Per-object fixed height, set once during loading (index == object id).
        self._object_heights: list[float] = []
        # object index -> its current pose [x,y,z,rx,ry,rz] (rotvec), updated
        # whenever a pick/place block moves it (loading, episode placement, or
        # scene reset), so the next episode/step picks it up from where it
        # really is instead of a fresh, independently-sampled guess.
        self._object_positions: dict[int, list[float]] = {}
        # (object_index, actual_place_pose) per object from the most recently
        # built episode, used by build_scene_reset_waypoints() to bring objects
        # back from their goal position to a fresh spot in pick_region.
        self._last_placements: list[tuple[int, list[float]]] = []
        # Previously commanded delta (position/rotation) and the wall-clock
        # time it was computed at, used to rate-limit how fast the commanded
        # step can change tick-to-tick (see _rate_limit_delta) instead of
        # jumping straight to max_step_size at the start of every waypoint.
        self._prev_delta_position = np.zeros(3)
        self._prev_delta_rotation = np.zeros(3)
        self._prev_tick_time: float | None = None
        # The single orientation (rotvec) used for every pick/place/loaded-
        # object pose, captured live from the robot at robot.init_pose (see
        # set_canonical_orientation_from_current_pose()) so it's always
        # exactly what init_pose actually is, never a separately configured
        # value that can drift from it.
        self._canonical_rotvec: np.ndarray | None = None

    def set_canonical_orientation_from_current_pose(self) -> None:
        """Capture the robot's current EE orientation as the fixed orientation
        used for every pick_region/place_region sample and every loaded
        object. Call this once, right after the robot has been moved to
        robot.init_pose, before object loading or any episode starts."""
        current_pose = self.robot.get_ee_pose()
        self._canonical_rotvec = np.array(current_pose[3:], dtype=float)

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

    def _reset_ramp(self) -> None:
        """Clear the acceleration-ramp state so the next commanded delta
        starts from a real standstill instead of inheriting velocity from
        whatever sequence just ended."""
        self._prev_delta_position = np.zeros(3)
        self._prev_delta_rotation = np.zeros(3)
        self._prev_tick_time = None

    def reset_step_size(self) -> None:
        """Per-episode reset hook (called once before each episode starts)."""
        self._build_episode_waypoints()
        self._current_idx = 0
        self._dwell_until = None
        self._episode_complete = False
        self._last_gripper = 0.0
        self._reset_ramp()

    def build_scene_reset_waypoints(self) -> bool:
        """Build a waypoint sequence that returns each object from where the last
        episode placed it to a freshly randomized position inside pick_region.

        Intended to be run through record_loop with dataset=None (not recorded as
        a demonstration) between episodes, so the scene is ready for the next
        episode's pick-and-place without human intervention. Returns False (no
        waypoints built) if there is no completed episode's placements to reset.
        """
        if not self._last_placements:
            return False

        waypoints: list[Waypoint] = []
        reset_positions: list[np.ndarray] = []
        for i, (object_idx, placed_pose) in enumerate(self._last_placements, start=1):
            height = self._object_heights[object_idx]
            reset_target = self._sample_region_pose(
                self.config.pick_region, height, reset_positions, self.config.pick_min_separation
            )
            reset_positions.append(np.array(reset_target[:3], dtype=float))
            self._object_positions[object_idx] = reset_target
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
        self._last_gripper = 0.0
        self._last_placements = []
        self._reset_ramp()
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
        # Axis-angle (rotvec) error, not an "xyz" Euler decomposition: for a
        # large correction (e.g. still far from a waypoint's target
        # orientation), decomposing into separate per-axis Euler angles can
        # distort badly and inject rotation around an axis that isn't
        # actually part of the true, minimal correction - visible as the arm
        # spinning around an axis it has no business moving on. rotvec gives
        # the true minimal-angle correction regardless of orientation.
        rot_error_vec = (R.from_rotvec(waypoint.rotvec) * current_rotation.inv()).as_rotvec()

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

        step_limit = self.config.max_step_size
        if waypoint.force_stop and self.config.force_stop_enabled:
            step_limit = min(step_limit, self.config.force_stop_approach_step_size)
        delta_position = _clip_to_norm(pos_error, step_limit)
        delta_rotation = _clip_to_norm(rot_error_vec, self.config.max_rot_step_size)
        delta_position, delta_rotation = self._rate_limit_delta(
            delta_position, delta_rotation, step_limit, self.config.max_rot_step_size
        )

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

    # ======= object loading (see scripts/core/object_loading.py) =======

    def build_wait_for_object_waypoints(self, hold_s: float) -> None:
        """A single waypoint that holds the robot's current pose with the
        gripper turned on (suction active) for hold_s seconds, so the operator
        can hand the arm an object to load.

        Call this only once the robot is already stationary at init_pose (via
        a direct robot.reset_to_init_pose() call, not through this
        controller). The target is read straight from get_ee_pose() rather
        than recomputed from robot.init_pose, because init_pose is specified
        in the TCP frame (as consumed by moveL/reset_to_init_pose) while
        get_ee_pose() - what every waypoint here is tracked against - returns
        the TCP pose with the tool's TCP offset removed. Rebuilding the target
        from init_pose would create a phantom error equal to that offset and
        drive the arm to "correct" it instead of holding still.
        """
        current_pose = self.robot.get_ee_pose()
        position = np.array(current_pose[:3], dtype=float)
        rotvec = np.array(current_pose[3:], dtype=float)
        self._waypoints = [Waypoint("load_wait", position, rotvec, gripper=1.0, dwell_s=hold_s)]
        self._current_idx = 0
        self._dwell_until = None
        self._episode_complete = False
        # The target *is* the current pose by construction, so there is no
        # real approach phase to wait out - turn suction on immediately
        # instead of deferring to the first "within tolerance" tick.
        self._last_gripper = 1.0
        self._reset_ramp()

    def canonicalize_pick_pose(self, placed_pose: list[float]) -> list[float]:
        """Replace a manually-placed pose's orientation with the canonical
        orientation (see set_canonical_orientation_from_current_pose()),
        keeping only its x/y/z.

        Manual jogging can leave the wrist rotated by whatever incidental
        amount the operator happened to approach at, even if they never meant
        to reorient it. Every later pick (every automatic scene reset)
        samples a fresh position but always uses the canonical orientation -
        so if a loaded object's registered pose kept the raw jogged rotation
        instead, the very first pick of that object would need a real (and
        often large) reorientation to align with it that no other pick ever
        needs. Canonicalizing here makes every pick, from the first one on,
        use the same orientation - which is exactly the orientation the robot
        already has at robot.init_pose.
        """
        if self._canonical_rotvec is None:
            raise ValueError(
                "AutoPickPlaceController has no canonical orientation. "
                "set_canonical_orientation_from_current_pose() must run first."
            )
        position = np.array(placed_pose[:3], dtype=float)
        return [*position.tolist(), *self._canonical_rotvec.tolist()]

    def build_retreat_after_manual_place_waypoints(self, placed_pose: list[float]) -> None:
        """Explicitly release the gripper at a manually-placed object's pose
        (rotvec), then lift straight up, so the object is guaranteed to be let
        go before the arm moves away - not left dependent on the operator
        having manually toggled the gripper open during jogging."""
        position = np.array(placed_pose[:3], dtype=float)
        rotvec = np.array(placed_pose[3:], dtype=float)
        hover_position, hover_rotvec = _hover(placed_pose, self.config.approach_height)
        self._waypoints = [
            Waypoint(
                "load_manual_release",
                position,
                rotvec,
                gripper=0.0,
                dwell_s=self.config.release_open_wait_s,
            ),
            Waypoint("load_manual_retreat", hover_position, hover_rotvec, gripper=None),
        ]
        self._current_idx = 0
        self._dwell_until = None
        self._episode_complete = False
        self._reset_ramp()

    def register_loaded_object(self, placed_pose: list[float], height: float) -> None:
        """Record a newly-loaded object's fixed height and current position so
        episodes can pick it up. Called once per object, after the operator
        has manually placed it and released the gripper."""
        object_idx = len(self._object_heights)
        self._object_heights.append(height)
        self._object_positions[object_idx] = placed_pose

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
        instead of resuming its descent toward the original target depth. The
        waypoint right after that (the post-grasp/release clearance step, if
        any) keeps the real contact x/y but is re-targeted to clearance height
        straight up from the real contact z, so the retreat always moves away
        from the surface it just touched instead of a fixed world height that
        may be below it."""
        original_target = self._waypoints[from_idx].position
        idx = from_idx
        while idx < len(self._waypoints) and np.allclose(self._waypoints[idx].position, original_target):
            self._waypoints[idx] = replace(self._waypoints[idx], position=frozen_position)
            idx += 1
        if idx < len(self._waypoints) and self._waypoints[idx].clearance_after_stop:
            clearance_position = frozen_position + np.array([0.0, 0.0, self.config.force_stop_clearance_height])
            self._waypoints[idx] = replace(self._waypoints[idx], position=clearance_position)

    def _signal_done(self) -> None:
        if self.events is not None:
            self.events["exit_early"] = True

    def _zero_action(self) -> dict[str, float]:
        # Any tick that returns a real zero (episode done, dwelling at a
        # waypoint) is a genuine standstill - reset the ramp so the next real
        # move starts accelerating from 0 instead of resuming the velocity
        # last commanded before this pause.
        self._reset_ramp()
        action = {key: 0.0 for key in ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz")}
        if self.use_gripper:
            action["gripper_position"] = self._last_gripper
        return action

    def _rate_limit_delta(
        self,
        delta_position: np.ndarray,
        delta_rotation: np.ndarray,
        step_limit: float,
        rot_step_limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clamp how much the commanded delta can change since the previous
        tick, so a waypoint transition ramps up to step_limit/rot_step_limit
        over config.ramp_time_s instead of jumping straight to full speed.
        The first tick after a reset (self._prev_tick_time is None) is passed
        through unlimited, since there is no previous tick to ramp from."""
        now = time.perf_counter()
        if self._prev_tick_time is None:
            dt = None
        else:
            dt = now - self._prev_tick_time
        self._prev_tick_time = now

        if dt is not None:
            max_pos_change = step_limit * dt / self.config.ramp_time_s
            delta_position = self._prev_delta_position + _clip_to_norm(
                delta_position - self._prev_delta_position, max_pos_change
            )
            max_rot_change = rot_step_limit * dt / self.config.ramp_time_s
            delta_rotation = self._prev_delta_rotation + _clip_to_norm(
                delta_rotation - self._prev_delta_rotation, max_rot_change
            )

        self._prev_delta_position = delta_position
        self._prev_delta_rotation = delta_rotation
        return delta_position, delta_rotation

    def _build_episode_waypoints(self) -> None:
        if not self._object_heights:
            raise ValueError(
                "AutoPickPlaceController has no loaded objects. "
                "load_objects_into_regions() must run before the first episode."
            )

        object_order = list(range(len(self._object_heights)))
        if self.config.shuffle_pick_order:
            random.shuffle(object_order)

        picked_positions: list[np.ndarray] = []
        placed_positions: list[np.ndarray] = []
        placements: list[tuple[int, list[float]]] = []
        waypoints: list[Waypoint] = []

        for i, object_idx in enumerate(object_order, start=1):
            # Objects sit exactly where loading (or the previous episode's
            # reset) left them - pick them up from there.
            pick_pose = self._object_positions[object_idx]
            picked_positions.append(np.array(pick_pose[:3], dtype=float))

            height = self._object_heights[object_idx]
            place_sampled = self._sample_region_pose(
                self.config.place_region, height, placed_positions, self.config.place_min_separation
            )
            placed_positions.append(np.array(place_sampled[:3], dtype=float))
            placements.append((object_idx, place_sampled))
            self._object_positions[object_idx] = place_sampled

            waypoints.extend(self._pick_place_block(i, pick_pose, place_sampled))
            logger.info(
                "====== [AUTO] Object %d/%d: pick=%s place=%s ======",
                i,
                len(object_order),
                [round(v, 4) for v in pick_pose],
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
            Waypoint(name("pre_pick"), source_hover_position, source_hover_rotvec, gripper=0.0),
        ]
        if self.config.overshoot_enabled and self.config.pick_overshoot_distance > 0.0:
            overshoot_position = _sample_overshoot_position(source_position, self.config.pick_overshoot_distance)
            waypoints.append(Waypoint(name("overshoot_pick"), overshoot_position, source_rotvec, gripper=0.0))
        waypoints.append(Waypoint(name("pick"), source_position, source_rotvec, gripper=0.0, force_stop=True))
        waypoints.append(
            Waypoint(
                name("grasp"),
                source_position,
                source_rotvec,
                gripper=1.0,
                dwell_s=self.config.grasp_close_wait_s,
            )
        )
        if self.config.force_stop_enabled:
            clearance_position = source_position + np.array([0.0, 0.0, self.config.force_stop_clearance_height])
            waypoints.append(
                Waypoint(
                    name("post_grasp_clear"),
                    clearance_position,
                    source_rotvec,
                    gripper=None,
                    clearance_after_stop=True,
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
        waypoints.append(
            Waypoint(name("place"), target_position, target_rotvec, gripper=None, force_stop=True)
        )
        waypoints.append(
            Waypoint(
                name("release"),
                target_position,
                target_rotvec,
                gripper=0.0,
                dwell_s=self.config.release_open_wait_s,
            )
        )
        if self.config.force_stop_enabled:
            clearance_position = target_position + np.array([0.0, 0.0, self.config.force_stop_clearance_height])
            waypoints.append(
                Waypoint(
                    name("post_release_clear"),
                    clearance_position,
                    target_rotvec,
                    gripper=None,
                    clearance_after_stop=True,
                )
            )
        waypoints.append(Waypoint(name("retreat"), target_hover_position, target_hover_rotvec, gripper=None))
        return waypoints

    def _sample_region_pose(
        self,
        region: Region,
        height: float,
        existing_positions: list[np.ndarray],
        min_separation: float,
    ) -> list[float]:
        """Sample a pose uniformly inside region at the given height, rejecting
        candidates too close to positions already placed earlier."""
        if self._canonical_rotvec is None:
            raise ValueError(
                "AutoPickPlaceController has no canonical orientation. "
                "set_canonical_orientation_from_current_pose() must run before sampling."
            )
        candidate = _sample_region_position(region, height, self._canonical_rotvec)
        for _ in range(self.config.place_sampling_max_attempts):
            candidate_position = np.array(candidate[:3], dtype=float)
            if all(
                np.linalg.norm(candidate_position - prev) >= min_separation
                for prev in existing_positions
            ):
                return candidate
            candidate = _sample_region_position(region, height, self._canonical_rotvec)

        logger.warning(
            "====== [AUTO] Could not find a position >= %.3fm from prior placements "
            "after %d attempts; using closest candidate found. ======",
            min_separation,
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
    offset = np.array(
        [magnitude * np.cos(angle), magnitude * np.sin(angle), np.random.uniform(0.03, 0.04)]
    )
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
