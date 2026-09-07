"""Jog-and-capture registration of pick/place poses for auto_pick_place mode.

Lets an operator jog the robot by hand (via a normal UR5eTeleop instance) to the
target-object pose(s) and the goal-box pose, and capture each with a keypress.
Captured poses are applied to the current run immediately and printed in
cfg.yaml-ready form for the operator to paste in for future runs.
"""

import logging
import sys

from scipy.spatial.transform import Rotation as R

from scripts.core.record_loop import record_loop

if sys.platform == "win32":
    import msvcrt
else:
    import termios

logger = logging.getLogger(__name__)


def _flush_stdin() -> None:
    if sys.platform == "win32":
        while msvcrt.kbhit():
            msvcrt.getch()
    else:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def _capture_pose_euler(robot) -> list[float]:
    """Read the robot's current EE pose as [x, y, z, roll, pitch, yaw] radians."""
    ee_pose = robot.get_ee_pose()
    euler = R.from_rotvec(ee_pose[3:]).as_euler("xyz")
    return [*ee_pose[:3], *euler.tolist()]


def _format_pose(pose: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in pose) + "]"


def _jog_and_capture(
    record_cfg,
    robot,
    jog_teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    prompt: str,
) -> list[float]:
    logging.info("\033[32m%s (press the right arrow key to capture) ======\033[0m", prompt)
    events["exit_early"] = False
    jog_teleop.reset_step_size()
    record_loop(
        robot=robot,
        events=events,
        fps=record_cfg.fps,
        teleop=jog_teleop,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        control_time_s=record_cfg.auto_pick_place.registration_time_sec,
        single_task=record_cfg.task_description,
        display_data=record_cfg.display,
    )
    if events["stop_recording"]:
        raise RuntimeError("Pose registration cancelled (stop_recording).")
    events["exit_early"] = False
    return _capture_pose_euler(robot)


def register_pick_place_poses(
    record_cfg,
    robot,
    jog_teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
) -> None:
    """Jog to and capture pick pose(s) + a place pose, applying them to this run.

    Updates record_cfg.auto_pick_place.pick_poses/place_pose in place (so the
    controller picks them up on its next episode reset) and logs a cfg.yaml-ready
    summary for the operator to save for future runs.
    """
    logging.info("====== [REGISTER] Starting pick/place pose registration ======")
    jog_teleop.set_robot(robot)
    jog_teleop.connect()

    try:
        pick_poses: list[list[float]] = []
        while True:
            ordinal = len(pick_poses) + 1
            pose = _jog_and_capture(
                record_cfg,
                robot,
                jog_teleop,
                events,
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
                prompt=f"====== [REGISTER] Jog to pick pose #{ordinal} (target object)",
            )
            pick_poses.append(pose)
            logging.info("====== [REGISTER] Captured pick pose #%d: %s ======", ordinal, _format_pose(pose))
            _flush_stdin()
            answer = input("Add another pick pose? (y/n): ").strip().lower()
            if answer != "y":
                break

        place_pose = _jog_and_capture(
            record_cfg,
            robot,
            jog_teleop,
            events,
            teleop_action_processor,
            robot_action_processor,
            robot_observation_processor,
            prompt="====== [REGISTER] Jog to the place pose (goal box)",
        )
        logging.info("====== [REGISTER] Captured place pose: %s ======", _format_pose(place_pose))
    finally:
        jog_teleop.disconnect()

    record_cfg.auto_pick_place.pick_poses = pick_poses
    record_cfg.auto_pick_place.place_pose = place_pose

    _log_cfg_yaml_summary(pick_poses, place_pose)


def _log_cfg_yaml_summary(pick_poses: list[list[float]], place_pose: list[float]) -> None:
    lines = ["====== [REGISTER] Paste into cfg.yaml's record.auto_pick_place to reuse next time: ======"]
    if len(pick_poses) == 1:
        lines.append(f"  pick_pose: {_format_pose(pick_poses[0])}")
    else:
        lines.append("  pick_pose:")
        lines.extend(f"    - {_format_pose(pose)}" for pose in pick_poses)
    lines.append(f"  place_pose: {_format_pose(place_pose)}")
    logging.info("\n".join(lines))