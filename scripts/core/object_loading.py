"""Interactive object loading for auto_pick_place mode.

Before episodes start, the operator is asked how many objects will be used.
For each one: the arm holds at robot.init_pose with the gripper turned on for
auto_pick_place.object_load_hold_s seconds so the operator can hand it the
object, then control is handed to a manual jog teleop so the operator can
move the (suctioned) object down onto the table wherever they like; pressing
the right arrow key (the same "confirm" convention used elsewhere in this
codebase) ends the jog. The gripper is then released by a scripted waypoint
(not left to the operator to remember to do themselves) before the arm lifts
away. The object's actual height at that point becomes its fixed height for
the rest of the run, and where it was released becomes its starting position
in pick_region.

Uses the AutoPickPlaceController itself (already connected/set_robot()) to
drive the scripted moves (hold, retreat) through record_loop, exactly like
build_scene_reset_waypoints() does between episodes - not recorded as
demonstration data (dataset=None).
"""

import logging
import sys

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


def _ask_num_objects() -> int:
    while True:
        _flush_stdin()
        answer = input("====== [LOAD] How many objects will you load? ").strip()
        try:
            num_objects = int(answer)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if num_objects <= 0:
            print("Please enter a number greater than 0.")
            continue
        return num_objects


def _run_step(
    record_cfg,
    robot,
    teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    control_time_s: float | None = None,
) -> None:
    events["exit_early"] = False
    record_loop(
        robot=robot,
        events=events,
        fps=record_cfg.fps,
        teleop=teleop,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        control_time_s=control_time_s if control_time_s is not None else record_cfg.episode_time_sec,
        display_data=record_cfg.display,
    )
    if events["stop_recording"]:
        raise RuntimeError("Object loading cancelled (stop_recording).")
    events["exit_early"] = False


def load_objects_into_regions(
    record_cfg,
    robot,
    teleop,
    jog_teleop,
    events: dict,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
) -> None:
    """Run the interactive per-object loading sequence described above.

    Populates the controller (teleop) with each loaded object's fixed height
    and current position via register_loaded_object(), so the first episode's
    reset_step_size() has objects to pick from pick_region.
    """
    logging.info("====== [LOAD] Starting object loading ======")
    num_objects = _ask_num_objects()

    jog_teleop.set_robot(robot)
    jog_teleop.connect()

    try:
        for i in range(1, num_objects + 1):
            logging.info(
                "\033[32m====== [LOAD] Object %d/%d: moving to init pose, gripper on for %.1fs ======\033[0m",
                i,
                num_objects,
                record_cfg.auto_pick_place.object_load_hold_s,
            )
            # Direct blocking moveL, same call used everywhere else to reach
            # init_pose - init_pose is a TCP-frame pose, so it must be reached
            # this way rather than streamed through the controller (which
            # tracks waypoints in the TCP-offset-removed EE frame).
            robot.reset_to_init_pose(record_cfg.init_pose, record_cfg.init_pose_range)
            teleop.build_wait_for_object_waypoints(record_cfg.auto_pick_place.object_load_hold_s)
            _run_step(
                record_cfg, robot, teleop, events, teleop_action_processor, robot_action_processor, robot_observation_processor
            )

            logging.info(
                "\033[32m====== [LOAD] Object %d/%d: jog it down to where it should sit, then press the "
                "right arrow key to confirm (the gripper releases automatically) ======\033[0m",
                i,
                num_objects,
            )
            # The scripted hold step above just turned suction on; seed the
            # jog teleop's own gripper state to match ("closed"/holding)
            # instead of its normal open-on-connect default, so handing off
            # control doesn't drop the object. Release itself is scripted
            # right after (build_retreat_after_manual_place_waypoints), not
            # left to the operator to remember.
            jog_teleop.gripper_action = 0
            jog_teleop.reset_step_size()
            _run_step(
                record_cfg,
                robot,
                jog_teleop,
                events,
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
                control_time_s=record_cfg.auto_pick_place.manual_place_time_sec,
            )

            # Keep the operator's x/y/height, but snap orientation to
            # pick_region's fixed rotation - every later pick (every
            # automatic scene reset) uses that same orientation, so the raw
            # jogged wrist angle would otherwise force a one-off, possibly
            # large reorientation the very first time this object is picked.
            placed_pose = teleop.canonicalize_pick_pose(robot.get_ee_pose())
            height = placed_pose[2]
            logging.info(
                "====== [LOAD] Object %d/%d placed at %s (height=%.4f) ======",
                i,
                num_objects,
                [round(v, 4) for v in placed_pose],
                height,
            )

            teleop.build_retreat_after_manual_place_waypoints(placed_pose)
            _run_step(
                record_cfg, robot, teleop, events, teleop_action_processor, robot_action_processor, robot_observation_processor
            )

            teleop.register_loaded_object(placed_pose, height)
    finally:
        jog_teleop.disconnect()

    logging.info("====== [LOAD] All %d objects loaded ======", num_objects)
