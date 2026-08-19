"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import math
import os
import signal
import sys
import time
import traceback
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

# Habitat-related imports
import habitat
# Import side effect only: this registers OVONDatasetV1 with habitat's dataset
# registry (name="OVON-v1") via ovon/__init__.py's own `from ovon.dataset import
# ovon_dataset`. Without this, make_dataset("OVON-v1", ...) fails with
# "Could not find dataset OVON-v1" -- confirmed 2026-08-04 this is exactly what
# was silently crashing every OVON run so far (never visibly hung, just never
# got its stdout captured until the tee fix -- see notes/2026-08-04.md).

# The conditional `import ovon` lives in __main__ (below), NOT here: at module
# scope this ran before _parse_dataset_arg() was defined (line 58 vs line 162),
# which is a NameError at import time. It also MUST run before compose(), since
# the whole point is to populate the Hydra ConfigStore -- __main__ satisfies both.
    
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d", "ovon"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2, mp3d or ovon (default: hm3dv2)",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def compute_oracle_step_count(env, success_distance, max_episode_steps):
    """Oracle shortest-path action count (t_i*) for StepSPL.

    Resolves the nearest goal the same way habitat's own DistanceToGoal/SPL
    measures do (distance_to="POINT", nearest of episode.goals -- confirmed
    no ApexNav config overrides this), then rolls out ShortestPathFollower
    via raw sim.step() calls (bypassing env.step(), which is what actually
    advances _elapsed_steps/task measurements -- so this rollout is invisible
    to the real episode's SPL/success/step tracking). Agent state is restored
    to the episode's start pose afterward so the real ROS-driven navigation
    loop starts from an untouched episode start.

    NOTE: passing `episode` into geodesic_distance() caches path.requested_ends
    and reuses it (ignoring new goal points) on later calls with that same
    episode object -- so the per-goal argmin below deliberately passes
    episode=None to bypass that cache instead of reusing it across goals.
    """
    episode = env.current_episode
    start_position = episode.start_position
    start_rotation = episode.start_rotation

    # Use each goal's navigable view_points, not the raw goal.position --
    # confirmed via [StepSPL DEBUG] (2026-08-09) that geodesic_distance to
    # the raw object position returns inf even on episodes the real agent
    # successfully completes, because goal.position (the object's own
    # location) is frequently off the navmesh. view_points are precomputed
    # navigable points near the object -- this is what habitat-lab's own
    # ObjectNav DistanceToGoal/Success measures resolve to internally.
    candidate_positions = [
        view_point.agent_state.position
        for goal in episode.goals
        for view_point in goal.view_points
    ]

    # Guard added 2026-08-10 -- a full hm3dv1/hm3dv2/mp3d sbatch sweep died with
    # NO traceback anywhere in .out/.err. Root cause: this function had two
    # unguarded failure paths across thousands of episodes -- (1) a goal with
    # zero view_points leaves candidate_positions empty, so np.argmin([]) raises
    # ValueError; (2) ShortestPathFollower.get_next_action() raises
    # habitat_sim.errors.GreedyFollowerError internally if it can't make
    # progress toward an unreachable point. Either exception propagates out of
    # main() and is caught by the top-level `except Exception as e: print(...);
    # os._exit(1)` in __main__ -- but os._exit() skips the normal stdio flush,
    # so on a file-redirected (block-buffered) stdout even that one-line
    # message never reached disk, killing the whole multi-day sweep on a
    # single bad episode with zero evidence of why. Degrade gracefully instead:
    # treat a bad episode's oracle as max_episode_steps (known simplification --
    # see docs/ApexNav_StepSPL_Implementation.md open questions) rather than
    # crash the run.
    if not candidate_positions:
        print(
            f"[StepSPL WARNING] episode {episode.episode_id} has no goal "
            "view_points -- skipping oracle rollout, treating as max_episode_steps"
        )
        env.sim.set_agent_state(start_position, start_rotation)
        return max_episode_steps

    distances = [
        env.sim.geodesic_distance(start_position, [position], None)
        for position in candidate_positions
    ]
    print(f"[StepSPL DEBUG] candidate view_point positions={candidate_positions}")
    print(f"[StepSPL DEBUG] geodesic distances={distances}")
    nearest_goal_position = candidate_positions[int(np.argmin(distances))]
    print(f"[StepSPL DEBUG] nearest_goal_position={nearest_goal_position}")

    follower = ShortestPathFollower(env.sim, success_distance, False)
    oracle_steps = 0
    try:
        action = follower.get_next_action(nearest_goal_position)
        print(f"[StepSPL DEBUG] first action from follower={action}, HabitatSimActions.stop={HabitatSimActions.stop}")
        while action != HabitatSimActions.stop and oracle_steps < max_episode_steps:
            env.sim.step(action)
            oracle_steps += 1
            action = follower.get_next_action(nearest_goal_position)
    except Exception as e:
        print(
            f"[StepSPL WARNING] episode {episode.episode_id} oracle rollout "
            f"failed ({type(e).__name__}: {e}) -- treating as max_episode_steps"
        )
        oracle_steps = max_episode_steps

    env.sim.set_agent_state(start_position, start_rotation)
    return oracle_steps


def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result

    # Only mp3d/hm3d (type: ObjectNav-v1) are meant to be funneled through MP3D's
    # canonical category naming -- their shipped LLM caches are keyed on the
    # post-remap names (confirmed: llm_answer_hm3d.txt has "potted plant", not
    # "plant"). OVON (type: OVON-v1) is keyed on its own raw category strings in
    # llm_answer_ovon_claude.txt, so skip the remap for it entirely -- otherwise
    # table/picture/plant would get silently rewritten and miss the cache.
    apply_mp3d_remap = cfg.habitat.dataset.type == "ObjectNav-v1"
    category_to_coco = {}
    id_to_name = {}
    if apply_mp3d_remap:
         # Load MP3D validation data for object category mapping
        with gzip.open(
            "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
        ) as f:
            val_data = json.load(f)
        category_to_coco = val_data.get("category_to_mp3d_category_id", {})
        id_to_name = {
            category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
            for idx, cat in enumerate(category_to_coco)
        }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # OVON is open-vocabulary -- habitat-lab's default ObjectGoalSensor (defined in
    # object_nav_task.py, pulled in by /habitat/task: objectnav) requires
    # dataset.category_to_task_category_id, a closed-vocab concept OVONDatasetV1
    # doesn't implement. ApexNav never reads this sensor's output (confirmed via
    # grep -rn "objectgoal" ApexNav/*.py -- own VLM/LLM pipeline does detection),
    # so drop it for OVON runs instead of faking a category mapping.
    if cfg.habitat.dataset.type == "OVON-v1":
        from omegaconf import OmegaConf, open_dict
        was_readonly = OmegaConf.is_readonly(cfg)
        OmegaConf.set_readonly(cfg, False)
        with open_dict(cfg):
            cfg.habitat.task.lab_sensors.pop("objectgoal_sensor", None)
        OmegaConf.set_readonly(cfg, was_readonly)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        step_spl_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    # Camera height from the SAME composed config the simulator is running, not a
    # constant -- see habitat2ros/habitat_publisher.py and derive_camera_params.py.
    ros_pub = habitat_publisher.ROSPublisher(
        camera_height=float(
            cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position[1]
        )
    )
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    for epi in range(number_of_episodes - num_total):
        # Publish progress information
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        oracle_step_count = compute_oracle_step_count(
            env, success_distance, max_episode_steps
        )
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)

            # Calculate ITM cosine similarity score
            cosine = get_itm_message_cosine(observations["rgb"], label, room)
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity: {cosine:.3f}")

            publish_float64(itm_score_pub, cosine)

            # Detect objects in the current observation
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                label, observations["rgb"], detector_cfg, llm_answer
            )

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # StepSPL: SPL's formula but in discrete action count instead of
        # continuous path length -- max(..., 1) guards the 0/0 case where the
        # agent starts within success_distance of the goal (oracle_step_count
        # and count_steps both 0), which would otherwise raise ZeroDivisionError.
        step_spl = success * (
            oracle_step_count / max(count_steps, oracle_step_count, 1)
        )

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        # Guard added 2026-08-10 -- Habitat's own SPL/soft_spl/distance_to_goal
        # measures return inf/NaN for episodes where the goal is genuinely
        # unreachable via the navmesh (confirmed on a hm3d-ovon val_seen episode,
        # target "container" -- a rarer/obscure category HM3D-ObjectNav's narrow
        # 6-category set would never sample near, despite OVON sharing the exact
        # same HM3DSem scene data/navmesh as hm3d/mp3d). is_on_same_floor() is
        # only a coarse 2m height-window heuristic, so "infeasible" per our own
        # check_failure() does NOT reliably mean the navmesh is actually
        # disconnected -- most infeasible episodes still have a finite (if
        # large) geodesic distance. These are running sums used to compute an
        # average every episode, so a single non-finite value permanently
        # poisons every subsequent average for the rest of the run (NaN/inf
        # propagate through +=). Treat non-finite as a 0.0 contribution instead
        # -- num_total still counts the episode; StepSPL is unaffected since its
        # formula can't produce non-finite values in the first place.
        num_total += 1
        spl_all += spl if math.isfinite(spl) else 0.0
        soft_spl_all += soft_spl if math.isfinite(soft_spl) else 0.0
        step_spl_all += step_spl
        distance_to_goal_all += (
            distance_to_goal if math.isfinite(distance_to_goal) else 0.0
        )
        distance_to_goal_reward_all += (
            distance_to_goal_reward
            if math.isfinite(distance_to_goal_reward)
            else 0.0
        )

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average StepSPL", f"{step_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total StepSPL", f"{step_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()

        # OVON-only, and only ever here. `import ovon` does cs.store(group="habitat",
        # name="habitat_config_base", node=OVONHabitatConfig), which OVERWRITES the
        # identical store in habitat-lab's default_structured_configs.py:1919. That
        # swaps habitat.simulator.type from Sim-v0 to OVONSim-v0 for EVERY dataset,
        # and OVONSim recomputes the navmesh on init and on every scene change
        # (ovon/task/simulator.py:12,38), discarding the shipped *.basis.navmesh the
        # episodes were generated against -- moving geodesic distance, traversability,
        # distance_to_goal, the SPL denominator and success. Confirmed 2026-08-11:
        # baseline hm3dv1 logged "initializing sim Sim-v0", the unconditional-import
        # runs logged "initializing sim OVONSim-v0". Must precede compose() so the
        # ConfigStore is populated in time for OVON's own config to resolve.
        if dataset == "ovon":
            import ovon  # noqa: F401

        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        # Print full traceback (not just str(e)) and force a flush before
        # os._exit() -- added 2026-08-10 after a full sbatch sweep died with
        # NO error text anywhere in its logs. os._exit() skips Python's normal
        # stdio flush, so on a file-redirected (block-buffered) stdout, even
        # this handler's own message could be lost if the buffer hadn't
        # naturally flushed yet. sys.stdout/stderr.flush() before _exit()
        # guarantees whatever we print here actually reaches disk.
        print(f"Unexpected error occurred: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
