"""
Usage:
(robodiff)$ python demo_real_robot_iphone.py -o <demo_save_dir> --robot_ip <ip>

Robot movement:
Connect your iPhone (running ARKit VIO app) to the Socket.IO server.
Press "E" to engage/disengage the clutch.
When engaged, moving the iPhone moves the robot EEF.
When disengaged, iPhone movement is ignored (robot holds position).

Movement modes (like SpaceMouse):
  Default    — XY translation only (rotation & height locked).
  Hold "R"   — rotation mode (translation locked, rotation unlocked).
  Hold "F"   — unlock height (Z-axis), combinable with default XY.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.iphone_arkit_receiver import IPhoneARKitReceiver
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="Robot IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--robot_type', '-rt', default='ur5', type=click.Choice(['ur5', 'realman']), help="Robot type: ur5 or realman.")
@click.option('--follow', is_flag=True, default=False, help="Realman: enable high-follow mode (use with --lookahead).")
@click.option('--lookahead', default=None, type=float, help="Realman: command-side lookahead smoothing in seconds (default: 0.1 with --follow, 0.0 without).")
@click.option('--iphone_port', default=5555, type=int, help="Socket.IO port for iPhone ARKit receiver.")
@click.option('--pos_scale', default=1.0, type=float, help="Position delta scale factor.")
@click.option('--rot_scale', default=1.0, type=float, help="Rotation delta scale factor.")
@click.option('--deadzone', default=0.001, type=float, help="Position deadzone in meters (default 1mm).")
@click.option('--rot_deadzone', default=0.01, type=float, help="Rotation deadzone in radians (default ~0.6 deg).")
@click.option('--filter_tau', default=0.05, type=float, help="EMA filter time constant in seconds (default 50ms, 0=off).")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency,
         robot_type, follow, lookahead, iphone_port, pos_scale, rot_scale,
         deadzone, rot_deadzone, filter_tau):
    if lookahead is None:
        lookahead = 0.1 if follow else 0.0
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            IPhoneARKitReceiver(
                port=iphone_port,
                pos_scale=pos_scale,
                rot_scale=rot_scale,
                axis_mapping=(
                    np.array([
                        [1,  0,  0],   # Robot X(right)   = ARKit X(right)
                        [0,  0, -1],   # Robot Y(forward) = -ARKit Z(backward)
                        [0,  1,  0],   # Robot Z(up)      = ARKit Y(up)
                    ]) if robot_type == 'realman' else None
                ),
                deadzone=deadzone,
                rot_deadzone=rot_deadzone,
                filter_tau=filter_tau,
            ) as iphone, \
            RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                robot_type=robot_type,
                realman_follow=follow,
                realman_lookahead_time=lookahead
            ) as env:
            cv2.setNumThreads(1)

            has_camera = env.realsense is not None
            if has_camera:
                # realsense exposure
                env.realsense.set_exposure(exposure=120, gain=0)
                # realsense white balance
                env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            print('Press E to engage/disengage iPhone control.')
            print('Hold R for rotation mode, hold F to unlock height.')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            is_engaged = False

            # Reference-relative state
            target_pose_ref = None       # robot target pose snapshot at engage time
            prev_rot_mode = False        # previous frame's rotation mode
            prev_height_unlock = False   # previous frame's height unlock
            stage = 0

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == KeyCode(char='e'):
                        # Toggle engage/disengage
                        is_engaged = not is_engaged
                        if is_engaged:
                            iphone.reset_reference()
                            target_pose_ref = target_pose.copy()
                            prev_rot_mode = key_counter.is_key_held(KeyCode(char='r'))
                            prev_height_unlock = key_counter.is_key_held(KeyCode(char='f'))
                            print('Engaged! iPhone controls robot.')
                        else:
                            target_pose_ref = None
                            print('Disengaged. iPhone movement ignored.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                    stage = key_counter[Key.space]

                # visualize
                if has_camera:
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                    episode_id = env.replay_buffer.n_episodes

                    # build status text lines
                    connected = iphone.is_connected
                    _rot = key_counter.is_key_held(KeyCode(char='r'))
                    _ht = key_counter.is_key_held(KeyCode(char='f'))
                    mode_str = 'ROT' if _rot else ('XYZ' if _ht else 'XY')
                    line1 = f'Ep: {episode_id} | iPhone: {"ON" if connected else "OFF"} | Clutch: {"ENGAGED" if is_engaged else "OFF"} | {mode_str}'
                    if is_recording:
                        line1 += ' | REC'

                    cv2.putText(
                        vis_img, line1,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        thickness=2,
                        color=(0, 255, 0) if is_engaged else (255, 255, 255)
                    )

                    cv2.imshow('default', vis_img)
                    cv2.pollKey()

                precise_wait(t_sample)

                # Detect mode changes and reset reference to prevent jumps
                rot_mode = key_counter.is_key_held(KeyCode(char='r'))
                height_unlock = key_counter.is_key_held(KeyCode(char='f'))
                if is_engaged and target_pose_ref is not None:
                    if rot_mode != prev_rot_mode or height_unlock != prev_height_unlock:
                        iphone.reset_reference()
                        target_pose_ref = target_pose.copy()
                prev_rot_mode = rot_mode
                prev_height_unlock = height_unlock

                # get teleop command from iPhone
                dpos, drot, has_data = iphone.get_relative_motion()

                # SpaceMouse-style lock: default XY-only, hold keys to unlock
                if not rot_mode:
                    drot = st.Rotation.identity()   # lock rotation
                else:
                    dpos = np.zeros(3)               # lock translation
                if not height_unlock:
                    dpos[2] = 0                      # lock Z

                if is_engaged and has_data and target_pose_ref is not None:
                    # Absolute: reference pose + total displacement
                    target_pose[:3] = target_pose_ref[:3] + dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose_ref[3:])).as_rotvec()

                # debug print every 10 iterations (~1s at 10Hz)
                if iter_idx % 10 == 0:
                    cur_state = env.get_robot_state()
                    cur_pose = cur_state.get('ActualTCPPose', None)
                    raw_pos, raw_rot, raw_dp_arkit, raw_dp_robot = iphone.get_raw_arkit_data()
                    raw_euler = st.Rotation.from_matrix(raw_rot).as_euler('xyz', degrees=True)
                    np.set_printoptions(precision=4, suppress=True)
                    print(f'--- iter {iter_idx} ---')
                    print(f'  iPhone: {"connected" if iphone.is_connected else "disconnected"} | engaged: {is_engaged}')
                    print(f'  [RAW] ARKit pos:     {raw_pos}')
                    print(f'  [RAW] ARKit euler:   {raw_euler}')
                    print(f'  [RAW] dp(ARKit):     {raw_dp_arkit}')
                    print(f'  [RAW] dp(robot):     {raw_dp_robot}')
                    print(f'  dpos(filtered):      {dpos}')
                    print(f'  drot(filtered):      {drot.as_rotvec()}')
                    print(f'  target:  {target_pose}')
                    print(f'  actual:  {cur_pose}')

                # execute teleop command
                env.exec_actions(
                    actions=[target_pose],
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
