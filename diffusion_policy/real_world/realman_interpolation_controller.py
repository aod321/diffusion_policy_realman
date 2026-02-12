import os
import time
import enum
import json
import socket
import struct
import threading
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


# ============ Unit Conversion Helpers ============

def pose_to_realman_quat(pose):
    """
    Convert internal 6D pose [x,y,z,rx,ry,rz] (meters + rotvec)
    to Realman pose_quat integers [x,y,z,w,qx,qy,qz] in 0.001mm / 1e-6 quat units.
    """
    pos = pose[:3]
    rotvec = pose[3:]
    rot = st.Rotation.from_rotvec(rotvec)
    # scipy as_quat returns [x, y, z, w]
    quat_xyzw = rot.as_quat()
    # Realman expects [w, x, y, z]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    # Position: meters -> 0.001mm (multiply by 1e6)
    pos_int = np.round(pos * 1e6).astype(np.int64)
    # Quaternion: multiply by 1e6 to get integer representation
    quat_int = np.round(quat_wxyz * 1e6).astype(np.int64)

    # Return 7-element list matching Realman pose_quat format: [x, y, z, w, qx, qy, qz]
    return [
        int(pos_int[0]), int(pos_int[1]), int(pos_int[2]),
        int(quat_int[0]), int(quat_int[1]), int(quat_int[2]), int(quat_int[3])
    ]


def realman_udp_to_pose(waypoint):
    """
    Convert Realman UDP waypoint (quat format) to internal 6D pose.
    waypoint: dict with 'position' (3-array, 0.000001m) and 'quat' (4-array [w,x,y,z], 0.000001)
    Returns: np.array([x, y, z, rx, ry, rz]) in meters + rotvec
    """
    # Position: 0.000001m -> meters (divide by 1e6)
    position = waypoint['position']
    pos = np.array([position[0], position[1], position[2]], dtype=np.float64) / 1e6

    # Quaternion: divide by 1e6 to get float, Realman sends [w, x, y, z]
    quat = waypoint['quat']
    w = quat[0] / 1e6
    qx = quat[1] / 1e6
    qy = quat[2] / 1e6
    qz = quat[3] / 1e6

    # scipy from_quat expects [x, y, z, w]
    rot = st.Rotation.from_quat([qx, qy, qz, w])
    rotvec = rot.as_rotvec()

    return np.concatenate([pos, rotvec])


def realman_udp_to_joints(joint_status, num_joints=7):
    """
    Convert Realman UDP joint_status to joint positions (rad) and velocities (rad/s).
    joint_status: dict with 'joint_position' array (0.001 degree units)
    Returns: (joint_positions, joint_velocities) as numpy arrays
    """
    positions = np.zeros(num_joints, dtype=np.float64)
    velocities = np.zeros(num_joints, dtype=np.float64)

    joint_position = joint_status.get('joint_position', [])
    for i in range(min(num_joints, len(joint_position))):
        # joint_position: units of 0.001 degrees -> radians
        positions[i] = joint_position[i] * 0.001 * (np.pi / 180.0)

    # No joint_speed field in UDP protocol; velocities remain zeros
    return positions, velocities


# ============ TCP Client ============

class RealmanTCPClient:
    """Persistent TCP connection for sending JSON commands to Realman robot."""

    def __init__(self, robot_ip, tcp_port=8080, verbose=False):
        self.robot_ip = robot_ip
        self.tcp_port = tcp_port
        self.verbose = verbose
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.robot_ip, self.tcp_port))
        self.sock.settimeout(2.0)
        if self.verbose:
            print(f"[RealmanTCP] Connected to {self.robot_ip}:{self.tcp_port}")

    def disconnect(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def send_command(self, cmd_dict):
        """Send a JSON command and receive response."""
        data = json.dumps(cmd_dict, separators=(',', ':'))
        self.sock.sendall(data.encode('utf-8'))
        # Read response
        try:
            response = self.sock.recv(4096)
            if response:
                return json.loads(response.decode('utf-8'))
        except socket.timeout:
            if self.verbose:
                print("[RealmanTCP] Response timeout")
        except json.JSONDecodeError:
            if self.verbose:
                print("[RealmanTCP] Invalid JSON response")
        return None

    def send_command_no_response(self, cmd_dict):
        """Send a JSON command without waiting for response (for high-freq commands)."""
        data = json.dumps(cmd_dict, separators=(',', ':'))
        self.sock.sendall(data.encode('utf-8'))

    def movep_canfd(self, pose_quat, follow=True):
        """
        Send movep_canfd pass-through command.
        pose_quat: 7-element list [x, y, z, w, qx, qy, qz] (integer units)
        follow: if True, robot will queue; if False, execute immediately
        """
        cmd = {
            'command': 'movep_canfd',
            'pose_quat': pose_quat,
            'follow': follow
        }
        self.send_command_no_response(cmd)

    def movej(self, joint_positions_deg, speed=20, block=True):
        """
        Send movej command to move to target joint configuration.
        joint_positions_deg: list of joint angles in degrees
        speed: percentage of max speed (0-100)
        block: if True, wait for completion
        """
        cmd = {
            'command': 'movej',
            'joint': [round(j * 1000) for j in joint_positions_deg],  # 0.001 degree units
            'v': speed,
            'r': 0,
            'trajectory_connect': 0,
        }
        resp = self.send_command(cmd)
        if block and resp is not None:
            # Wait for motion to complete by polling state
            time.sleep(0.1)
        return resp

    def set_realtime_push(self, cycle, port, force_coordinate=0, ip=''):
        """
        Configure UDP real-time state push.
        cycle: push frequency divider (1=5ms, 2=10ms, etc.)
        port: UDP port to push to
        """
        cmd = {
            'command': 'set_realtime_push',
            'enable': True,
            'cycle': cycle,
            'port': port,
            'force_coordinate': force_coordinate,
            'ip': ip
        }
        return self.send_command(cmd)

    def set_arm_slow_stop(self):
        """Send slow stop command."""
        cmd = {'command': 'set_arm_slow_stop'}
        return self.send_command(cmd)


# ============ UDP Receiver ============

class RealmanUDPReceiver:
    """Background thread that receives UDP state reports from Realman robot."""

    def __init__(self, host_ip='0.0.0.0', udp_port=8089, verbose=False):
        self.host_ip = host_ip
        self.udp_port = udp_port
        self.verbose = verbose
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._latest_state = None
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host_ip, self.udp_port))
        self.sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print(f"[RealmanUDP] Listening on {self.host_ip}:{self.udp_port}")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def get_latest_state(self):
        with self._lock:
            return self._latest_state

    def _receive_loop(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(65536)
                state = json.loads(data.decode('utf-8'))
                with self._lock:
                    self._latest_state = state
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                if self.verbose:
                    print("[RealmanUDP] Invalid JSON in UDP packet")
            except Exception as e:
                if self._running and self.verbose:
                    print(f"[RealmanUDP] Receive error: {e}")


# ============ Main Controller ============

class RealmanInterpolationController(mp.Process):
    """
    Drop-in replacement for RTDEInterpolationController using Realman robot protocol.
    Uses TCP JSON for sending movep_canfd commands and UDP for receiving robot state.
    """

    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ip,
            frequency=100,  # 10ms cycle for movep_canfd
            max_pos_speed=0.25,
            max_rot_speed=0.16,
            launch_timeout=10,
            joints_init=None,
            joints_init_speed=20,  # percentage of max speed for Realman
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            # Realman-specific parameters
            tcp_port=8080,
            udp_port=8089,
            udp_cycle=2,  # 2 = 10ms push cycle
            num_joints=7,
            follow=True,
            host_ip='0.0.0.0',
            ):
        # verify
        assert 0 < frequency <= 200
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (num_joints,)

        super().__init__(name="RealmanPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.tcp_port = tcp_port
        self.udp_port = udp_port
        self.udp_cycle = udp_cycle
        self.num_joints = num_joints
        self.follow = follow
        self.host_ip = host_ip

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',
                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]

        # Create example data for ring buffer (no robot connection needed at init time)
        example = dict()
        for key in receive_keys:
            if key in ('ActualTCPPose', 'ActualTCPSpeed', 'TargetTCPPose', 'TargetTCPSpeed'):
                example[key] = np.zeros(6, dtype=np.float64)
            elif key in ('ActualQ', 'ActualQd', 'TargetQ', 'TargetQd'):
                example[key] = np.zeros(num_joints, dtype=np.float64)
            else:
                example[key] = np.zeros(6, dtype=np.float64)
        example['robot_receive_timestamp'] = time.time()

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RealmanPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # Connect TCP to robot
        tcp_client = RealmanTCPClient(
            robot_ip=self.robot_ip,
            tcp_port=self.tcp_port,
            verbose=self.verbose
        )
        tcp_client.connect()

        # Start UDP receiver
        udp_receiver = RealmanUDPReceiver(
            host_ip=self.host_ip,
            udp_port=self.udp_port,
            verbose=self.verbose
        )
        udp_receiver.start()

        try:
            if self.verbose:
                print(f"[RealmanPositionalController] Connected to robot: {self.robot_ip}")

            # Configure UDP push
            resp = tcp_client.set_realtime_push(
                cycle=self.udp_cycle,
                port=self.udp_port
            )
            if self.verbose:
                print(f"[RealmanPositionalController] UDP push configured: {resp}")

            # Move to init joints if specified
            if self.joints_init is not None:
                joints_deg = np.rad2deg(self.joints_init)
                tcp_client.movej(joints_deg.tolist(), speed=self.joints_init_speed, block=True)
                # Wait for motion to complete
                time.sleep(2.0)
                if self.verbose:
                    print("[RealmanPositionalController] Moved to initial joint configuration")

            # Wait for first UDP state packet
            deadline = time.time() + self.launch_timeout
            while time.time() < deadline:
                state = udp_receiver.get_latest_state()
                if state is not None:
                    break
                time.sleep(0.01)

            if udp_receiver.get_latest_state() is None:
                raise RuntimeError("Timeout waiting for first UDP state packet from Realman robot")

            # Get initial pose from UDP state
            udp_state = udp_receiver.get_latest_state()
            curr_pose = realman_udp_to_pose(udp_state['waypoint'])
            prev_pose = curr_pose.copy()
            prev_time = time.time()

            # Initialize trajectory interpolator
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # Main loop
            dt = 1. / self.frequency
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_start = time.perf_counter()

                # Send command to robot
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)
                pose_quat = pose_to_realman_quat(pose_command)
                tcp_client.movep_canfd(pose_quat, follow=self.follow)

                # Read latest UDP state and convert
                udp_state = udp_receiver.get_latest_state()
                if udp_state is not None:
                    now_time = time.time()
                    state = dict()

                    # Parse pose from waypoint
                    if 'waypoint' in udp_state:
                        actual_pose = realman_udp_to_pose(udp_state['waypoint'])
                    else:
                        actual_pose = np.zeros(6, dtype=np.float64)

                    # Numerically differentiate TCP speed
                    time_delta = now_time - prev_time
                    if time_delta > 0:
                        tcp_speed = (actual_pose - prev_pose) / time_delta
                    else:
                        tcp_speed = np.zeros(6, dtype=np.float64)
                    prev_pose = actual_pose.copy()
                    prev_time = now_time

                    # Parse joint status
                    joint_status = udp_state.get('joint_status', [])
                    joint_positions, joint_velocities = realman_udp_to_joints(
                        joint_status, self.num_joints)

                    # Fill state dict based on receive_keys
                    for key in self.receive_keys:
                        if key == 'ActualTCPPose':
                            state[key] = actual_pose
                        elif key == 'ActualTCPSpeed':
                            state[key] = tcp_speed
                        elif key == 'ActualQ':
                            state[key] = joint_positions
                        elif key == 'ActualQd':
                            state[key] = joint_velocities
                        elif key == 'TargetTCPPose':
                            state[key] = pose_command.copy()
                        elif key == 'TargetTCPSpeed':
                            state[key] = np.zeros(6, dtype=np.float64)
                        elif key == 'TargetQ':
                            state[key] = np.zeros(self.num_joints, dtype=np.float64)
                        elif key == 'TargetQd':
                            state[key] = np.zeros(self.num_joints, dtype=np.float64)

                    state['robot_receive_timestamp'] = now_time
                    self.ring_buffer.put(state)

                # Fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # Execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RealmanPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # Regulate frequency with busy wait
                t_end = t_start + dt
                while time.perf_counter() < t_end:
                    pass

                # First loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    actual_dt = time.perf_counter() - t_start
                    print(f"[RealmanPositionalController] Actual frequency {1/actual_dt:.1f} Hz")

        finally:
            # Mandatory cleanup
            try:
                tcp_client.set_arm_slow_stop()
            except Exception:
                pass
            udp_receiver.stop()
            tcp_client.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RealmanPositionalController] Disconnected from robot: {self.robot_ip}")
