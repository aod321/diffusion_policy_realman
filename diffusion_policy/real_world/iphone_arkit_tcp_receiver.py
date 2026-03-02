"""
IPhoneARKitTCPReceiver — receive iPhone ARKit VIO poses via raw TCP binary
protocol (compatible with the current iPhoneVIO app's NetworkClient.swift).

Drop-in replacement for IPhoneARKitReceiver. Same API:
    - get_relative_motion() → (dpos, drot, has_data)
    - get_raw_arkit_data()  → (raw_pos, raw_rot, raw_dp_arkit, raw_dp_robot)
    - reset_reference()
    - is_connected

Binary frame protocol (from iPhone):
    Header (8 bytes):
        [0:4]  payload_len  (uint32 LE)
        [4]    msg_type     (0x00=sessionMetadata JSON, 0x01=frameData)
        [5:8]  reserved

    frameData payload:
        [0:4]   jpeg_size    (uint32 LE)
        [4:68]  transform    (16 x float32, column-major 4x4)
        [68:76] device_ts    (float64)
        [76:84] wall_clock   (float64)
        [84:..]  jpeg_data

Usage:
    with IPhoneARKitTCPReceiver(port=5555) as iphone:
        while True:
            dpos, drot, has_data = iphone.get_relative_motion()
            ...
"""

import struct
import time
import json
import socket
import threading
import numpy as np
from scipy.spatial.transform import Rotation


def decode_tcp_frame(payload: bytes):
    """Decode a frameData payload from the TCP binary protocol.

    Returns (transform_4x4_rowmajor, device_ts, wall_clock, jpeg_data).
    """
    jpeg_size = struct.unpack_from('<I', payload, 0)[0]
    # 16 floats column-major (col0_row0, col0_row1, ..., col3_row3)
    floats = struct.unpack_from('<16f', payload, 4)
    # Reshape as column-major: floats[i*4+j] = col i, row j
    mat = np.zeros((4, 4), dtype=np.float64)
    for col in range(4):
        for row in range(4):
            mat[col, row] = floats[col * 4 + row]
    # Transpose: mat was [col][row], numpy is [row][col]
    mat = mat.T

    device_ts = struct.unpack_from('<d', payload, 68)[0]
    wall_clock = struct.unpack_from('<d', payload, 76)[0]
    jpeg_data = payload[84:84 + jpeg_size] if jpeg_size > 0 else b''
    return mat, device_ts, wall_clock, jpeg_data


class IPhoneARKitTCPReceiver:
    """Receive iPhone ARKit poses via raw TCP and compute reference-relative
    motion with deadzone and low-pass filtering for robot teleoperation.

    Parameters
    ----------
    port : int
        TCP listen port (default 5555).
    pos_scale : float
        Scale factor applied to translational displacement.
    rot_scale : float
        Scale factor applied to rotational displacement.
    axis_mapping : array-like, shape (3, 3)
        Rotation matrix mapping ARKit world frame to robot base frame.
    deadzone : float
        Position deadzone in meters (default 1 mm).
    rot_deadzone : float
        Rotation deadzone in radians (default ~0.6 deg).
    filter_tau : float
        EMA time constant in seconds (default 50 ms, 0=off).
    advertise_mdns : bool
        Advertise _vioserver._tcp via mDNS so iPhone auto-discovers (default True).
    """

    def __init__(
        self,
        port: int = 5555,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        axis_mapping=None,
        deadzone: float = 0.001,
        rot_deadzone: float = 0.01,
        filter_tau: float = 0.05,
        advertise_mdns: bool = True,
    ):
        self.port = port
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.deadzone = deadzone
        self.rot_deadzone = rot_deadzone
        self.filter_tau = filter_tau
        self.advertise_mdns = advertise_mdns

        if axis_mapping is None:
            self.axis_mapping = np.array([
                [0, 0, -1],
                [1, 0,  0],
                [0, 1,  0],
            ], dtype=np.float64)
        else:
            self.axis_mapping = np.asarray(axis_mapping, dtype=np.float64)

        # --- shared state (protected by _lock) ---
        self._lock = threading.Lock()
        self._T_ref = None
        self._filtered_dpos = np.zeros(3)
        self._filtered_drot_quat = np.array([0., 0., 0., 1.])  # identity [x,y,z,w]
        self._last_filter_time = None
        self._has_data = False
        self._connected = False
        self._raw_pos_arkit = np.zeros(3)
        self._raw_rot_arkit = np.eye(3)
        self._raw_dp_arkit = np.zeros(3)
        self._raw_dp_robot = np.zeros(3)

        # Session metadata from iPhone (if received)
        self._session_metadata = None

        # Latest JPEG frame (for optional visualization)
        self._latest_jpeg = None

        # --- threads ---
        self._stop_event = threading.Event()
        self._server_thread = None
        self._zeroconf = None
        self._service_info = None

    # ------------------------------------------------------------- deadzone

    def _apply_pos_deadzone(self, dp):
        norm = np.linalg.norm(dp)
        if norm <= self.deadzone:
            return np.zeros(3)
        return dp * (norm - self.deadzone) / norm

    def _apply_rot_deadzone(self, drotvec):
        angle = np.linalg.norm(drotvec)
        if angle <= self.rot_deadzone:
            return np.zeros(3)
        return drotvec * (angle - self.rot_deadzone) / angle

    # ------------------------------------------------------------------ API

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def get_relative_motion(self):
        """Return filtered displacement from reference frame.

        Returns (dpos, drot, has_data) — same as IPhoneARKitReceiver.
        """
        with self._lock:
            dpos = self._filtered_dpos.copy()
            drot = Rotation.from_quat(self._filtered_drot_quat)
            has_data = self._has_data
        return dpos, drot, has_data

    def get_raw_arkit_data(self):
        """Return latest raw ARKit values (for debugging).

        Returns (raw_pos, raw_rot, raw_dp_arkit, raw_dp_robot).
        """
        with self._lock:
            return (
                self._raw_pos_arkit.copy(),
                self._raw_rot_arkit.copy(),
                self._raw_dp_arkit.copy(),
                self._raw_dp_robot.copy(),
            )

    def get_latest_jpeg(self):
        """Return latest JPEG frame data (or None)."""
        with self._lock:
            return self._latest_jpeg

    def reset_reference(self):
        """Reset the reference pose (call when re-engaging clutch)."""
        with self._lock:
            self._T_ref = None
            self._filtered_dpos[:] = 0.0
            self._filtered_drot_quat[:] = [0., 0., 0., 1.]
            self._last_filter_time = None
            self._has_data = False

    def get_delta_and_reset(self):
        """Deprecated shim — use get_relative_motion() instead."""
        return self.get_relative_motion()

    # --------------------------------------------------------- start / stop

    def start(self):
        self._stop_event.clear()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        if self.advertise_mdns:
            self._start_mdns()

    def stop(self):
        self._stop_event.set()
        if self._zeroconf is not None:
            self._stop_mdns()
        if self._server_thread is not None:
            self._server_thread.join(timeout=3.0)
            self._server_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ---------------------------------------------------- mDNS advertising

    def _start_mdns(self):
        """Advertise _vioserver._tcp so iPhone can auto-discover us."""
        try:
            from zeroconf import Zeroconf, ServiceInfo
            hostname = socket.gethostname()
            # Get local IP
            local_ip = self._get_local_ip()
            self._zeroconf = Zeroconf()
            self._service_info = ServiceInfo(
                "_vioserver._tcp.local.",
                f"{hostname}-VIOServer._vioserver._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties={'version': '2.0', 'protocol': 'tcp-binary'},
            )
            self._zeroconf.register_service(self._service_info)
            print(f'[IPhoneARKitTCPReceiver] mDNS: advertising _vioserver._tcp on {local_ip}:{self.port}')
        except ImportError:
            print('[IPhoneARKitTCPReceiver] zeroconf not installed, skipping mDNS advertising')
            print('  Install with: pip install zeroconf')
        except Exception as e:
            print(f'[IPhoneARKitTCPReceiver] mDNS advertising failed: {e}')

    def _stop_mdns(self):
        try:
            if self._service_info is not None:
                self._zeroconf.unregister_service(self._service_info)
            if self._zeroconf is not None:
                self._zeroconf.close()
        except Exception:
            pass
        self._zeroconf = None
        self._service_info = None

    @staticmethod
    def _get_local_ip():
        """Get local IP address (non-loopback)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
        except Exception:
            return '127.0.0.1'
        finally:
            s.close()

    # ---------------------------------------------------- TCP server

    def _run_server(self):
        """Accept TCP connections and process binary frames."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(1.0)
        server.bind(('', self.port))
        server.listen(1)
        print(f'[IPhoneARKitTCPReceiver] Listening on 0.0.0.0:{self.port}')

        while not self._stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            print(f'[IPhoneARKitTCPReceiver] Client connected: {addr}')
            with self._lock:
                self._connected = True
                self._T_ref = None

            client_thread = threading.Thread(
                target=self._handle_client, args=(conn, addr), daemon=True
            )
            client_thread.start()

        server.close()

    def _handle_client(self, conn: socket.socket, addr):
        """Read frames from a single TCP client."""
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        buf = b''
        frame_count = 0

        try:
            while not self._stop_event.is_set():
                # Read until we have at least 8 bytes (header)
                while len(buf) < 8:
                    chunk = conn.recv(65536)
                    if not chunk:
                        raise ConnectionError("Client disconnected")
                    buf += chunk

                # Parse header
                payload_len = struct.unpack_from('<I', buf, 0)[0]
                msg_type = buf[4]
                total_len = 8 + payload_len

                # Read remaining payload
                while len(buf) < total_len:
                    chunk = conn.recv(65536)
                    if not chunk:
                        raise ConnectionError("Client disconnected")
                    buf += chunk

                payload = buf[8:total_len]
                buf = buf[total_len:]

                if msg_type == 0x00:
                    # Session metadata (JSON)
                    try:
                        meta = json.loads(payload)
                        with self._lock:
                            self._session_metadata = meta
                        print(f'[IPhoneARKitTCPReceiver] Session metadata: {meta.get("deviceModel", "?")} session={meta.get("sessionId", "?")[:8]}...')
                    except Exception as e:
                        print(f'[IPhoneARKitTCPReceiver] Failed to parse metadata: {e}')
                elif msg_type == 0x01:
                    # Frame data
                    if payload_len < 84:
                        continue
                    mat, device_ts, wall_clock, jpeg_data = decode_tcp_frame(payload)
                    with self._lock:
                        self._latest_jpeg = jpeg_data if jpeg_data else None
                    self._process_frame(mat)
                    frame_count += 1
                    if frame_count % 300 == 0:
                        print(f'[IPhoneARKitTCPReceiver] Received {frame_count} frames from {addr}')

        except (ConnectionError, OSError) as e:
            print(f'[IPhoneARKitTCPReceiver] Client {addr} disconnected: {e}')
        finally:
            conn.close()
            with self._lock:
                self._connected = False
                self._T_ref = None
            print(f'[IPhoneARKitTCPReceiver] Connection closed: {addr}')

    # -------------------------------------------------------- frame processing

    def _process_frame(self, T_arkit: np.ndarray):
        """Compute reference-relative displacement, apply deadzone and EMA.

        Same logic as IPhoneARKitReceiver._process_frame().
        """
        now = time.monotonic()
        with self._lock:
            # Store raw ARKit pose
            self._raw_pos_arkit = T_arkit[:3, 3].copy()
            self._raw_rot_arkit = T_arkit[:3, :3].copy()

            # First frame after reset → set as reference
            if self._T_ref is None:
                self._T_ref = T_arkit.copy()
                self._last_filter_time = now
                return

            # Total displacement in ARKit WORLD frame
            dp_arkit = T_arkit[:3, 3] - self._T_ref[:3, 3]

            # Relative rotation in ARKit WORLD frame
            R_ref = self._T_ref[:3, :3]
            R_cur = T_arkit[:3, :3]
            dR_arkit = R_cur @ R_ref.T

            # Store raw displacement in ARKit frame
            self._raw_dp_arkit = dp_arkit.copy()

            # Map to robot frame
            M = self.axis_mapping
            dp_robot = M @ dp_arkit * self.pos_scale

            # Store raw displacement in robot frame (before deadzone/filter)
            self._raw_dp_robot = dp_robot.copy()

            # Map rotation: R_robot = M @ R_arkit @ M^T
            dR_robot = M @ dR_arkit @ M.T
            drotvec_robot = Rotation.from_matrix(dR_robot).as_rotvec()
            if self.rot_scale != 1.0:
                drotvec_robot *= self.rot_scale

            # Apply smooth deadzone
            dp_robot = self._apply_pos_deadzone(dp_robot)
            drotvec_robot = self._apply_rot_deadzone(drotvec_robot)

            # Time-aware EMA
            dt = now - self._last_filter_time
            if self.filter_tau > 0 and dt > 0:
                alpha = 1.0 - np.exp(-dt / self.filter_tau)
            else:
                alpha = 1.0
            self._filtered_dpos += alpha * (dp_robot - self._filtered_dpos)
            # Quaternion NLERP for rotation: avoids rotvec discontinuity at ±π
            q_f = self._filtered_drot_quat
            q_t = Rotation.from_rotvec(drotvec_robot).as_quat()  # includes rot_scale + deadzone
            if np.dot(q_f, q_t) < 0:   # shortest-path: flip sign if quaternions diverge
                q_t = -q_t
            q_new = q_f + alpha * (q_t - q_f)
            self._filtered_drot_quat = q_new / np.linalg.norm(q_new)
            self._last_filter_time = now
            self._has_data = True
