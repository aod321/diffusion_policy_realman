"""
IPhoneARKitReceiver — receive iPhone ARKit VIO poses via Socket.IO
and compute reference-relative motion with deadzone and low-pass filtering
for robot teleoperation.

Instead of accumulating frame-to-frame deltas (which amplifies noise),
this implementation computes displacement relative to a *reference frame*
set at engage time, applies a smooth deadzone to suppress small jitter,
and smooths the result with a time-aware exponential moving average (EMA).

Usage as a context manager:

    with IPhoneARKitReceiver(port=5555) as iphone:
        while True:
            dpos, drot, has_data = iphone.get_relative_motion()
            ...
"""

import base64
import struct
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Data decoding (same logic as record_server.py)
# ---------------------------------------------------------------------------

def decode_arkit_data(encoded_str: str):
    """Decode base64-encoded ARKit frame.

    Format: 16 floats (column-major 4x4) + 1 double timestamp + 1 double wall_clock.
    Returns (transform_4x4, timestamp, wall_clock).
    """
    data_bytes = base64.b64decode(encoded_str)
    mat = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            offset = 4 * (4 * i + j)
            mat[i, j] = struct.unpack('f', data_bytes[offset:offset + 4])[0]
    # Swift simd_float4x4 is column-major → transpose to row-major
    mat = mat.T
    timestamp = struct.unpack('d', data_bytes[64:72])[0]
    wall_clock = struct.unpack('d', data_bytes[72:80])[0] if len(data_bytes) >= 80 else 0.0
    return mat, timestamp, wall_clock


# ---------------------------------------------------------------------------
# IPhoneARKitReceiver
# ---------------------------------------------------------------------------

class IPhoneARKitReceiver:
    """Receive iPhone ARKit poses and compute reference-relative motion.

    Motion is computed as displacement from a reference frame (set at engage
    time via ``reset_reference()``), passed through a smooth deadzone, and
    smoothed with a time-aware EMA filter.

    Parameters
    ----------
    port : int
        Socket.IO listen port (default 5555).
    pos_scale : float
        Scale factor applied to translational displacement.
    rot_scale : float
        Scale factor applied to rotational displacement.
    axis_mapping : array-like, shape (3, 3)
        Rotation matrix mapping ARKit world frame to robot base frame.
        Default: ARKit (X-right, Y-up, Z-toward-user) →
                 Robot  (X-forward, Y-right, Z-up).
    deadzone : float
        Position deadzone in meters (default 1 mm).
    rot_deadzone : float
        Rotation deadzone in radians (default ~0.6°).
    filter_tau : float
        EMA time constant in seconds (default 50 ms).
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
    ):
        self.port = port
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.deadzone = deadzone
        self.rot_deadzone = rot_deadzone
        self.filter_tau = filter_tau

        if axis_mapping is None:
            # Default: ARKit → Robot (X-fwd, Y-right, Z-up)
            self.axis_mapping = np.array([
                [ 0,  0, -1],
                [ 1,  0,  0],
                [ 0,  1,  0],
            ], dtype=np.float64)
        else:
            self.axis_mapping = np.asarray(axis_mapping, dtype=np.float64)

        # --- shared state (protected by _lock) ---
        self._lock = threading.Lock()
        self._T_ref = None                        # reference ARKit 4x4 (set on engage)
        self._filtered_dpos = np.zeros(3)         # filtered position displacement
        self._filtered_drotvec = np.zeros(3)      # filtered rotation displacement (rotvec)
        self._last_filter_time = None             # timestamp for time-aware EMA
        self._has_data = False
        self._connected = False
        self._raw_pos_arkit = np.zeros(3)             # latest raw ARKit position
        self._raw_rot_arkit = np.eye(3)                # latest raw ARKit rotation matrix
        self._raw_dp_arkit = np.zeros(3)               # raw displacement in ARKit frame
        self._raw_dp_robot = np.zeros(3)               # raw displacement in robot frame (before deadzone/filter)

        # --- daemon thread ---
        self._thread = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------- deadzone

    def _apply_pos_deadzone(self, dp):
        """Smooth-ramp position deadzone (no jump at threshold)."""
        norm = np.linalg.norm(dp)
        if norm <= self.deadzone:
            return np.zeros(3)
        return dp * (norm - self.deadzone) / norm

    def _apply_rot_deadzone(self, drotvec):
        """Smooth-ramp rotation deadzone (no jump at threshold)."""
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

        Returns
        -------
        dpos : np.ndarray, shape (3,)
            Filtered position displacement in *robot* frame.
        drot : Rotation
            Filtered orientation displacement in *robot* frame.
        has_data : bool
            True if at least one ARKit frame was processed since last reset.
        """
        with self._lock:
            dpos = self._filtered_dpos.copy()
            drotvec = self._filtered_drotvec.copy()
            has_data = self._has_data
        drot = Rotation.from_rotvec(drotvec)
        return dpos, drot, has_data

    def get_raw_arkit_data(self):
        """Return the latest raw ARKit values (for debugging).

        Returns
        -------
        raw_pos : np.ndarray, shape (3,)
            Latest ARKit absolute position (x, y, z).
        raw_rot : np.ndarray, shape (3, 3)
            Latest ARKit absolute rotation matrix.
        raw_dp_arkit : np.ndarray, shape (3,)
            Raw displacement from reference in ARKit frame.
        raw_dp_robot : np.ndarray, shape (3,)
            Raw displacement in robot frame (before deadzone/filter).
        """
        with self._lock:
            return (
                self._raw_pos_arkit.copy(),
                self._raw_rot_arkit.copy(),
                self._raw_dp_arkit.copy(),
                self._raw_dp_robot.copy(),
            )

    def get_delta_and_reset(self):
        """Deprecated shim — use ``get_relative_motion()`` instead."""
        return self.get_relative_motion()

    def reset_reference(self):
        """Reset the reference pose so the next frame becomes the new origin.

        Call this when re-engaging the clutch or switching modes.
        """
        with self._lock:
            self._T_ref = None
            self._filtered_dpos[:] = 0.0
            self._filtered_drotvec[:] = 0.0
            self._last_filter_time = None
            self._has_data = False

    # --------------------------------------------------------- start / stop

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ---------------------------------------------------- background server

    def _run_server(self):
        """Run eventlet-based Socket.IO server in a daemon thread.

        eventlet is imported lazily here so it does not monkey-patch the
        main process (which may use multiprocessing).
        """
        import eventlet
        eventlet.hubs.use_hub('selects')
        import socketio

        sio = socketio.Server()
        app = socketio.WSGIApp(sio)

        @sio.event
        def connect(sid, environ):
            print(f'[IPhoneARKitReceiver] Client connected: {sid}')
            with self._lock:
                self._connected = True
                self._T_ref = None

        @sio.event
        def disconnect(sid):
            print(f'[IPhoneARKitReceiver] Client disconnected: {sid}')
            with self._lock:
                self._connected = False
                self._T_ref = None

        @sio.on('update')
        def handle_update(sid, data):
            mat, _ts, _wc = decode_arkit_data(data)
            self._process_frame(mat)

        print(f'[IPhoneARKitReceiver] Listening on 0.0.0.0:{self.port}')
        listener = eventlet.listen(('', self.port))
        eventlet.wsgi.server(listener, app, log_output=False)

    # -------------------------------------------------------- frame processing

    def _process_frame(self, T_arkit: np.ndarray):
        """Compute reference-relative displacement, apply deadzone and EMA."""
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

            # Total displacement in ARKit WORLD frame (independent of phone orientation)
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
                alpha = 1.0  # no filtering
            self._filtered_dpos += alpha * (dp_robot - self._filtered_dpos)
            self._filtered_drotvec += alpha * (drotvec_robot - self._filtered_drotvec)
            self._last_filter_time = now
            self._has_data = True
