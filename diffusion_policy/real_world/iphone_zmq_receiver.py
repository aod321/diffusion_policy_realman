"""
IPhoneZMQReceiver — receive iPhone VIO data via ZMQ from node_iphone.py.

Inherits IPhoneARKitTCPReceiver and reuses all motion processing logic
(deadzone, EMA, axis mapping). Only replaces the transport layer:
TCP server → ZMQ SUB subscriber.

Data flow:
    iPhone --TCP--> node_iphone.py --ZMQ PUB--> IPhoneZMQReceiver

Usage:
    with IPhoneZMQReceiver(zmq_endpoint="tcp://localhost:5556") as iphone:
        while True:
            dpos, drot, has_data = iphone.get_relative_motion()
            for event in iphone.get_teleop_events():
                print(event)  # {"cmd": "clutch_engage", "ts": ...}
"""

import json
import threading
import time

import numpy as np
import zmq

from diffusion_policy.real_world.iphone_arkit_tcp_receiver import IPhoneARKitTCPReceiver


class IPhoneZMQReceiver(IPhoneARKitTCPReceiver):
    """ZMQ-based iPhone VIO receiver.

    Subscribes to node_iphone.py's ZMQ PUB socket for frame, teleop, and
    meta topics. Reuses parent's _process_frame() for all motion computation.

    Parameters
    ----------
    zmq_endpoint : str
        ZMQ endpoint to connect to (e.g. "tcp://localhost:5556").
    **kwargs
        Passed to IPhoneARKitTCPReceiver (pos_scale, rot_scale, axis_mapping,
        deadzone, rot_deadzone, filter_tau).
    """

    def __init__(self, zmq_endpoint: str = "tcp://localhost:5556", **kwargs):
        # Initialize parent with port=0 and no mDNS (we don't run a TCP server)
        kwargs.setdefault('port', 0)
        kwargs.setdefault('advertise_mdns', False)
        super().__init__(**kwargs)

        self._zmq_endpoint = zmq_endpoint
        self._zmq_thread = None
        self._teleop_events = []
        self._teleop_lock = threading.Lock()
        self._last_frame_time = 0.0

    # ---- Teleop events API ----

    def get_teleop_events(self) -> list:
        """Return and clear all teleop events since last call.

        Each event is a dict, e.g. {"cmd": "clutch_engage", "ts": 1234567890.123}
        """
        with self._teleop_lock:
            events = self._teleop_events
            self._teleop_events = []
        return events

    # ---- Override is_connected ----

    @property
    def is_connected(self) -> bool:
        """Connected if we received a frame within the last 3 seconds."""
        return (time.monotonic() - self._last_frame_time) < 3.0

    # ---- Override start/stop ----

    def start(self):
        self._stop_event.clear()
        self._zmq_thread = threading.Thread(target=self._zmq_subscriber_loop, daemon=True)
        self._zmq_thread.start()
        # Do NOT call super().start() — we don't want the TCP server

    def stop(self):
        self._stop_event.set()
        if self._zmq_thread is not None:
            self._zmq_thread.join(timeout=3.0)
            self._zmq_thread = None

    # ---- ZMQ subscriber loop ----

    def _zmq_subscriber_loop(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self._zmq_endpoint)
        sub.setsockopt(zmq.SUBSCRIBE, b"frame")
        sub.setsockopt(zmq.SUBSCRIBE, b"teleop")
        sub.setsockopt(zmq.SUBSCRIBE, b"meta")
        # Use a poller so we can check _stop_event periodically
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        print(f"[IPhoneZMQReceiver] Subscribed to {self._zmq_endpoint}")

        try:
            while not self._stop_event.is_set():
                socks = dict(poller.poll(timeout=500))  # 500ms timeout
                if sub not in socks:
                    continue

                topic, data = sub.recv_multipart()

                if topic == b"frame":
                    try:
                        msg = json.loads(data)
                        pose = np.array(msg["pose"], dtype=np.float64).reshape(4, 4)
                        self._process_frame(pose)
                        self._last_frame_time = time.monotonic()
                        with self._lock:
                            self._connected = True
                    except Exception as e:
                        print(f"[IPhoneZMQReceiver] Frame decode error: {e}")

                elif topic == b"teleop":
                    try:
                        event = json.loads(data)
                        with self._teleop_lock:
                            self._teleop_events.append(event)
                        print(f"[IPhoneZMQReceiver] Teleop: {event.get('cmd', '?')}")
                    except Exception as e:
                        print(f"[IPhoneZMQReceiver] Teleop decode error: {e}")

                elif topic == b"meta":
                    try:
                        meta = json.loads(data)
                        with self._lock:
                            self._session_metadata = meta
                        print(f"[IPhoneZMQReceiver] Session metadata: {meta.get('deviceModel', '?')}")
                    except Exception as e:
                        print(f"[IPhoneZMQReceiver] Meta decode error: {e}")

        finally:
            sub.close()
            ctx.term()
            with self._lock:
                self._connected = False
            print("[IPhoneZMQReceiver] Subscriber stopped")
