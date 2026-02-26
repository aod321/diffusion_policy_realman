from typing import Tuple
from dataclasses import dataclass
import warnings
import multiprocessing as mp

import numpy as np
from multiprocessing.managers import SharedMemoryManager

try:
    from atomics import atomicview, MemoryOrder, UINT
    _ATOMICS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local install
    atomicview = None
    MemoryOrder = None
    UINT = None
    _ATOMICS_IMPORT_ERROR = exc


@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self,
            shm_manager: SharedMemoryManager,
            size: int = 8  # 64bit int
            ):
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self._mod = 1 << (8 * size)
        self._lock = mp.Lock()
        self._use_atomics = atomicview is not None

        if self._use_atomics:
            try:
                self.store(0)
            except Exception as exc:
                self._disable_atomics(exc)
                self._store_locked(0)
        else:
            if _ATOMICS_IMPORT_ERROR is not None:
                warnings.warn(
                    f"atomics import failed ({_ATOMICS_IMPORT_ERROR}); using lock-based shared counter.",
                    RuntimeWarning,
                )
            self._store_locked(0)

    @property
    def buf(self):
        return self.shm.buf[:self.size]

    def _disable_atomics(self, exc: Exception):
        if self._use_atomics:
            warnings.warn(
                f"atomics backend unavailable ({exc}); falling back to lock-based shared counter.",
                RuntimeWarning,
            )
        self._use_atomics = False

    def _load_locked(self) -> int:
        with self._lock:
            return int.from_bytes(self.buf.tobytes(), byteorder='little', signed=False)

    def _store_locked(self, value: int):
        v = int(value) % self._mod
        raw = v.to_bytes(self.size, byteorder='little', signed=False)
        with self._lock:
            self.buf[:] = raw

    def _add_locked(self, value: int):
        with self._lock:
            current = int.from_bytes(self.buf.tobytes(), byteorder='little', signed=False)
            updated = (current + int(value)) % self._mod
            self.buf[:] = updated.to_bytes(self.size, byteorder='little', signed=False)

    def load(self) -> int:
        if self._use_atomics:
            try:
                with atomicview(buffer=self.buf, atype=UINT) as a:
                    return a.load(order=MemoryOrder.ACQUIRE)
            except Exception as exc:
                self._disable_atomics(exc)
        return self._load_locked()

    def store(self, value: int):
        if self._use_atomics:
            try:
                with atomicview(buffer=self.buf, atype=UINT) as a:
                    a.store(int(value), order=MemoryOrder.RELEASE)
                return
            except Exception as exc:
                self._disable_atomics(exc)
        self._store_locked(value)

    def add(self, value: int):
        if self._use_atomics:
            try:
                with atomicview(buffer=self.buf, atype=UINT) as a:
                    a.add(int(value), order=MemoryOrder.ACQ_REL)
                return
            except Exception as exc:
                self._disable_atomics(exc)
        self._add_locked(value)
