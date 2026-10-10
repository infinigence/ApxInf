"""Read host memory observations without loading a model or changing system state."""

import ctypes
from functools import lru_cache
import os
import struct
import sys
import time

try:
    import resource
except ImportError:
    resource = None


@lru_cache(maxsize=1)
def _sysctl_function():
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    try:
        function = library.sysctlbyname
    except AttributeError as error:
        raise OSError("The macOS sysctlbyname interface is unavailable.") from error
    function.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                         ctypes.c_void_p, ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


def _sysctl_bytes(name, capacity):
    if sys.platform != "darwin":
        raise OSError("The host memory sensor requires macOS.")
    buffer = ctypes.create_string_buffer(capacity)
    size = ctypes.c_size_t(capacity)
    ctypes.set_errno(0)
    status = _sysctl_function()(name.encode("ascii"), buffer, ctypes.byref(size), None, 0)
    if status != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), name)
    if not 0 < size.value <= capacity:
        raise ValueError(f"The {name} sensor returned an invalid size: {size.value}.")
    return buffer.raw[:size.value]


def _fixed_read(reader, name, size):
    value = reader(name, size)
    if not isinstance(value, bytes) or len(value) != size:
        raise ValueError(f"The {name} sensor must return exactly {size} bytes.")
    return value


def _pressure(reader):
    result = {"state": "unknown", "dispatch_value": None, "error": None}
    try:
        value, = struct.unpack("=I", _fixed_read(reader, "kern.memorystatus_vm_pressure_level", 4))
        result["dispatch_value"] = value
        state = {1: "normal", 2: "warning", 4: "critical"}.get(value)
        if state is None:
            raise ValueError(f"The pressure sensor returned an unknown dispatch value: {value}.")
        result["state"] = state
    except (OSError, ValueError, TypeError) as error:
        result["error"] = str(error)
    return result


def _swap(reader):
    result = {"total_bytes": None, "used_bytes": None, "available_bytes": None,
              "page_size_bytes": None, "encrypted": None, "error": None}
    try:
        # These fixed-width fields follow the public macOS xsw_usage ABI.
        total, available, used, page_size, encrypted = struct.unpack(
            "=QQQII", _fixed_read(reader, "vm.swapusage", 32))
        if page_size == 0 or encrypted not in (0, 1) or used > total or available > total:
            raise ValueError("The swap sensor returned invalid fields.")
        result.update(total_bytes=total, used_bytes=used, available_bytes=available,
                      page_size_bytes=page_size, encrypted=bool(encrypted))
    except (OSError, ValueError, TypeError) as error:
        result["error"] = str(error)
    return result


def _peak_rss():
    if resource is None:
        raise OSError("The process resource interface is unavailable.")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def snapshot(*, sysctl_reader=None, rusage_reader=None, platform_name=None):
    """Return pressure, swap, and this process's lifetime peak RSS as JSON values."""
    reader = _sysctl_bytes if sysctl_reader is None else sysctl_reader
    rusage_reader = _peak_rss if rusage_reader is None else rusage_reader
    platform_name = sys.platform if platform_name is None else platform_name
    started = time.monotonic_ns()
    peak = None
    peak_error = None
    if platform_name == "darwin":
        try:
            peak = rusage_reader()
            if type(peak) is not int or peak < 0:
                raise ValueError("The process peak RSS must be a nonnegative integer.")
        except (OSError, ValueError, TypeError) as error:
            peak = None
            peak_error = str(error)
    else:
        peak_error = "Process peak RSS byte units are supported only on macOS."
    return {"monotonic_ns": started, "pressure": _pressure(reader),
            "process_peak_rss_bytes": peak, "process_peak_rss_error": peak_error,
            "swap": _swap(reader)}


def hardware_snapshot(*, sysctl_reader=None):
    """Return bounded hardware identity fields and explicit per-field errors."""
    reader = _sysctl_bytes if sysctl_reader is None else sysctl_reader
    result = {"model": None, "memory_bytes": None, "errors": {}}
    try:
        raw = reader("hw.model", 256)
        if not isinstance(raw, bytes) or not 2 <= len(raw) <= 256 or raw[-1:] != b"\0" or b"\0" in raw[:-1]:
            raise ValueError("The hardware model must be a bounded, terminated string.")
        result["model"] = raw[:-1].decode("utf-8", errors="strict")
    except (OSError, ValueError, TypeError) as error:
        result["errors"]["model"] = str(error)
    try:
        value, = struct.unpack("=Q", _fixed_read(reader, "hw.memsize", 8))
        if value == 0:
            raise ValueError("The hardware memory size must be positive.")
        result["memory_bytes"] = value
    except (OSError, ValueError, TypeError) as error:
        result["errors"]["memory_bytes"] = str(error)
    return result
