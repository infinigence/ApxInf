"""Original CPU tests for the read-only host memory observation boundary."""

import ctypes
import errno
import importlib.util
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("host_memory_test_module", ROOT / "benchmarks/serving/host_memory.py")
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


class HostMemoryTests(unittest.TestCase):
    def reader(self, pressure=1, swap=None):
        values = {"kern.memorystatus_vm_pressure_level": struct.pack("=I", pressure),
                  "vm.swapusage": swap if swap is not None else struct.pack("=QQQII", 8192, 6144, 2048, 4096, 1)}
        return lambda name, size: values[name]

    def snapshot(self, reader=None, peak=12345, platform_name="darwin"):
        return host.snapshot(sysctl_reader=reader or self.reader(), rusage_reader=lambda: peak,
                             platform_name=platform_name)

    def test_dispatch_values_match_exact_states(self):
        for value, state in ((1, "normal"), (2, "warning"), (4, "critical")):
            with self.subTest(value=value):
                observation = self.snapshot(self.reader(value))
                self.assertEqual(observation["pressure"], {"state": state, "dispatch_value": value, "error": None})
                self.assertGreater(observation["monotonic_ns"], 0)
                json.dumps(observation, allow_nan=False)

    def test_unknown_and_combined_values_never_allow_execution(self):
        for value in (0, 3, 5, 6, 7, 8, 0xffffffff):
            with self.subTest(value=value):
                pressure = self.snapshot(self.reader(value))["pressure"]
                self.assertEqual(pressure["state"], "unknown")
                self.assertEqual(pressure["dispatch_value"], value)
                self.assertIn("unknown dispatch value", pressure["error"])

    def test_pressure_size_errors_remain_unknown(self):
        for raw in (b"", b"\x01", b"\x01" * 8, "invalid"):
            with self.subTest(raw=raw):
                reader = self.reader()
                result = self.snapshot(lambda name, size: raw if size == 4 else reader(name, size))
                self.assertEqual(result["pressure"]["state"], "unknown")
                self.assertIsNone(result["pressure"]["dispatch_value"])
                self.assertIn("exactly 4 bytes", result["pressure"]["error"])
                self.assertIsNone(result["swap"]["error"])

    def test_permission_error_retains_reason_without_zero_samples(self):
        def denied(name, size):
            raise PermissionError(errno.EPERM, "Sensor denied", name)
        observation = self.snapshot(denied)
        self.assertEqual(observation["pressure"]["state"], "unknown")
        self.assertIn("Sensor denied", observation["pressure"]["error"])
        self.assertIsNone(observation["swap"]["used_bytes"])
        self.assertIn("vm.swapusage", observation["swap"]["error"])
        self.assertEqual(observation["process_peak_rss_bytes"], 12345)

    def test_swap_preserves_byte_fields_and_encryption(self):
        swap = self.snapshot()["swap"]
        self.assertEqual(swap, {"total_bytes": 8192, "used_bytes": 2048, "available_bytes": 6144,
                               "page_size_bytes": 4096, "encrypted": True, "error": None})
        raw = struct.pack("=QQQII", 0, 0, 0, 16384, 0)
        self.assertEqual(self.snapshot(self.reader(swap=raw))["swap"]["encrypted"], False)

    def test_invalid_swap_does_not_invalidate_pressure(self):
        for raw in (b"short", struct.pack("=QQQII", 8, 4, 4, 0, 1),
                    struct.pack("=QQQII", 8, 4, 4, 4096, 2),
                    struct.pack("=QQQII", 8, 9, 0, 4096, 1),
                    struct.pack("=QQQII", 8, 0, 9, 4096, 1)):
            with self.subTest(raw=raw):
                result = self.snapshot(self.reader(swap=raw))
                self.assertIsNotNone(result["swap"]["error"])
                self.assertIsNone(result["swap"]["total_bytes"])
                self.assertEqual(result["pressure"]["state"], "normal")

    def test_macos_peak_rss_already_uses_bytes(self):
        result = self.snapshot(peak=1537)
        self.assertEqual(result["process_peak_rss_bytes"], 1537)
        self.assertIsNone(result["process_peak_rss_error"])

    def test_other_platforms_do_not_guess_peak_units_or_call_rusage(self):
        for platform_name in ("linux", "win32"):
            result = host.snapshot(sysctl_reader=self.reader(), platform_name=platform_name,
                                   rusage_reader=lambda: self.fail("Unexpected resource read"))
            self.assertIsNone(result["process_peak_rss_bytes"])
            self.assertIn("only on macOS", result["process_peak_rss_error"])

    def test_bad_rusage_values_and_errors_remain_null(self):
        for value in (-1, True, 2.5, float("nan")):
            result = self.snapshot(peak=value)
            self.assertIsNone(result["process_peak_rss_bytes"])
            self.assertIsNotNone(result["process_peak_rss_error"])
            json.dumps(result, allow_nan=False)
        def unavailable():
            raise OSError("Resource read failed")
        result = host.snapshot(sysctl_reader=self.reader(), rusage_reader=unavailable, platform_name="darwin")
        self.assertIsNone(result["process_peak_rss_bytes"])
        self.assertIn("Resource read failed", result["process_peak_rss_error"])

    def test_hardware_uses_separate_fields_and_bounded_reads(self):
        calls = []
        def reader(name, size):
            calls.append((name, size))
            return b"MacTest,1\0" if name == "hw.model" else struct.pack("=Q", 16 * 1024**3)
        result = host.hardware_snapshot(sysctl_reader=reader)
        self.assertEqual(result, {"model": "MacTest,1", "memory_bytes": 16 * 1024**3, "errors": {}})
        self.assertEqual(calls, [("hw.model", 256), ("hw.memsize", 8)])

    def test_invalid_hardware_preserves_per_field_errors(self):
        for raw in (b"", b"unterminated", b"\0", b"x" * 256 + b"\0", b"a\0b\0", b"\xff\0"):
            result = host.hardware_snapshot(sysctl_reader=lambda name, size: raw if name == "hw.model" else b"bad")
            self.assertIsNone(result["model"])
            self.assertIsNone(result["memory_bytes"])
            self.assertEqual(set(result["errors"]), {"model", "memory_bytes"})
        result = host.hardware_snapshot(sysctl_reader=lambda name, size: b"MacTest\0" if name == "hw.model" else bytes(8))
        self.assertEqual(set(result["errors"]), {"memory_bytes"})

    def test_missing_sysctl_symbol_returns_unknown_with_explicit_errors(self):
        host._sysctl_function.cache_clear()
        try:
            with patch.object(host.sys, "platform", "darwin"), patch.object(host.ctypes, "CDLL", return_value=object()):
                observation = host.snapshot(rusage_reader=lambda: 12345)
                hardware = host.hardware_snapshot()
            self.assertEqual(observation["pressure"]["state"], "unknown")
            self.assertIsNone(observation["pressure"]["dispatch_value"])
            self.assertIn("sysctlbyname interface is unavailable", observation["pressure"]["error"])
            self.assertIsNone(observation["swap"]["used_bytes"])
            self.assertIn("sysctlbyname interface is unavailable", observation["swap"]["error"])
            self.assertEqual(set(hardware["errors"]), {"model", "memory_bytes"})
            self.assertEqual(observation["process_peak_rss_bytes"], 12345)
        finally:
            host._sysctl_function.cache_clear()

    def test_native_reader_checks_errno_size_and_read_only_arguments(self):
        calls = []
        def native(name, buffer, size_pointer, new_value, new_size):
            calls.append((name, new_value, new_size))
            ctypes.memmove(buffer, b"abcd", 4)
            ctypes.cast(size_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 4
            return 0
        with patch.object(host.sys, "platform", "darwin"), patch.object(host, "_sysctl_function", return_value=native):
            self.assertEqual(host._sysctl_bytes("test.sensor", 4), b"abcd")
        self.assertEqual(calls, [(b"test.sensor", None, 0)])
        def invalid_size(name, buffer, size_pointer, new_value, new_size):
            ctypes.cast(size_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 5
            return 0
        with patch.object(host.sys, "platform", "darwin"), patch.object(host, "_sysctl_function", return_value=invalid_size):
            with self.assertRaisesRegex(ValueError, "invalid size"):
                host._sysctl_bytes("test.sensor", 4)
        def denied(*args):
            ctypes.set_errno(errno.EPERM)
            return -1
        with patch.object(host.sys, "platform", "darwin"), patch.object(host, "_sysctl_function", return_value=denied):
            with self.assertRaises(OSError) as raised:
                host._sysctl_bytes("test.sensor", 4)
        self.assertEqual(raised.exception.errno, errno.EPERM)


if __name__ == "__main__":
    unittest.main()
