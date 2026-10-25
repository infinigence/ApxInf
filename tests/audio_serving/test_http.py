from concurrent.futures import ThreadPoolExecutor
import http.client
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/apxinf/apxinf/serving"))
from audio_wav import encode_pcm16, inspect

BIN = os.environ.get("APXINF_AUDIO_BIN", str(ROOT / "target/debug/apxinf-audio-serve"))


def eventually(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


class RunningService:
    def __init__(self, task, delay=0, fault=None):
        directory = ROOT / "devlocal/audio-serving/runtime-tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.directory = Path(self.temporary.name)
        self.task = task
        self.model = "fixture-" + task
        manifest = {"schema":"apxinf-audio-manifest/1", "task":task, "model":self.model,
            "provider":"cpu_fixture", "family":"fixture", "precision":"fixture_pcm16",
            "model_dir":str(self.directory), "runtime":{"python":"fixture", "mlx":"none", "mlx_audio":"none", "mlx_audio_commit":"none", "packages":{}},
            "artifacts":[], "languages":["en"], "voices":{"fixture":str(self.directory / "fixture.safetensors")} if task == "tts" else {}}
        manifest_path = self.directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
        command = [BIN, "--manifest", str(manifest_path), "--python", sys.executable,
            "--listen", "127.0.0.1:0", "--media-parent", str(self.directory),
            "--allow-cpu-fixture", "--disable-pressure-check", "--fixture-delay-ms", str(delay)]
        env = dict(os.environ)
        if fault:
            command += ["--worker", str(ROOT / "tests/audio_serving/fault_worker.py")]
            env["APXINF_AUDIO_TEST_FAULT"] = fault
        self.stderr = open(self.directory / "stderr.log", "wb")
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.stderr, text=True, env=env)
        line = self.process.stdout.readline()
        if "listening " not in line:
            raise AssertionError("service did not start: " + line)
        self.host, port = line.strip().rsplit(" ", 1)[1].split(":")
        self.port = int(port)

    def request(self, path, body=None, content_type="application/json", method=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=12)
        connection.request(method or ("GET" if body is None else "POST"), path, body=body,
            headers={} if body is None else {"Content-Type":content_type})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def ready(self):
        return self.request("/readyz")[0] == 200

    def metrics(self):
        raw = self.request("/metrics")[2].decode()
        return {key:int(value) for key, value in (line.split() for line in raw.splitlines())}

    def multipart(self, fields=None, audio=None):
        if fields is None:
            fields = [("model", self.model)]
        if audio is None:
            audio = encode_pcm16(b"\x00\x00" * 320, 16000)
        boundary = "apxinf-original-audio-fixture"
        parts = []
        for name, value in fields:
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode())
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="recording.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode() + audio + b"\r\n")
        parts.append((f"--{boundary}--\r\n").encode())
        return b"".join(parts), "multipart/form-data; boundary=" + boundary

    def asr(self, fields=None, audio=None):
        body, content_type = self.multipart(fields, audio)
        return self.request("/v1/audio/transcriptions", body, content_type)

    def tts(self, **changes):
        value = {"model":self.model, "input":"A complete original fixture.", "voice":"fixture"}
        value.update(changes)
        return self.request("/v1/audio/speech", json.dumps(value).encode())

    def close(self, signum=signal.SIGTERM):
        if self.process.poll() is None:
            self.process.send_signal(signum)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise AssertionError("service did not reap and stop within five seconds")
        self.process.stdout.close()
        self.stderr.close()
        residual = [p for p in self.directory.iterdir() if p.name.startswith("apxinf-audio-")]
        if residual:
            raise AssertionError("media survived service shutdown: " + str(residual))
        self.temporary.cleanup()


class HttpAudio(unittest.TestCase):
    def start(self, task, **kwargs):
        service = RunningService(task, **kwargs)
        self.addCleanup(service.close)
        eventually(service.ready)
        return service

    def test_file_asr_and_resident_epoch(self):
        service = self.start("asr")
        first = json.loads(service.request("/readyz")[2])
        for _ in range(3):
            status, _, body = service.asr()
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body), {"text":"fixture transcript: 320 frames"})
        second = json.loads(service.request("/readyz")[2])
        self.assertEqual(first["worker_epoch"], second["worker_epoch"])
        self.assertEqual(service.metrics()["apxinf_audio_completed_total"], 3)
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)

    def test_complete_wav_tts(self):
        service = self.start("tts")
        for _ in range(2):
            status, headers, audio = service.tts()
            self.assertEqual(status, 200, audio)
            self.assertEqual(headers["content-type"], "audio/wav")
            info = inspect(audio, False)
            self.assertEqual((info.sample_rate, info.frames), (24000, 2400))
            samples = struct.unpack_from("<2400h", audio, info.data_offset)
            self.assertGreater(max(samples), 3000)
            self.assertLess(min(samples), -3000)
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)

    def test_wire_content_length_normalization_and_conflict(self):
        service = self.start("tts")
        body = json.dumps({"model":service.model,"input":"Original framing check.","voice":"fixture"}).encode()
        for lengths, expected in (([str(len(body)), str(len(body))], 200),
            ([str(len(body)), str(len(body) + 1)], 400)):
            connection = http.client.HTTPConnection(service.host, service.port, timeout=5)
            connection.putrequest("POST", "/v1/audio/speech")
            connection.putheader("Content-Type", "application/json")
            for length in lengths:
                connection.putheader("Content-Length", length)
            connection.endheaders(body)
            response = connection.getresponse()
            self.assertEqual(response.status, expected)
            response.read()
            connection.close()
        self.assertEqual(service.metrics()["apxinf_audio_completed_total"], 1)

    def test_public_validation_and_error_mapping(self):
        service = self.start("asr")
        cases = [ ([("model", "other")], None, 404, "model_not_found"),
            ([("model",service.model),("model",service.model)], None, 400,"invalid_request"),
            ([("model",service.model),("stream","true")],None,400,"unsupported_feature"),
            ([("model",service.model),("extra","x")],None,400,"invalid_request"),
            ([("model",service.model),("response_format","srt")],None,400,"unsupported_feature"),
            (None,b"not a WAV",400,"invalid_audio"),
            (None,encode_pcm16(b"\x00\x00",24000),400,"invalid_audio") ]
        for fields, audio, status, code in cases:
            with self.subTest(code=code,fields=fields):
                got,_,body = service.asr(fields,audio)
                self.assertEqual((got,json.loads(body)["error"]["code"]),(status,code))
        status,_,body = service.request("/v1/audio/speech",b"{}")
        self.assertEqual((status,json.loads(body)["error"]["code"]),(400,"task_mismatch"))
        self.assertEqual(service.metrics()["apxinf_audio_completed_total"],0)

    def test_tts_strict_fields(self):
        service = self.start("tts")
        for changes, expected in [({"stream":False},"unsupported_feature"), ({"speed":1},"unsupported_feature"),
            ({"extra":1},"invalid_request"),({"voice":"other"},"unsupported_feature"),
            ({"response_format":None},"invalid_request"),({"input":"x" * 4097},"payload_too_large")]:
            with self.subTest(changes=changes):
                status,_,body = service.tts(**changes)
                self.assertNotEqual(status,200)
                self.assertEqual(json.loads(body)["error"]["code"],expected)
        raw = ('{"model":"' + service.model + '","input":"one","input":"two","voice":"fixture"}').encode()
        self.assertEqual(service.request("/v1/audio/speech",raw)[0],400)

    def test_queue_limit(self):
        service = self.start("asr",delay=1000)
        with ThreadPoolExecutor(max_workers=6) as executor:
            first = executor.submit(service.asr)
            eventually(lambda: service.metrics()["apxinf_audio_active"] == 1)
            rest = []
            for count in range(1,5):
                rest.append(executor.submit(service.asr))
                eventually(lambda: service.metrics()["apxinf_audio_queued"] == count)
            rest.append(executor.submit(service.asr))
            results = [first.result()] + [future.result() for future in rest]
        self.assertEqual(sorted(status for status,_,_ in results),[200,200,200,200,200,429])
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)

    def test_queue_deadline(self):
        service = self.start("asr",delay=6500)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(service.asr)
            eventually(lambda: service.metrics()["apxinf_audio_active"] == 1)
            status,_,body = service.asr()
            self.assertEqual((status,json.loads(body)["error"]["code"]),(504,"deadline_exceeded"))
            self.assertEqual(first.result()[0],200)

    def test_response_send_deadline_starts_after_execution(self):
        service = self.start("asr",delay=16000)
        body, content_type = service.multipart()
        connection = http.client.HTTPConnection(service.host,service.port,timeout=25)
        connection.request("POST","/v1/audio/transcriptions",body=body,headers={"Content-Type":content_type})
        response = connection.getresponse()
        self.assertEqual(response.status,200)
        self.assertEqual(json.loads(response.read()),{"text":"fixture transcript: 320 frames"})
        connection.close()

    def test_disconnect_cancels_and_releases(self):
        service = self.start("asr",delay=2000)
        body, content_type = service.multipart()
        client = socket.create_connection((service.host,service.port))
        client.sendall((f"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n\r\n").encode()+body)
        eventually(lambda: service.metrics()["apxinf_audio_active"] == 1)
        client.close()
        eventually(lambda: service.metrics()["apxinf_audio_active"] == 0,timeout=3)
        self.assertEqual(service.metrics()["apxinf_audio_completed_total"],0)
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)
        self.assertTrue(service.ready())

    def test_upload_limit_and_declared_body_bound(self):
        service = self.start("asr")
        clients = []
        for _ in range(2):
            client = socket.create_connection((service.host,service.port))
            client.sendall(b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=fixture\r\nContent-Length: 1024\r\n\r\nx")
            clients.append(client)
        self.addCleanup(lambda: [client.close() for client in clients])
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 6144)
        self.assertEqual(service.asr()[0],503)
        for client in clients:
            client.close()
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)
        client = socket.create_connection((service.host,service.port))
        client.sendall(b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=fixture\r\nContent-Length: 99999999\r\n\r\n")
        self.assertIn(b"413",client.recv(4096))
        client.close()
        self.assertEqual(service.metrics()["apxinf_audio_media_reserved_bytes"],0)

    def test_worker_protocol_faults_reap_and_fence(self):
        for fault in ("missing_terminal","missing_settlement","double_terminal","stale_epoch"):
            with self.subTest(fault=fault):
                service = self.start("asr",fault=fault)
                status,_,body = service.asr()
                self.assertEqual((status,json.loads(body)["error"]["code"]),(500,"protocol_fault"))
                eventually(lambda: service.metrics()["apxinf_audio_worker_reaped"] == 1)
                self.assertFalse(service.ready())
                self.assertEqual(service.asr()[0],503)
                eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)

    def test_nonregular_output_reaps_without_blocking_the_actor(self):
        service = self.start("tts", fault="fifo_output")
        connection = http.client.HTTPConnection(service.host, service.port, timeout=2)
        self.addCleanup(connection.close)
        body = json.dumps({"model":service.model,"input":"A complete original request.","voice":"fixture"}).encode()
        connection.request("POST", "/v1/audio/speech", body, {"Content-Type":"application/json"})
        response = connection.getresponse()
        self.assertEqual((response.status, json.loads(response.read())["error"]["code"]), (500, "protocol_fault"))
        eventually(lambda: service.metrics()["apxinf_audio_worker_reaped"] == 1)
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)
        self.assertFalse(service.ready())

    def test_idle_eof_and_partial_frame(self):
        service = self.start("asr",fault="idle_eof")
        eventually(lambda: service.metrics()["apxinf_audio_worker_reaped"] == 1)
        self.assertFalse(service.ready())
        partial = RunningService("asr",fault="partial_ready")
        self.addCleanup(partial.close)
        eventually(lambda: partial.metrics()["apxinf_audio_worker_reaped"] == 1,timeout=4)
        self.assertFalse(partial.ready())

    def test_stderr_saturation_does_not_block_protocol(self):
        service = self.start("asr",fault="stderr_flood")
        self.assertEqual(service.asr()[0],200)

    def test_sigint_and_sigterm_reap_during_work(self):
        for signum in (signal.SIGINT,signal.SIGTERM):
            with self.subTest(signal=signum):
                service = self.start("asr",delay=10000)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    pending = executor.submit(service.asr)
                    eventually(lambda: service.metrics()["apxinf_audio_active"] == 1)
                    service.process.send_signal(signum)
                    service.process.wait(timeout=4)
                    try:
                        pending.result()
                    except (http.client.HTTPException, ConnectionError):
                        pass
                self.assertEqual(service.process.returncode,0)

    def test_signals_during_loading_reap(self):
        for signum in (signal.SIGINT,signal.SIGTERM):
            service = RunningService("asr",fault="partial_ready")
            self.addCleanup(service.close)
            self.assertFalse(service.ready())
            service.process.send_signal(signum)
            service.process.wait(timeout=4)
            self.assertEqual(service.process.returncode,0)

    def test_cancelled_attempt_retains_media_until_settlement(self):
        service = self.start("asr",delay=2000,fault="slow_settlement")
        body, content_type = service.multipart()
        client = socket.create_connection((service.host,service.port))
        client.sendall((f"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n\r\n").encode()+body)
        eventually(lambda: service.metrics()["apxinf_audio_active"] == 1)
        client.close()
        time.sleep(0.1)
        metrics = service.metrics()
        self.assertEqual(metrics["apxinf_audio_active"],1)
        self.assertGreater(metrics["apxinf_audio_media_reserved_bytes"],0)
        self.assertEqual(metrics["apxinf_audio_completed_total"],0)
        eventually(lambda: service.metrics()["apxinf_audio_active"] == 0)
        eventually(lambda: service.metrics()["apxinf_audio_media_reserved_bytes"] == 0)

    def test_changed_manifest_fails_before_provider_construction(self):
        directory = ROOT / "devlocal/audio-serving/runtime-tests"
        directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary) / "manifest.json"
            original = b'{"provider":"cpu_fixture"}'
            path.write_bytes(b'{"provider":"a_provider_that_must_never_load"}')
            result = subprocess.run([sys.executable,str(ROOT / "python/apxinf/apxinf/serving/audio_worker.py"),
                "--manifest",str(path),"--manifest-revision",hashlib.sha256(original).hexdigest(),
                "--epoch","11111111-1111-4111-8111-111111111111","--media-dir",temporary,
                "--allow-cpu-fixture","--disable-pressure-check"], input=b"",capture_output=True,timeout=4)
            self.assertEqual(result.returncode,2)
            self.assertEqual(result.stdout,b"")
            self.assertIn(b"manifest changed before provider construction",result.stderr)

    def test_startup_failure_with_open_stdin_exits_without_interpreter_abort(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_bytes(b'{}')
            process = subprocess.Popen([sys.executable,str(ROOT / "python/apxinf/apxinf/serving/audio_worker.py"),
                "--manifest",str(path),"--manifest-revision","0" * 64,
                "--epoch","11111111-1111-4111-8111-111111111111","--media-dir",temporary,
                "--allow-cpu-fixture","--disable-pressure-check"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                self.assertEqual(process.wait(timeout=4), 2)
                diagnostics = process.stderr.read()
                self.assertIn(b"manifest changed before provider construction", diagnostics)
                self.assertNotIn(b"Fatal Python error", diagnostics)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
