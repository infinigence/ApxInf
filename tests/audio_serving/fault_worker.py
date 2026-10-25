import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/apxinf/apxinf/serving"))
import audio_worker
from audio_contracts import Ready, Terminal, ResourcesReleased

scenario = os.environ["APXINF_AUDIO_TEST_FAULT"]
original = audio_worker.Worker.emit


def emit(self, message):
    if isinstance(message, Ready) and scenario == "partial_ready":
        os.write(self.protocol_fd, b'{"protocol":')
        time.sleep(30)
        return
    if isinstance(message, Ready) and scenario == "stderr_flood":
        for _ in range(1024):
            try:
                os.write(2, b"x" * 4096)
            except BlockingIOError:
                pass
    if isinstance(message, Terminal) and scenario == "missing_terminal":
        return
    if isinstance(message, ResourcesReleased) and scenario == "missing_settlement":
        return
    if isinstance(message, ResourcesReleased) and scenario == "slow_settlement":
        time.sleep(0.6)
    if isinstance(message, Terminal) and scenario == "stale_epoch":
        self.args.epoch = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    original(self, message)
    if isinstance(message, Terminal) and scenario == "double_terminal":
        original(self, message)
    if isinstance(message, Ready) and scenario == "idle_eof":
        threading.Thread(target=lambda: (time.sleep(0.2), os._exit(3)), daemon=True).start()


audio_worker.Worker.emit = emit
audio_worker.main()
