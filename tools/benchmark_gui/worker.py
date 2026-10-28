import os
import subprocess
import sys

from PySide6.QtCore import QThread, Signal


class BenchmarkWorker(QThread):
    log_line = Signal(str)
    finished_with_code = Signal(int)

    def __init__(self, command, cwd, parent=None):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd
        self.process = None
        self._stop_requested = False

    def run(self):
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            self.log_line.emit("$ " + " ".join(self.command))
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert self.process.stdout is not None
            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    break
                self.log_line.emit(line.rstrip("\n"))
                if self._stop_requested and self.process.poll() is None:
                    self.process.terminate()
            code = self.process.wait()
            self.finished_with_code.emit(code)
        except Exception as exc:
            self.log_line.emit(f"[GUI 错误] {type(exc).__name__}: {exc}")
            self.finished_with_code.emit(-1)

    def stop(self):
        self._stop_requested = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
