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
                self._terminate_then_kill()

            code = self.process.wait()
            self.finished_with_code.emit(code)
        except Exception as exc:
            self.log_line.emit(f"[GUI 错误] {type(exc).__name__}: {exc}")
            self.finished_with_code.emit(-1)

    def _terminate_then_kill(self):
        """先礼貌 terminate，等待 3 秒，仍不退出再 kill。"""
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.log_line.emit("[GUI] 子进程未在 3 秒内退出，执行 kill。")
            self.process.kill()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.log_line.emit("[GUI] kill 后仍未退出，请手动检查进程。")

    def stop(self):
        self._stop_requested = True
        self._terminate_then_kill()