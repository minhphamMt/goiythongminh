from __future__ import annotations

import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from app.config import BASE_DIR, settings


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    process: subprocess.Popen | None = None

    def start(self) -> None:
        print(f"Starting {self.name}: {' '.join(self.command)}")
        self.process = subprocess.Popen(self.command, cwd=str(BASE_DIR))

    def poll(self) -> int | None:
        if self.process is None:
            return None
        return self.process.poll()

    def terminate(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()

    def kill(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.kill()

    def wait(self, timeout: float | None = None) -> int | None:
        if self.process is None:
            return None
        return self.process.wait(timeout=timeout)


class ServiceSupervisor:
    def __init__(self) -> None:
        self.shutdown_requested = False
        self.exit_code = 0
        self.processes = self._build_processes()

    @staticmethod
    def _python_module_command(module_name: str) -> list[str]:
        return [sys.executable, "-m", module_name]

    def _build_processes(self) -> list[ManagedProcess]:
        processes = [
            ManagedProcess("embedding-api", self._python_module_command("app.api")),
        ]

        if settings.run_metadata_worker:
            processes.append(
                ManagedProcess("metadata-worker", self._python_module_command("app.workers.metadata_worker"))
            )

        if settings.run_audio_worker:
            processes.append(
                ManagedProcess("audio-worker", self._python_module_command("app.workers.audio_worker"))
            )

        return processes

    def request_shutdown(self, signum: int | None = None) -> None:
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        if signum is not None:
            print(f"Received signal {signum}, shutting down embedding service...")

    def start_all(self) -> None:
        for managed in self.processes:
            managed.start()

    def monitor(self) -> None:
        while not self.shutdown_requested:
            for managed in self.processes:
                code = managed.poll()
                if code is not None:
                    print(f"{managed.name} exited unexpectedly with code {code}")
                    self.exit_code = code if code != 0 else 1
                    self.request_shutdown()
                    return
            time.sleep(settings.supervisor_poll_seconds)

    def stop_all(self) -> None:
        for managed in reversed(self.processes):
            managed.terminate()

        deadline = time.time() + settings.supervisor_stop_timeout_seconds
        for managed in reversed(self.processes):
            remaining = max(deadline - time.time(), 0)
            try:
                managed.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"Force killing {managed.name}")
                managed.kill()
                try:
                    managed.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass


def main() -> None:
    supervisor = ServiceSupervisor()

    def _handle_signal(signum, _frame) -> None:
        supervisor.request_shutdown(signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    supervisor.start_all()
    try:
        supervisor.monitor()
    finally:
        supervisor.stop_all()

    raise SystemExit(supervisor.exit_code)


if __name__ == "__main__":
    main()
