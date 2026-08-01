import json
from types import SimpleNamespace
import time

from iic.telemetry import ResourceMonitor, resource_snapshot
import iic.telemetry as telemetry


def test_resource_snapshot_records_host_and_nvidia_rows(monkeypatch):
    def fake_run(command, **_kwargs):
        query = command[1]
        if query.startswith("--query-gpu="):
            output = "0, GPU-a, Tesla V100, 32768, 1024, 72, 18, 61, 190\n"
        else:
            output = "GPU-a, 42, python, 1024\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)
    snapshot = resource_snapshot()

    assert snapshot["host"]["logical_cpu_count"] is not None
    assert snapshot["nvidia_gpus"]["available"] is True
    assert snapshot["nvidia_gpus"]["rows"][0]["memory.used"] == "1024"
    assert snapshot["nvidia_processes"]["rows"][0]["pid"] == "42"


def test_resource_monitor_writes_append_only_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "resource_snapshot",
        lambda: {"recorded_at": "now", "host": {}, "nvidia_gpus": {}},
    )
    path = tmp_path / "resources.jsonl"
    monitor = ResourceMonitor(path, interval_seconds=0.01)
    monitor.start()
    time.sleep(0.035)
    monitor.stop()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) >= 2
    assert all(row["recorded_at"] == "now" for row in rows)
