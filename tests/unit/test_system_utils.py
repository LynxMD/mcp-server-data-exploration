from unittest import mock

import mcp_server_ds.system_utils as system_utils


def test_log_system_status_basic(monkeypatch):
    class VM:
        percent = 80.5
        used = 8 * 1024 * 1024 * 1024 // 2  # fake
        total = 16 * 1024 * 1024 * 1024

    class DU:
        percent = 10.1
        used = 10 * 1024 * 1024 * 1024
        total = 460 * 1024 * 1024 * 1024

    class Proc:
        class Mem:
            rss = 123 * 1024 * 1024

        def memory_info(self):
            return Proc.Mem()

    monkeypatch.setattr(system_utils.psutil, "virtual_memory", lambda: VM())
    monkeypatch.setattr(system_utils.psutil, "disk_usage", lambda _: DU())
    monkeypatch.setattr(system_utils.psutil, "Process", lambda: Proc())

    with mock.patch("mcp_server_ds.system_utils.send_slack_alert_if_needed") as m:
        system_utils.log_system_status("TTLInMemoryDataManager")
        m.assert_called_once()


def test_log_system_status_handles_exception(monkeypatch):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(system_utils.psutil, "virtual_memory", boom)

    # Should not raise
    system_utils.log_system_status("TTLInMemoryDataManager")
