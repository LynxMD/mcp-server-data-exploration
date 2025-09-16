import sys
import psutil
import logging

from .slack_utils import send_slack_alert_if_needed

logger = logging.getLogger(__name__)


def log_system_status(data_manager_name: str, include_process_rss: bool = True) -> None:
    """Log DataManager and system resource stats, and send Slack alert if configured."""
    try:
        vm = psutil.virtual_memory()
        du = psutil.disk_usage("/")
        process_rss_mb: int | None = None
        if include_process_rss:
            try:
                current_process = psutil.Process()
                process_rss_mb = current_process.memory_info().rss // (1024**2)
            except Exception:
                process_rss_mb = None

        msg = (
            f"DataManager={data_manager_name} | RAM used={vm.percent:.1f}% "
            f"({vm.used // (1024**2)}MB/{vm.total // (1024**2)}MB) | "
            f"Disk used={du.percent:.1f}% "
            f"({du.used // (1024**3)}GB/{du.total // (1024**3)}GB)"
            + (
                f" | MCP Process RSS={process_rss_mb}MB"
                if process_rss_mb is not None
                else ""
            )
        )
        logger.info(msg)
        print(f"[MCP-Server] {msg}", file=sys.stderr, flush=True)

        # Slack alert if needed
        send_slack_alert_if_needed(
            vm.percent, du.percent, data_manager_name, process_rss_mb
        )
    except Exception as exc:  # pragma: no cover
        logger.debug(f"Failed to log system status: {exc}")
