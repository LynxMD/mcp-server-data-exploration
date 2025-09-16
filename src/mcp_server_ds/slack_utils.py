import json
import os
import sys
import urllib.request
import ssl

_CERTIFI_AVAILABLE = False
_certifi_mod: object | None = None
try:  # Prefer certifi CA bundle if available to avoid SSL issues
    import certifi as _certifi

    _certifi_mod = _certifi
    _CERTIFI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _CERTIFI_AVAILABLE = False


def send_slack_alert_if_needed(
    memory_percent: float,
    disk_percent: float,
    data_manager_name: str,
    process_rss_mb: int | None = None,
) -> tuple[bool, int | None]:
    """Send a Slack alert via webhook if configured and threshold exceeded.

    Returns a tuple: (attempted, status_code). If not attempted, status_code is None.
    """
    alerts_enabled = os.environ.get("MCP_SLACK_ALERTS_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    webhook_url = os.environ.get("MCP_SLACK_WEBHOOK_URL")
    threshold_pct_str = os.environ.get("MCP_SLACK_MEMORY_THRESHOLD", "90")
    try:
        threshold_pct = float(threshold_pct_str)
    except ValueError:
        threshold_pct = 90.0

    should_alert = (
        alerts_enabled and bool(webhook_url) and (memory_percent >= threshold_pct)
    )
    print(
        f"[MCP-Server][Slack] enabled={alerts_enabled} vm={memory_percent:.1f}% "
        f"threshold={threshold_pct:.1f}% has_webhook={'yes' if webhook_url else 'no'}",
        file=sys.stderr,
        flush=True,
    )

    if not should_alert:
        return False, None

    verify_ssl = os.environ.get("MCP_SLACK_VERIFY_SSL", "true").lower() == "true"
    ssl_ctx: ssl.SSLContext | None
    if verify_ssl:
        if _CERTIFI_AVAILABLE:
            cafile = getattr(_certifi_mod, "where")()
            ssl_ctx = ssl.create_default_context(cafile=cafile)
            print(
                "[MCP-Server][Slack] SSL verify=on (certifi)",
                file=sys.stderr,
                flush=True,
            )
        else:
            ssl_ctx = ssl.create_default_context()
            print(
                "[MCP-Server][Slack] SSL verify=on (system CA)",
                file=sys.stderr,
                flush=True,
            )
    else:
        ssl_ctx = ssl._create_unverified_context()
        print(
            "[MCP-Server][Slack] SSL verify=OFF (unverified)",
            file=sys.stderr,
            flush=True,
        )

    payload = {
        "text": f"🚨 MCP Server High Memory Usage ({memory_percent:.1f}%)",
        "attachments": [
            {
                "color": "danger",
                "fields": [
                    {
                        "title": "Server",
                        "value": "Data Science Explorer",
                        "short": True,
                    },
                    {"title": "DataManager", "value": data_manager_name, "short": True},
                    {
                        "title": "RAM Used",
                        "value": f"{memory_percent:.1f}%",
                        "short": True,
                    },
                    {
                        "title": "Disk Used",
                        "value": f"{disk_percent:.1f}%",
                        "short": True,
                    },
                    {
                        "title": "MCP Process RSS",
                        "value": f"{process_rss_mb}MB"
                        if process_rss_mb is not None
                        else "n/a",
                        "short": True,
                    },
                ],
            }
        ],
    }

    try:
        req = urllib.request.Request(
            webhook_url or "",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        print("[MCP-Server][Slack] sending alert...", file=sys.stderr, flush=True)
        with urllib.request.urlopen(req, timeout=5, context=ssl_ctx) as resp:
            code = getattr(resp, "status", None) or getattr(resp, "code", None)
            print(
                f"[MCP-Server][Slack] sent, status={code}",
                file=sys.stderr,
                flush=True,
            )
            try:
                return True, int(code) if code is not None else None
            except Exception:
                return True, None
    except Exception as slack_err:  # pragma: no cover
        print(
            f"[MCP-Server][Slack] send failed: {slack_err}",
            file=sys.stderr,
            flush=True,
        )
        return True, None
