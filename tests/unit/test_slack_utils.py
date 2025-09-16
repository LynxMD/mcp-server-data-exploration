import os
from unittest import mock

import mcp_server_ds.slack_utils as slack_utils


def _set_env(env: dict[str, str]):
    os.environ.pop("MCP_SLACK_ALERTS_ENABLED", None)
    os.environ.pop("MCP_SLACK_WEBHOOK_URL", None)
    os.environ.pop("MCP_SLACK_MEMORY_THRESHOLD", None)
    os.environ.pop("MCP_SLACK_VERIFY_SSL", None)
    for k, v in env.items():
        os.environ[k] = v


def test_no_alert_when_disabled(monkeypatch):
    _set_env(
        {
            "MCP_SLACK_ALERTS_ENABLED": "false",
            "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "MCP_SLACK_MEMORY_THRESHOLD": "10",
        }
    )

    attempted, status = slack_utils.send_slack_alert_if_needed(
        50.0, 20.0, "TTLInMemoryDataManager", 123
    )
    assert attempted is False
    assert status is None


def test_no_alert_without_webhook(monkeypatch):
    _set_env(
        {
            "MCP_SLACK_ALERTS_ENABLED": "true",
            "MCP_SLACK_MEMORY_THRESHOLD": "10",
        }
    )

    attempted, status = slack_utils.send_slack_alert_if_needed(
        90.0, 20.0, "TTLInMemoryDataManager", 123
    )
    assert attempted is False
    assert status is None


def test_no_alert_below_threshold(monkeypatch):
    _set_env(
        {
            "MCP_SLACK_ALERTS_ENABLED": "true",
            "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "MCP_SLACK_MEMORY_THRESHOLD": "95",
        }
    )

    attempted, status = slack_utils.send_slack_alert_if_needed(
        80.0, 20.0, "TTLInMemoryDataManager", 123
    )
    assert attempted is False
    assert status is None


def test_alert_sends_with_system_ssl(monkeypatch):
    _set_env(
        {
            "MCP_SLACK_ALERTS_ENABLED": "true",
            "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "MCP_SLACK_MEMORY_THRESHOLD": "10",
            "MCP_SLACK_VERIFY_SSL": "true",
        }
    )

    # Force certifi unavailable path
    monkeypatch.setattr(slack_utils, "_CERTIFI_AVAILABLE", False)

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=5, context=None):
        assert timeout == 5
        assert context is not None
        return DummyResponse()

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        attempted, status = slack_utils.send_slack_alert_if_needed(
            99.0, 40.0, "TTLInMemoryDataManager", 321
        )
        assert attempted is True
        assert status == 200


def test_alert_sends_with_certifi_ssl(monkeypatch):
    _set_env(
        {
            "MCP_SLACK_ALERTS_ENABLED": "true",
            "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "MCP_SLACK_MEMORY_THRESHOLD": "10",
            "MCP_SLACK_VERIFY_SSL": "true",
        }
    )

    # Force certifi available path
    monkeypatch.setattr(slack_utils, "_CERTIFI_AVAILABLE", True)

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=5, context=None):
        assert timeout == 5
        assert context is not None
        return DummyResponse()

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        attempted, status = slack_utils.send_slack_alert_if_needed(
            91.0, 40.0, "TTLInMemoryDataManager"
        )
        assert attempted is True
        assert status == 200


def test_alert_sends_with_verify_off(monkeypatch):
    _set_env(
        {
            "MCP_SLACK_ALERTS_ENABLED": "true",
            "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "MCP_SLACK_MEMORY_THRESHOLD": "10",
            "MCP_SLACK_VERIFY_SSL": "false",
        }
    )

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=5, context=None):
        assert timeout == 5
        # context can be None or unverified; we only check that call succeeds
        return DummyResponse()

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        attempted, status = slack_utils.send_slack_alert_if_needed(
            92.0, 40.0, "TTLInMemoryDataManager"
        )
        assert attempted is True
        assert status == 200


def test_alert_handles_exception(monkeypatch):
    _set_env(
        {
            "MCP_SLACK_ALERTS_ENABLED": "true",
            "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "MCP_SLACK_MEMORY_THRESHOLD": "10",
        }
    )

    with mock.patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
        attempted, status = slack_utils.send_slack_alert_if_needed(
            95.0, 40.0, "TTLInMemoryDataManager"
        )
        assert attempted is True
        assert status is None
