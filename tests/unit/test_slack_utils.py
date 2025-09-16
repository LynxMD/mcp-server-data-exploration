"""
Unit tests for Slack Utils

Tests the Slack notification utilities.
"""

import os
from unittest.mock import patch, MagicMock

from mcp_server_ds.slack_utils import send_slack_alert_if_needed


class TestSlackUtils:
    """Test suite for Slack utilities."""

    def test_send_slack_alert_disabled(self):
        """Test Slack alert when disabled."""
        with patch.dict(os.environ, {"MCP_SLACK_ALERTS_ENABLED": "false"}):
            # Should not send alert when disabled
            result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", 1024)
            assert result == (False, None)

    def test_send_slack_alert_no_webhook(self):
        """Test Slack alert when no webhook URL."""
        with patch.dict(os.environ, {"MCP_SLACK_ALERTS_ENABLED": "true"}, clear=True):
            # Should not send alert when no webhook URL
            result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", 1024)
            assert result == (False, None)

    def test_send_slack_alert_below_threshold(self):
        """Test Slack alert when below threshold."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            # Should not send alert when below threshold
            result = send_slack_alert_if_needed(85.0, 80.0, "TestManager", 1024)
            assert result == (False, None)

    def test_send_slack_alert_above_threshold(self):
        """Test Slack alert when above threshold."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_urlopen.return_value.__enter__.return_value = mock_response

                # Should send alert when above threshold
                result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", 1024)
                assert result == (True, 200)

                # Verify request was made
                mock_urlopen.assert_called_once()

    def test_send_slack_alert_invalid_threshold(self):
        """Test Slack alert with invalid threshold (lines 37-38)."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "invalid",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_urlopen.return_value.__enter__.return_value = mock_response

                # Should use default threshold (90.0) when invalid
                result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", 1024)
                assert result == (True, 200)

                # Verify request was made (95.0 > 90.0 default)
                mock_urlopen.assert_called_once()

    def test_send_slack_alert_ssl_context_with_certifi(self):
        """Test SSL context creation with certifi (lines 65-73)."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200  # Set status attribute
                mock_response.code = 200  # Set code attribute
                mock_urlopen.return_value.__enter__.return_value = mock_response

                # Mock certifi to be available
                with patch("certifi.where", return_value="/path/to/certifi.pem"):
                    with patch("ssl.create_default_context") as mock_ssl_context:
                        result = send_slack_alert_if_needed(
                            95.0, 80.0, "TestManager", 1024
                        )
                        assert result == (True, 200)
                        # Verify SSL context was created with certifi
                        mock_ssl_context.assert_called_once()

    def test_send_slack_alert_return_code_exception_handling(self):
        """Test return code exception handling (lines 130-131)."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                # Mock status/code to return something that can't be converted to int
                mock_response.status = "invalid_code"
                mock_response.code = "invalid_code"
                mock_urlopen.return_value.__enter__.return_value = mock_response

                result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", 1024)
                # Should return True, None when code conversion fails
                assert result == (True, None)

    def test_send_slack_alert_without_process_rss(self):
        """Test Slack alert without process RSS."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_urlopen.return_value.__enter__.return_value = mock_response

                # Should send alert without process RSS
                result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", None)
                assert result == (True, 200)

                # Verify request was made
                mock_urlopen.assert_called_once()

    def test_send_slack_alert_request_failure(self):
        """Test Slack alert when request fails."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = Exception("Network error")

                # Should handle request failure gracefully
                result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", 1024)
                assert result == (True, None)

                # Verify request was attempted
                mock_urlopen.assert_called_once()

    def test_send_slack_alert_http_error(self):
        """Test Slack alert when HTTP request fails."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = Exception("HTTP Error 400: Bad Request")

                # Should handle HTTP error gracefully
                result = send_slack_alert_if_needed(95.0, 80.0, "TestManager", 1024)
                assert result == (True, None)

                # Verify request was attempted
                mock_urlopen.assert_called_once()

    def test_send_slack_alert_ssl_context_without_certifi(self):
        """Test SSL context creation without certifi (lines 65-73)."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_urlopen.return_value.__enter__.return_value = mock_response

                # Mock certifi to not be available
                with patch("mcp_server_ds.slack_utils._CERTIFI_AVAILABLE", False):
                    with patch("ssl.create_default_context") as mock_ssl_context:
                        with patch("sys.stderr") as mock_stderr:
                            result = send_slack_alert_if_needed(
                                95.0, 80.0, "TestManager", 1024
                            )
                            assert result == (True, 200)
                            # Verify SSL context was created without certifi
                            mock_ssl_context.assert_called_once()
                            # Verify debug message was printed
                            mock_stderr.write.assert_called()

    def test_send_slack_alert_ssl_verify_off(self):
        """Test SSL context creation with SSL verification disabled (lines 72-73)."""
        with patch.dict(
            os.environ,
            {
                "MCP_SLACK_ALERTS_ENABLED": "true",
                "MCP_SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
                "MCP_SLACK_MEMORY_THRESHOLD": "90.0",
                "MCP_SLACK_VERIFY_SSL": "false",
            },
        ):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_urlopen.return_value.__enter__.return_value = mock_response

                with patch("ssl._create_unverified_context") as mock_unverified_ssl:
                    with patch("sys.stderr") as mock_stderr:
                        result = send_slack_alert_if_needed(
                            95.0, 80.0, "TestManager", 1024
                        )
                        assert result == (True, 200)
                        # Verify unverified SSL context was created
                        mock_unverified_ssl.assert_called_once()
                        # Verify debug message was printed
                        mock_stderr.write.assert_called()
