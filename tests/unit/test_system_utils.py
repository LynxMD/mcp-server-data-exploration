"""
Unit tests for System Utils

Tests the system utilities for logging and monitoring.
"""

from unittest.mock import patch, MagicMock

from mcp_server_ds.system_utils import log_system_status


class TestSystemUtils:
    """Test suite for system utilities."""

    def test_log_system_status_success(self):
        """Test successful system status logging."""
        with (
            patch("psutil.virtual_memory") as mock_vm,
            patch("psutil.disk_usage") as mock_du,
            patch("psutil.Process") as mock_process,
            patch(
                "mcp_server_ds.system_utils.send_slack_alert_if_needed"
            ) as mock_slack,
            patch("mcp_server_ds.system_utils.logger") as mock_logger,
        ):
            # Mock system resources
            mock_vm.return_value.percent = 75.5
            mock_vm.return_value.used = 8 * 1024**3  # 8GB
            mock_vm.return_value.total = 16 * 1024**3  # 16GB

            mock_du.return_value.percent = 60.0
            mock_du.return_value.used = 100 * 1024**3  # 100GB
            mock_du.return_value.total = 500 * 1024**3  # 500GB

            # Mock process
            mock_process_instance = MagicMock()
            mock_process_instance.memory_info.return_value.rss = 512 * 1024**2  # 512MB
            mock_process.return_value = mock_process_instance

            # Call the function
            log_system_status("TestManager", include_process_rss=True)

            # Verify logging was called
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "DataManager=TestManager" in log_message
            assert "RAM used=75.5%" in log_message
            assert "Disk used=60.0%" in log_message
            assert "MCP Process RSS=512MB" in log_message

            # Verify Slack alert was called
            mock_slack.assert_called_once_with(75.5, 60.0, "TestManager", 512)

    def test_log_system_status_without_process_rss(self):
        """Test system status logging without process RSS."""
        with (
            patch("psutil.virtual_memory") as mock_vm,
            patch("psutil.disk_usage") as mock_du,
            patch(
                "mcp_server_ds.system_utils.send_slack_alert_if_needed"
            ) as mock_slack,
            patch("mcp_server_ds.system_utils.logger") as mock_logger,
        ):
            # Mock system resources
            mock_vm.return_value.percent = 50.0
            mock_vm.return_value.used = 4 * 1024**3  # 4GB
            mock_vm.return_value.total = 8 * 1024**3  # 8GB

            mock_du.return_value.percent = 30.0
            mock_du.return_value.used = 50 * 1024**3  # 50GB
            mock_du.return_value.total = 200 * 1024**3  # 200GB

            # Call the function without process RSS
            log_system_status("TestManager", include_process_rss=False)

            # Verify logging was called
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "DataManager=TestManager" in log_message
            assert "RAM used=50.0%" in log_message
            assert "Disk used=30.0%" in log_message
            assert "MCP Process RSS" not in log_message

            # Verify Slack alert was called with None for process RSS
            mock_slack.assert_called_once_with(50.0, 30.0, "TestManager", None)

    def test_log_system_status_process_rss_exception(self):
        """Test system status logging when process RSS fails (line 20-21)."""
        with (
            patch("psutil.virtual_memory") as mock_vm,
            patch("psutil.disk_usage") as mock_du,
            patch("psutil.Process") as mock_process,
            patch(
                "mcp_server_ds.system_utils.send_slack_alert_if_needed"
            ) as mock_slack,
            patch("mcp_server_ds.system_utils.logger") as mock_logger,
        ):
            # Mock system resources
            mock_vm.return_value.percent = 40.0
            mock_vm.return_value.used = 2 * 1024**3  # 2GB
            mock_vm.return_value.total = 8 * 1024**3  # 8GB

            mock_du.return_value.percent = 25.0
            mock_du.return_value.used = 25 * 1024**3  # 25GB
            mock_du.return_value.total = 100 * 1024**3  # 100GB

            # Mock process to raise exception
            mock_process.side_effect = Exception("Process access denied")

            # Call the function
            log_system_status("TestManager", include_process_rss=True)

            # Verify logging was called without process RSS
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "DataManager=TestManager" in log_message
            assert "RAM used=40.0%" in log_message
            assert "Disk used=25.0%" in log_message
            assert "MCP Process RSS" not in log_message

            # Verify Slack alert was called with None for process RSS
            mock_slack.assert_called_once_with(40.0, 25.0, "TestManager", None)

    def test_log_system_status_exception_handling(self):
        """Test system status logging exception handling (line 41-42)."""
        with (
            patch("psutil.virtual_memory") as mock_vm,
            patch("mcp_server_ds.system_utils.logger") as mock_logger,
        ):
            # Mock psutil to raise exception
            mock_vm.side_effect = Exception("System access denied")

            # Call the function
            log_system_status("TestManager")

            # Verify exception was logged
            mock_logger.debug.assert_called_once()
            debug_message = mock_logger.debug.call_args[0][0]
            assert "Failed to log system status" in debug_message
            assert "System access denied" in debug_message
