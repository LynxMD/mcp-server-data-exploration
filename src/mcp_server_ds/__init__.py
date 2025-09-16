from . import server
from importlib.metadata import version, PackageNotFoundError


def main():
    """Main entry point for the package."""
    server.main()


# Package metadata helpers
try:
    __version__ = version("mcp-server-ds")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+dev"

# Server identity (keep in sync with server title)
SERVER_NAME = "Data Science Explorer 🔬"

# Public API
__all__ = ["main", "server", "__version__", "SERVER_NAME"]
