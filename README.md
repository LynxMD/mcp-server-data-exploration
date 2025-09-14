# MCP Server for Data Exploration

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/package%20manager-uv-orange.svg)](https://github.com/astral-sh/uv)
[![FastMCP](https://img.shields.io/badge/framework-FastMCP%202.0-green.svg)](https://gofastmcp.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checking: MyPy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://mypy.readthedocs.io/)

> **Your personal Data Scientist assistant** - Interactive data exploration with FastMCP 2.0

Transform complex datasets into clear, actionable insights with this powerful MCP server built for Claude Desktop.

## 🚀 Quick Start

### 1. Install Claude Desktop
Download from [claude.ai/download](https://claude.ai/download)

### 2. Setup MCP Server
```bash
# Install the server
uvx mcp-server-ds

# Or for development
git clone https://github.com/your-org/mcp-server-data-exploration
cd mcp-server-data-exploration
uv sync
```

### 3. Configure Claude Desktop
Add to your `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mcp-server-ds": {
      "command": "uvx",
      "args": ["mcp-server-ds"]
    }
  }
}
```

### 4. Start Exploring!
1. Restart Claude Desktop
2. Select the **explore-data** prompt template
3. Provide your CSV path and exploration topic
4. Watch the magic happen! ✨

## 📊 Features

### 🔧 **Core Tools**
- **`load_csv`** - Load CSV files into DataFrames
- **`run_script`** - Execute Python data analysis scripts
- **Auto-discovery** - Find CSV files in common directories

### 📈 **Data Science Libraries**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **scikit-learn** - Machine learning
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization

### 🛡️ **Enterprise Features**
- **Session-based security** - Isolated user sessions
- **Memory management** - Automatic cleanup and optimization
- **Error handling** - Robust error recovery
- **Resource monitoring** - Real-time system health

## 🛠️ Development Setup

### Prerequisites
- **Python 3.11+**
- **UV** package manager
- **Git**

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/mcp-server-data-exploration
cd mcp-server-data-exploration

# Install dependencies
uv sync

# Install development tools
uv sync --group dev
```

### Development Tools Setup
```bash
# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push

# Run quality checks
uv run pre-commit run --all-files

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=mcp_server_ds --cov-report=term-missing
```

### Code Quality Stack
- **Ruff** - Ultra-fast linting and formatting
- **MyPy** - Static type checking
- **Pyupgrade** - Automatic Python syntax modernization
- **Pytest** - Testing framework with coverage
- **Pre-commit** - Automated quality checks
- **Gitlint** - Commit message standards

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=mcp_server_ds --cov-report=html

# Run fast tests only
uv run pytest tests/ -m "not slow" -v

# Run specific test categories
uv run pytest tests/unit/ -v          # Unit tests
uv run pytest tests/integration/ -v   # Integration tests
```

## 🏗️ Architecture

### FastMCP 2.0 Benefits
- **70% Code Reduction** - From 336 to 236 lines
- **Simplified Architecture** - Decorator-based design
- **Enhanced Performance** - Built-in optimizations
- **Better Maintainability** - Clean, Pythonic code

### Session Management
- **Isolated Sessions** - User data separation
- **Memory Optimization** - Automatic cleanup
- **Security** - Session ID override protection
- **Monitoring** - Real-time health checks

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Setup
```bash
# Fork and clone
git clone https://github.com/your-username/mcp-server-data-exploration
cd mcp-server-data-exploration

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push

# Make your changes and test
uv run pytest tests/ -v
uv run pre-commit run --all-files
```

### Commit Message Format
We use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add new feature
fix: resolve bug
docs: update documentation
test: add tests
refactor: improve code structure
```
