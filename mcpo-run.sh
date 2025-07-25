#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Run the MCP server using uv
uv run --no-sync mcp-server-ds


###### Claude Desktop MCP Configuration JSON:
# {
#   "mcpServers": {
#     "mcp-server-ds": {
#       "command": "/Users/omer/code/github/mcp-server-data-exploration/mcpo-run-direct.sh"
#     }
#   }
# }