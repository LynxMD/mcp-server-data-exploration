import logging
import glob
import os
from typing import Any

# FastMCP 2.0 import
from fastmcp import FastMCP

# Data analysis libraries
import pandas as pd
import numpy as np
import scipy
import sklearn
import statsmodels.api as sm
from io import StringIO
import sys
import pyarrow
from PIL import Image
import pytesseract
import pymupdf

# Data management
from .base_data_manager import DataManager
from .hybrid_data_manager import HybridDataManager
from .system_utils import log_system_status

logger = logging.getLogger(__name__)
# Ensure logs are visible in the FastMCP subprocess even if no handlers configured
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.info("Starting FastMCP 2.0 data science exploration server")

# Create FastMCP instance
mcp = FastMCP("Data Science Explorer 🔬")

# Prompt template (preserved from original)
PROMPT_TEMPLATE = """
You are a professional Data Scientist tasked with performing exploratory data analysis on a dataset. Your goal is to provide insightful analysis while ensuring stability and manageable result sizes.

First, load the CSV file from the following path:

<csv_path>
{csv_path}
</csv_path>

Your analysis should focus on the following topic:

<analysis_topic>
{topic}
</analysis_topic>

You have access to the following tools for your analysis:
1. load_csv: Use this to load the CSV file.
2. run_script: Use this to execute Python scripts on the MCP server.

Please follow these steps carefully:

1. Load the CSV file using the load_csv tool.

2. Explore the dataset. Provide a brief summary of its structure, including the number of rows, columns, and data types. Wrap your exploration process in <dataset_exploration> tags, including:
   - List of key statistics about the dataset
   - Potential challenges you foresee in analyzing this data

3. Wrap your thought process in <analysis_planning> tags:
   Analyze the dataset size and complexity:
   - How many rows and columns does it have?
   - Are there any potential computational challenges based on the data types or volume?
   - What kind of questions would be appropriate given the dataset's characteristics and the analysis topic?
   - How can we ensure that our questions won't result in excessively large outputs?

   Based on this analysis:
   - List 10 potential questions related to the analysis topic
   - Evaluate each question against the following criteria:
     * Directly related to the analysis topic
     * Can be answered with reasonable computational effort
     * Will produce manageable result sizes
     * Provides meaningful insights into the data
   - Select the top 5 questions that best meet all criteria

4. List the 5 questions you've selected, ensuring they meet the criteria outlined above.

5. For each question, follow these steps:
   a. Wrap your thought process in <analysis_planning> tags:
      - How can I structure the Python script to efficiently answer this question?
      - What data preprocessing steps are necessary?
      - How can I limit the output size to ensure stability?
      - What type of visualization would best represent the results?
      - Outline the main steps the script will follow

   b. Write a Python script to answer the question. Include comments explaining your approach and any measures taken to limit output size.

   c. Use the run_script tool to execute your Python script on the MCP server.

   d. Render the results returned by the run-script tool as a chart using plotly.js (prefer loading from cdnjs.cloudflare.com). Do not use react or recharts, and do not read the original CSV file directly. Provide the plotly.js code to generate the chart.

6. After completing the analysis for all 5 questions, provide a brief summary of your findings and any overarching insights gained from the data.

Remember to prioritize stability and manageability in your analysis. If at any point you encounter potential issues with large result sets, adjust your approach accordingly.

Please begin your analysis by loading the CSV file and providing an initial exploration of the dataset.
"""


# ScriptRunner class with session isolation
class ScriptRunner:
    def __init__(self, data_manager: DataManager | None = None):
        # Initialize data manager
        # Default: Hybrid storage (memory + filesystem) for optimal performance
        # and persistence. Falls back to TTL in-memory for demos if needed.
        self.data_manager = data_manager or HybridDataManager()
        # Session-based notes: {session_id: [notes]}
        self.session_notes: dict[str, list[str]] = {}
        # Session-based DataFrame counters: {session_id: count}
        self.session_df_count: dict[str, int] = {}

    def log_system_status(self) -> None:
        """Delegate to system utils for logging and alerting."""
        dm_name = self.data_manager.__class__.__name__
        log_system_status(dm_name)

    def _validate_session_id(self, session_id: str) -> str:
        """Validate that session_id is a non-empty string."""
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        return session_id.strip()

    def _get_session_data(self, session_id: str) -> dict[str, Any]:
        """Get or create session data storage."""
        return self.data_manager.get_session_data(session_id)

    def _get_session_notes(self, session_id: str) -> list[str]:
        """Get or create session notes storage."""
        if session_id not in self.session_notes:
            self.session_notes[session_id] = []
        return self.session_notes[session_id]

    def _get_session_df_count(self, session_id: str) -> int:
        """Get or create session DataFrame counter."""
        if session_id not in self.session_df_count:
            self.session_df_count[session_id] = 0
        return self.session_df_count[session_id]

    def _increment_session_df_count(self, session_id: str) -> int:
        """Increment and return session DataFrame counter."""
        if session_id not in self.session_df_count:
            self.session_df_count[session_id] = 0
        self.session_df_count[session_id] += 1
        return self.session_df_count[session_id]

    def load_csv(
        self, csv_path: str, df_name: str | None = None, session_id: str = None
    ) -> str:
        """Load CSV with session isolation."""
        if not session_id:
            raise ValueError("session_id is required for session isolation")

        session_id = self._validate_session_id(session_id)
        session_notes = self._get_session_notes(session_id)

        df_count = self._increment_session_df_count(session_id)
        if not df_name:
            df_name = f"df_{df_count}"

        try:
            df_data = pd.read_csv(csv_path)
            self.data_manager.set_dataframe(session_id, df_name, df_data)
            session_notes.append(
                f"Successfully loaded CSV into dataframe '{df_name}' for session '{session_id}'"
            )
            return f"Successfully loaded CSV into dataframe '{df_name}'"
        except Exception as e:
            error_msg = f"Error loading CSV: {str(e)}"
            session_notes.append(error_msg)
            raise Exception(error_msg)

    def safe_eval(
        self,
        script: str,
        save_to_memory: list[str] | None = None,
        session_id: str = None,
    ) -> str:
        """Safely run a script with session isolation."""
        if not session_id:
            raise ValueError("session_id is required for session isolation")

        session_id = self._validate_session_id(session_id)
        session_data = self._get_session_data(session_id)
        session_notes = self._get_session_notes(session_id)

        # Extract dataframes from session-specific data
        local_dict = {
            **{df_name: df for df_name, df in session_data.items()},
        }

        # Execute the script and return the result
        try:
            stdout_capture = StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            session_notes.append(
                f"Running script for session '{session_id}': \n{script}"
            )
            # pylint: disable=exec-used
            exec(
                script,
                {
                    "pd": pd,
                    "np": np,
                    "scipy": scipy,
                    "sklearn": sklearn,
                    "statsmodels": sm,
                    "pyarrow": pyarrow,
                    "Image": Image,
                    "pytesseract": pytesseract,
                    "pymupdf": pymupdf,
                },
                local_dict,
            )
            sys.stdout = old_stdout
            std_out_script = stdout_capture.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"Error running script for session '{session_id}': {str(e)}"
            session_notes.append(error_msg)
            raise Exception(str(e))

        # Save dataframes to session-specific memory
        if save_to_memory:
            for df_name in save_to_memory:
                session_notes.append(
                    f"Saving dataframe '{df_name}' to memory for session '{session_id}'"
                )
                self.data_manager.set_dataframe(
                    session_id, df_name, local_dict.get(df_name)
                )

        output = std_out_script if std_out_script else "No output"
        session_notes.append(f"Result for session '{session_id}': {output}")
        return output


# Global script runner instance
script_runner = ScriptRunner()


# === PROMPTS ===
@mcp.prompt
def explore_data(csv_path: str, topic: str = "general data exploration") -> str:
    """A prompt to explore a CSV dataset as a data scientist."""
    return PROMPT_TEMPLATE.format(csv_path=csv_path, topic=topic)


# === TOOLS ===
@mcp.tool
def load_csv(csv_path: str, df_name: str | None = None, session_id: str = None) -> str:
    """Load a local CSV file into a DataFrame with session isolation.

    Args:
        csv_path: Path to the CSV file
        df_name: Optional name for the DataFrame. If not provided, will auto-assign df_1, df_2, etc.
        session_id: Session ID for data isolation (required)

    Returns:
        Success message with DataFrame name
    """
    if not session_id:
        raise ValueError("session_id is required for session isolation")

    session_id = session_id.strip()
    if not session_id:
        raise ValueError("session_id must be a non-empty string")

    # Log environment at tool entry
    script_runner.log_system_status()
    return script_runner.load_csv(csv_path, df_name, session_id)


@mcp.tool
def run_script(
    script: str, save_to_memory: list[str] | None = None, session_id: str = None
) -> str:
    """Execute Python scripts for data analytics tasks with session isolation.

    Args:
        script: The Python script to execute
        save_to_memory: Optional list of DataFrame names to save to memory
        session_id: Session ID for data isolation (required)

    Returns:
        Script execution result
    """
    if not session_id:
        raise ValueError("session_id is required for session isolation")

    session_id = session_id.strip()
    if not session_id:
        raise ValueError("session_id must be a non-empty string")

    # Log environment at tool entry
    script_runner.log_system_status()
    return script_runner.safe_eval(script, save_to_memory, session_id)


# === RESOURCES ===
@mcp.resource("data-exploration://notes/{session_id}")
def get_exploration_notes(session_id: str) -> str:
    """Notes generated by the data exploration server for a specific session."""
    session_id = session_id.strip()
    if not session_id:
        return "Invalid session_id - cannot retrieve session-specific notes"

    session_notes = script_runner._get_session_notes(session_id)
    return (
        "\n".join(session_notes)
        if session_notes
        else f"No notes yet for session '{session_id}'"
    )


@mcp.resource("data-exploration://csv-files")
def list_csv_files() -> str:
    """List available CSV files in common data directories."""
    common_paths = ["~/code/ai/data"]  # , ["~/Downloads", "~/tmp"]
    csv_files = []

    for path in common_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            csv_files.extend(glob.glob(f"{expanded_path}/*.csv"))

    if not csv_files:
        return "No CSV files found in common directories (~/code/ai/data, ~/Downloads, ~/tmp)"

    return "CSV file listing:\n" + "\n".join(csv_files)


# Note: list_dataframes and get_analysis_history resources removed
# These functions are no longer compatible with session-based storage
# Use session-specific resources instead (e.g., data-exploration://notes/{session_id})


# === MAIN ENTRY POINT ===
def main():
    """Main entry point for the FastMCP 2.0 server."""
    mcp.run()


if __name__ == "__main__":
    main()
