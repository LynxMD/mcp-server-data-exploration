import logging
from typing import Optional, List
import glob
import os
from pathlib import Path

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

logger = logging.getLogger(__name__)
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

# ScriptRunner class (preserved from original)
class ScriptRunner:
    def __init__(self):
        self.data = {}
        self.df_count = 0
        self.notes: list[str] = []

    def load_csv(self, csv_path: str, df_name: str = None):
        self.df_count += 1
        if not df_name:
            df_name = f"df_{self.df_count}"
        try:
            self.data[df_name] = pd.read_csv(csv_path)
            self.notes.append(f"Successfully loaded CSV into dataframe '{df_name}'")
            return f"Successfully loaded CSV into dataframe '{df_name}'"
        except Exception as e:
            error_msg = f"Error loading CSV: {str(e)}"
            self.notes.append(error_msg)
            raise Exception(error_msg)

    def safe_eval(self, script: str, save_to_memory: Optional[List[str]] = None):
        """safely run a script, return the result if valid, otherwise return the error message"""
        # first extract dataframes from the self.data
        local_dict = {
            **{df_name: df for df_name, df in self.data.items()},
        }
        # execute the script and return the result and if there is error, return the error message
        try:
            stdout_capture = StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            self.notes.append(f"Running script: \n{script}")
            # pylint: disable=exec-used
            exec(script, 
                {'pd': pd, 'np': np, 'scipy': scipy, 'sklearn': sklearn, 'statsmodels': sm, 'pyarrow': pyarrow, 'Image': Image,
                 'pytesseract': pytesseract, 'pymupdf': pymupdf}, 
                local_dict)
            sys.stdout = old_stdout
            std_out_script = stdout_capture.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"Error running script: {str(e)}"
            self.notes.append(error_msg)
            raise Exception(str(e))

        # check if the result is a dataframe
        if save_to_memory:
            for df_name in save_to_memory:
                self.notes.append(f"Saving dataframe '{df_name}' to memory")
                self.data[df_name] = local_dict.get(df_name)

        output = std_out_script if std_out_script else "No output"
        self.notes.append(f"Result: {output}")
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
def load_csv(csv_path: str, df_name: str = None) -> str:
    """Load a local CSV file into a DataFrame.
    
    Args:
        csv_path: Path to the CSV file
        df_name: Optional name for the DataFrame. If not provided, will auto-assign df_1, df_2, etc.
    
    Returns:
        Success message with DataFrame name
    """
    return script_runner.load_csv(csv_path, df_name)

@mcp.tool
def run_script(script: str, save_to_memory: List[str] = None) -> str:
    """Execute Python scripts for data analytics tasks.
    
    Args:
        script: The Python script to execute
        save_to_memory: Optional list of DataFrame names to save to memory
    
    Returns:
        Script execution result
    """
    return script_runner.safe_eval(script, save_to_memory)

# === RESOURCES ===
@mcp.resource("data-exploration://notes")
def get_exploration_notes() -> str:
    """Notes generated by the data exploration server."""
    return "\n".join(script_runner.notes) if script_runner.notes else "No notes yet"

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

@mcp.resource("data-exploration://dataframes")
def list_dataframes() -> str:
    """List currently loaded DataFrames with their information."""
    if not script_runner.data:
        return "No DataFrames loaded"
    
    info = ["Currently loaded DataFrames:"]
    for name, df in script_runner.data.items():
        info.append(f"- {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        info.append(f"  Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
    
    return "\n".join(info)

@mcp.resource("data-exploration://history")
def get_analysis_history() -> str:
    """Get history of recent analysis operations."""
    if not script_runner.notes:
        return "No analysis history yet"
    
    return "Recent analysis history:\n" + "\n".join(script_runner.notes[-10:])

# === MAIN ENTRY POINT ===
def main():
    """Main entry point for the FastMCP 2.0 server."""
    mcp.run()

if __name__ == "__main__":
    main()
