"""Integration tests for MCP server functionality."""

import pytest
from mcp_server_ds.server import script_runner


class TestMCPIntegration:
    """Integration tests for complete MCP workflows."""

    def test_complete_data_analysis_workflow(self, temp_csv_file):
        """Test a complete data analysis workflow."""
        # Step 1: Load data
        load_result = script_runner.load_csv(temp_csv_file, "people")
        assert "Successfully loaded CSV" in load_result

        # Step 2: Basic analysis
        analysis_script = """
print("=== Data Analysis ===")
print(f"Dataset shape: {people.shape}")
print(f"Columns: {list(people.columns)}")
print(f"Age statistics:")
print(people['age'].describe())
"""
        analysis_result = script_runner.safe_eval(analysis_script)
        assert "Dataset shape: (3, 3)" in analysis_result
        assert "Columns: ['name', 'age', 'city']" in analysis_result

        # Step 3: Create derived data
        derived_script = """
# Create age groups
people['age_group'] = people['age'].apply(lambda x: 'young' if x < 30 else 'mature')
print(f"Age groups: {people['age_group'].value_counts().to_dict()}")
"""
        derived_result = script_runner.safe_eval(
            derived_script, save_to_memory=["people"]
        )
        assert "Age groups:" in derived_result

        # Step 4: Check notes
        notes = (
            "\n".join(script_runner.notes) if script_runner.notes else "No notes yet"
        )
        assert "Successfully loaded CSV" in notes
        assert "Data Analysis" in notes

    def test_multiple_dataframes_workflow(self, temp_csv_file):
        """Test workflow with multiple DataFrames."""
        # Load same data with different names
        script_runner.load_csv(temp_csv_file, "dataset1")
        script_runner.load_csv(temp_csv_file, "dataset2")

        # Work with both datasets
        script = """
print("=== Multi-Dataset Analysis ===")
print(f"Dataset1 shape: {dataset1.shape}")
print(f"Dataset2 shape: {dataset2.shape}")

# Compare datasets
comparison = dataset1.equals(dataset2)
print(f"Datasets are identical: {comparison}")
"""
        result = script_runner.safe_eval(script)
        assert "Dataset1 shape: (3, 3)" in result
        assert "Dataset2 shape: (3, 3)" in result
        assert "Datasets are identical: True" in result

    def test_error_recovery_workflow(self, temp_csv_file):
        """Test error handling and recovery in workflows."""
        # Load data successfully
        script_runner.load_csv(temp_csv_file, "data")

        # Try script with error
        error_script = "undefined_variable + 1"
        with pytest.raises(Exception):
            script_runner.safe_eval(error_script)

        # Continue with valid script
        valid_script = "print(f'Data loaded successfully: {data.shape}')"
        result = script_runner.safe_eval(valid_script)
        assert "Data loaded successfully: (3, 3)" in result

    @pytest.mark.slow
    def test_large_script_workflow(self, temp_csv_file):
        """Test workflow with a larger, more complex script."""
        script_runner.load_csv(temp_csv_file, "df")

        large_script = """
import pandas as pd
import numpy as np

print("=== Complex Analysis ===")

# Basic statistics
print("Basic statistics:")
print(df.describe())

# Data manipulation
df['age_squared'] = df['age'] ** 2
df['name_length'] = df['name'].str.len()

# Grouping operations
city_stats = df.groupby('city')['age'].agg(['mean', 'std', 'count'])
print("\\nCity statistics:")
print(city_stats)

# Conditional operations
young_people = df[df['age'] < 30]
print(f"\\nYoung people count: {len(young_people)}")

# String operations
df['name_upper'] = df['name'].str.upper()
print("\\nNames in uppercase:")
print(df['name_upper'].tolist())
"""
        result = script_runner.safe_eval(large_script, save_to_memory=["df"])
        assert "Basic statistics:" in result
        assert "City statistics:" in result
        assert "Young people count: 1" in result
        assert "Names in uppercase:" in result
