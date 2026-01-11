import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class IntervalDivider:
    """
    A class to divide the range of each column in a dataset into equal intervals and map samples to these intervals.
    """
    def __init__(self, file_path, divider):
        """
        Initializes the IntervalDivider with a dataset and a divider value.

        Args:
            file_path (str): The path to the cleaned data.
            divider (int): The number of intervals to divide each column's range into.
        """
        if divider <= 0:
            raise ValueError("Divider must be greater than zero.")
        
        try:
            self.data = pd.read_excel(file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found at the provided path: {file_path}") from e
        except Exception as e:
            raise Exception(f"An error occurred while reading the Excel file: {e}") from e

        self.divider = divider
        self.intervals = self.get_intervals()
        self.grid_map = self.map_samples_to_grids()

    def get_intervals(self):
        """
        Calculates the intervals for each column in the dataset, excluding the last column (assumed to be the target/label column).

        Returns:
            dict: A dictionary with column names as keys and lists of interval ranges as values.
        """
        intervals = {}
        for column in self.data.columns[:-1]:
            min_val = self.data[column].min()
            max_val = self.data[column].max()

            if max_val == min_val:
                logging.warning(f"Column '{column}' has a constant value; skipping interval division.")
                continue

            interval_range = (max_val - min_val) / self.divider
            column_intervals = [
                [round(min_val + i * interval_range, 4), round(min_val + (i + 1) * interval_range, 4)]
                for i in range(self.divider)
            ]
            column_intervals[-1][1] = max_val
            intervals[column] = column_intervals
        
        return intervals

    def map_samples_to_grids(self):
        """
        Maps each sample in the dataset to a grid based on its attribute values by comparing them against the calculated intervals.

        Returns:
            dict: A dictionary with grid IDs (tuples of interval indices) as keys and lists of sample indices as values.
        """
        grid_map = {}
        for index, row in self.data.iterrows():
            grid_id = self.get_grid_id(row)
            grid_map.setdefault(grid_id, []).append(index)
        return grid_map

    def get_grid_id(self, row):
        """
        Generates a grid ID for a given sample based on its attribute values.

        Args:
            row (pd.Series): A row of the dataset representing a sample.

        Returns:
            tuple: A tuple of interval indices representing the grid ID.
        """
        grid_id = []
        for column in self.data.columns[:-1]:
            value = row[column]
            interval_index = self.find_interval_index(self.intervals[column], value)
            grid_id.append(interval_index)
        return tuple(grid_id)

    def find_interval_index(self, intervals, value):
        """
        Finds the index of the interval that a given value falls into.

        Args:
            intervals (list of list): A list of intervals represented as [start, end] lists.
            value (float): The value to be located in the intervals.

        Returns:
            int: The index of the interval that contains the value.

        Raises:
            ValueError: If the value is not found within any interval.
        """
        for index, (start, end) in enumerate(intervals):
            if index == len(intervals) - 1 and start <= value <= end:
                return index
            elif start <= value < end:
                return index
        raise ValueError(f"Value {value} not found in intervals.")
