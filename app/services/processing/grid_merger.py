import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from collections import Counter
import pandas as pd
from app.services.processing.interval_divider import IntervalDivider

class GridMerger:
    """
    A class responsible for merging grids based on label histograms and threshold.
    
    Attributes:
        grid_map (dict): Pre-calculated mapping of samples to grids.
        threshold (float): Threshold percentage for comparing histogram differences for merging grids.
    """

    def __init__(self, grid_map: dict, threshold: float):
        """
        Initializes the GridMerger with a grid mapping and a threshold for determining merge eligibility.

        Args:
            grid_map (dict): Pre-calculated mapping of samples to grids.
            threshold (float): Threshold percentage for comparing histogram differences for merging grids.

        Raises:
            ValueError: If threshold is not between 0 and 1.
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        
        self.grid_map = grid_map
        self.threshold = threshold

    def calculate_label_histograms(self, data: pd.DataFrame) -> dict:
        """
        Calculates the label histogram for each grid based on the dataset's last column (assumed as labels).

        Args:
            data (pd.DataFrame): The dataset including the labels.

        Returns:
            dict: A dictionary with grid IDs as keys and label histograms (Counter objects) as values.
        """
        histograms = {}
        for grid_id, sample_indices in self.grid_map.items():
            labels = [data.iloc[i].iloc[-1] for i in sample_indices]
            histograms[grid_id] = Counter(labels)
        return histograms

    def calculate_histogram_difference(self, histogram1: Counter, histogram2: Counter) -> int:
        """
        Calculates the difference between two histograms as the sum of absolute differences for each label.

        Args:
            histogram1 (Counter): The histogram for the first grid.
            histogram2 (Counter): The histogram for the second grid.

        Returns:
            int: The total difference between the two histograms.
        """
        all_labels = set(histogram1) | set(histogram2)
        total_diff = sum(abs(histogram1.get(label, 0) - histogram2.get(label, 0)) for label in all_labels)
        return total_diff

    def is_difference_within_threshold(self, histogram1: Counter, histogram2: Counter, total_samples: int) -> bool:
        """
        Determines if the difference between two histograms is within a specified threshold.

        Args:
            histogram1 (Counter): The histogram for the first grid.
            histogram2 (Counter): The histogram for the second grid.
            total_samples (int): The total number of samples in both grids combined.

        Returns:
            bool: True if the difference is within the threshold; False otherwise.
        """
        total_diff = self.calculate_histogram_difference(histogram1, histogram2)
        diff_percentage = total_diff / total_samples if total_samples else 0
        return diff_percentage < self.threshold

    def are_grids_adjacent(self, grid_id1, grid_id2):
        """
        Determines if two grids identified by their grid IDs are adjacent.

        Args:
            grid_id1 (tuple): The ID of the first grid, elements may be int or tuple.
            grid_id2 (tuple): The ID of the second grid, elements may be int or tuple.

        Returns:
            bool: True if the grids are adjacent; False otherwise.
        """
        if len(grid_id1) != len(grid_id2):
            return False

        diff_count = 0
        for dim1, dim2 in zip(grid_id1, grid_id2):
            min1, max1 = (dim1[0], dim1[-1]) if isinstance(dim1, tuple) else (dim1, dim1)
            min2, max2 = (dim2[0], dim2[-1]) if isinstance(dim2, tuple) else (dim2, dim2)

            if min1 == min2 and max1 == max2:
                continue
            elif max1 + 1 == min2 or max2 + 1 == min1:
                diff_count += 1
            else:
                return False

        return diff_count == 1

    def merge_grids(self, grid_id1: tuple, grid_id2: tuple, samples1: list, samples2: list) -> tuple:
        """
        Merges two grids into a new grid with a combined grid ID and a list of sample IDs from both grids.

        Args:
            grid_id1 (tuple): The ID of the first grid to merge.
            grid_id2 (tuple): The ID of the second grid to merge.
            samples1 (list): The list of sample IDs in the first grid.
            samples2 (list): The list of sample IDs in the second grid.

        Returns:
            tuple: The new combined grid ID and the combined list of sample IDs from both grids.
        """
        combined_samples = samples1 + samples2

        new_grid_id = tuple(dim1 if dim1 == dim2 else (min(dim1, dim2), max(dim1, dim2))
                            for dim1, dim2 in zip(grid_id1, grid_id2))

        return new_grid_id, combined_samples

    def merge_until_threshold(self, data: pd.DataFrame) -> tuple:
        """
        Repeatedly merges pairs of adjacent grids with the minimum difference in their label histograms,
        until no more pairs can be merged without exceeding the threshold.

        Args:
            data (pd.DataFrame): The dataset including the labels.

        Returns:
            tuple: The final mapping of samples to merged grids and the updated histograms.
        """
        histograms = self.calculate_label_histograms(data)
        changes_made = True
        
        while changes_made:
            changes_made = False
            min_diff = float('inf')
            pair_to_merge = None

            grid_ids = list(self.grid_map.keys())
            for i in range(len(grid_ids)):
                for j in range(i + 1, len(grid_ids)):
                    grid_id1 = grid_ids[i]
                    grid_id2 = grid_ids[j]
                    if self.are_grids_adjacent(grid_id1, grid_id2):
                        total_samples = len(self.grid_map[grid_id1]) + len(self.grid_map[grid_id2])
                        if self.is_difference_within_threshold(histograms[grid_id1], histograms[grid_id2], total_samples):
                            diff = self.calculate_histogram_difference(histograms[grid_id1], histograms[grid_id2])
                            if diff < min_diff:
                                min_diff = diff
                                pair_to_merge = (grid_id1, grid_id2)

            if pair_to_merge:
                grid_id1, grid_id2 = pair_to_merge
                new_grid_id, combined_samples = self.merge_grids(
                    grid_id1, grid_id2, self.grid_map[grid_id1], self.grid_map[grid_id2]
                )
                self.grid_map[new_grid_id] = combined_samples
                del self.grid_map[grid_id1]
                del self.grid_map[grid_id2]
                histograms = self.calculate_label_histograms(data)
                changes_made = True

        return self.grid_map, histograms
