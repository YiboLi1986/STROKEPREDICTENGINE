import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import pandas as pd
from collections import Counter
from app.services.processing.interval_divider import IntervalDivider
from app.services.processing.grid_merger import GridMerger

class TestGridMerger(unittest.TestCase):

    def setUp(self):
        self.file_path = "app/services/data/data_clean_2.xlsx" 
        self.divider = 2
        interval_divider = IntervalDivider(self.file_path, self.divider)
        self.grid_map = interval_divider.grid_map
        self.threshold = 0.3
        self.data = interval_divider.data
        self.merger = GridMerger(self.grid_map, self.threshold)

    def test_calculate_label_histograms(self):
        histograms = self.merger.calculate_label_histograms(self.data)
        self.assertIsInstance(histograms, dict)
        for grid_id, histogram in histograms.items():
            self.assertIsInstance(histogram, Counter)

    def test_calculate_histogram_difference(self):
        histogram1 = Counter({'A': 3, 'B': 2})
        histogram2 = Counter({'A': 1, 'B': 3})
        difference = self.merger.calculate_histogram_difference(histogram1, histogram2)
        self.assertEqual(difference, 3)

    def test_is_difference_within_threshold(self):
        histogram1 = Counter({'A': 3, 'B': 2})
        histogram2 = Counter({'A': 2, 'B': 3})
        total_samples = 10
        within_threshold = self.merger.is_difference_within_threshold(histogram1, histogram2, total_samples)
        self.assertTrue(within_threshold)

    def test_are_grids_adjacent(self):
        grid_id1 = (1, 2)
        grid_id2 = (1, 3)
        self.assertTrue(self.merger.are_grids_adjacent(grid_id1, grid_id2))
        grid_id3 = (1, 2)
        grid_id4 = (2, 4)
        self.assertFalse(self.merger.are_grids_adjacent(grid_id3, grid_id4))

    def test_merge_until_threshold(self):
        final_grid_map, histograms = self.merger.merge_until_threshold(self.data)
        self.assertIsInstance(final_grid_map, dict)
        self.assertIsInstance(histograms, dict)

if __name__ == "__main__":
    unittest.main(exit=False)
