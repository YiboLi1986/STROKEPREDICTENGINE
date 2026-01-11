import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
from app.services.interval_divider import IntervalDivider

class TestIntervalDivider(unittest.TestCase):

    def setUp(self):
        # Setup code here to initialize IntervalDivider with a sample file and divider
        self.file_path = "app/services/data/data_clean.xlsx" 
        self.divider = 2
        self.interval_divider = IntervalDivider(self.file_path, self.divider)

    def test_intervals_calculation(self):
        # Test if intervals are calculated correctly
        intervals = self.interval_divider.intervals
        self.assertIsInstance(intervals, dict)
        for column, column_intervals in intervals.items():
            self.assertEqual(len(column_intervals), self.divider)

    def test_grid_mapping(self):
        # Test if samples are correctly mapped to grids
        grid_map = self.interval_divider.grid_map
        self.assertIsInstance(grid_map, dict)
        for grid_id, sample_indices in grid_map.items():
            self.assertIsInstance(grid_id, tuple)
            self.assertIsInstance(sample_indices, list)

    def test_find_interval_index(self):
        # Test if interval index is found correctly
        sample_value = 0.6
        intervals = self.interval_divider.intervals[next(iter(self.interval_divider.intervals))]
        index = self.interval_divider.find_interval_index(intervals, sample_value)
        self.assertIsInstance(index, int)

if __name__ == "__main__":
    unittest.main(exit=False)
