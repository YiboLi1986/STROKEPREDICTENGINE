import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import pandas as pd
from collections import Counter
from app.services.processing.interval_divider import IntervalDivider
from app.services.processing.grid_merger import GridMerger
from app.services.processing.grid_processor import GridProcessor

class TestGridProcessor(unittest.TestCase):
    
    def setUp(self):
        file_path = "app/services/data/data_clean_2.xlsx"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        
        # Initialize IntervalDivider and GridMerger
        divider = 2
        interval_divider = IntervalDivider(file_path, divider)
        
        # Retrieve grid map and intervals
        grid_map = interval_divider.grid_map
        diff_threshold = 0.3
        
        # Use GridMerger to merge grids
        merger = GridMerger(grid_map, diff_threshold)
        final_grid_map, histograms = merger.merge_until_threshold(interval_divider.data)
        
        # Initialize GridProcessor
        self.processor = GridProcessor(
            final_grid_map=final_grid_map,
            histograms=histograms,
            file_path=file_path,
            len_threshold=2,
            len_threshold_0=2,
            len_threshold_1=3,
            num_threshold_0=0.7,
            num_threshold_1=0.7
        )

    def test_select_grids(self):
        # Test the select_grids method
        selected_grids = self.processor.select_grids()
        self.assertIsInstance(selected_grids, dict)
        self.assertGreaterEqual(len(selected_grids), 1)  # At least one grid should be selected

    def test_select_majority_grids(self):
        # Test the select_majority_grids method
        majority_grids = self.processor.select_majority_grids()
        self.assertIsInstance(majority_grids, dict)
        # Check if specific grids meet the selection criteria based on majority labels

    def test_statistics_of_selected_grids(self):
        # Test the statistics_of_selected_grids method
        selected_grids = self.processor.select_grids()
        stats = self.processor.statistics_of_selected_grids(selected_grids)
        self.assertIsInstance(stats, tuple)
        self.assertEqual(len(stats), 4)  # Should contain four statistical values

    def test_create_new_data_from_selected_grids(self):
        # Test the create_new_data_from_selected_grids method
        selected_grids = self.processor.select_grids()
        new_data = self.processor.create_new_data_from_selected_grids(selected_grids)
        self.assertIsInstance(new_data, pd.DataFrame)

    def test_train_and_evaluate_nn(self):
        # Test the train_and_evaluate_nn method
        selected_grids = self.processor.select_grids()
        accuracy = self.processor.train_and_evaluate_nn(selected_grids)
        self.assertGreaterEqual(accuracy, 0.0)  # Ensure accuracy is a valid float

if __name__ == "__main__":
    unittest.main(exit=False)
