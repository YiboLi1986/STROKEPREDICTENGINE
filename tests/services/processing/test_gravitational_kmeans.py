import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import pandas as pd
from app.services.processing.interval_divider import IntervalDivider
from app.services.processing.grid_merger import GridMerger
from app.services.processing.grid_processor import GridProcessor
from app.services.processing.gravitational_kmeans import GravitationalKMeans

class TestGravitationalKMeans(unittest.TestCase):

    def setUp(self):
        # Define file path and check if the file exists
        file_path = 'app/services/data/data_clean.xlsx'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        # Initialize IntervalDivider
        divider = 2
        interval_divider = IntervalDivider(file_path, divider)
        grid_map = interval_divider.grid_map
        threshold = 0.3

        # Initialize GridMerger
        merger = GridMerger(grid_map, threshold)
        final_grid_map, histograms = merger.merge_until_threshold(interval_divider.data)

        # Initialize GridProcessor
        len_threshold = 2
        len_threshold_0 = 2
        len_threshold_1 = 3
        num_threshold_0 = 0.7
        num_threshold_1 = 0.7
        processor = GridProcessor(final_grid_map, histograms, file_path, len_threshold, len_threshold_0, len_threshold_1, num_threshold_0, num_threshold_1)
        new_final_grid_map = processor.select_majority_grids()

        # Prepare initial clusters for GravitationalKMeans
        initial_clusters = [samples for samples in new_final_grid_map.values()]
        data = pd.read_excel(file_path)

        # Initialize GravitationalKMeans
        self.gkm = GravitationalKMeans(data, initial_clusters)

    def test_fit_incremental(self):
        """ Test the incremental fitting method and check the final cluster state """
        self.gkm.fit_incremental()
        self.assertIsInstance(self.gkm.clusters, list)
        self.assertGreaterEqual(len(self.gkm.clusters), 1)
        self.assertGreater(len(self.gkm.accuracy_history), 0)

    def test_train_and_evaluate_nn_2(self):
        """ Test training and evaluation of the neural network on the entire dataset """
        accuracy = self.gkm.train_and_evaluate_nn_2()
        self.assertGreaterEqual(accuracy, 0.0)
        print(f"Overall Neural Network Test Accuracy: {accuracy:.2f}%")

    def test_train_and_evaluate_best_clusters(self):
        """ Test training on the best clusters identified and evaluate all models """
        self.gkm.fit_incremental()
        accuracy_nn = self.gkm.train_and_evaluate_nn(self.gkm.best_clusters, flag=1)
        self.assertGreaterEqual(accuracy_nn, 0.0)
        print(f"Neural Network Model Accuracy: {accuracy_nn:.2f}%")

        accuracy_rf = self.gkm.train_and_evaluate_rf(self.gkm.best_clusters, flag=1)
        self.assertGreaterEqual(accuracy_rf, 0.0)
        print(f"Random Forest Model Accuracy: {accuracy_rf:.2f}%")

        accuracy_xgb = self.gkm.train_and_evaluate_xgb(self.gkm.best_clusters, flag=1)
        self.assertGreaterEqual(accuracy_xgb, 0.0)
        print(f"XGBoost Model Accuracy: {accuracy_xgb:.2f}%")

    def test_plot_accuracy_history(self):
        """ Test the plotting of accuracy history for incremental clustering """
        self.gkm.fit_incremental()
        self.gkm.plot_accuracy_history()
        # Ensure that accuracy history has been recorded
        self.assertGreater(len(self.gkm.accuracy_history), 0)

if __name__ == "__main__":
    unittest.main(exit=False)
