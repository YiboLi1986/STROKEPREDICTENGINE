import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from collections import Counter
from app.services.processing.grid_merger import GridMerger
from app.services.processing.interval_divider import IntervalDivider

class GridProcessor:
    """
    A class for processing grids based on sample counts and thresholds, and for training a neural network on selected data.
    """

    def __init__(self, final_grid_map: dict, histograms: dict, file_path: str, len_threshold: int,
                 len_threshold_0: int, len_threshold_1: int, num_threshold_0: float, num_threshold_1: float):
        """
        Initializes the GridProcessor class.

        Args:
            final_grid_map (dict): Grid IDs mapped to lists of sample IDs.
            histograms (dict): Grid IDs mapped to label histograms.
            file_path (str): Path to the dataset file.
            len_threshold (int): Sample count threshold for grid selection.
            len_threshold_0 (int): Threshold for the number of 0-label samples.
            len_threshold_1 (int): Threshold for the number of 1-label samples.
            num_threshold_0 (float): Proportion threshold for 0-label samples.
            num_threshold_1 (float): Proportion threshold for 1-label samples.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        self.final_grid_map = final_grid_map
        self.histograms = histograms
        self.data = pd.read_excel(file_path)
        self.len_threshold = len_threshold
        self.len_threshold_0 = len_threshold_0
        self.len_threshold_1 = len_threshold_1
        self.num_threshold_0 = num_threshold_0
        self.num_threshold_1 = num_threshold_1

    def select_grids(self) -> dict:
        """
        Selects grids based on the sample count threshold.

        Returns:
            dict: Selected grids based on sample count threshold.
        """
        return {grid_id: sample_ids for grid_id, sample_ids in self.final_grid_map.items() 
                if len(sample_ids) >= self.len_threshold}

    def select_majority_grids(self) -> dict:
        """
        Selects grids where the proportion and count of labels meet specified thresholds.

        Returns:
            dict: Selected grids based on majority label thresholds.
        """
        new_final_grid_map = {}
        for grid_id, sample_ids in self.final_grid_map.items():
            histogram = self.histograms.get(grid_id, {0: 0, 1: 0})
            total_samples = len(sample_ids)
            num_zeros = histogram.get(0, 0)
            num_ones = histogram.get(1, 0)

            zero_proportion = num_zeros / total_samples if total_samples else 0
            one_proportion = num_ones / total_samples if total_samples else 0

            if (zero_proportion >= self.num_threshold_0 and num_zeros >= self.len_threshold_0) or \
               (one_proportion >= self.num_threshold_1 and num_ones >= self.len_threshold_1):
                new_final_grid_map[grid_id] = sample_ids

        return new_final_grid_map

    def statistics_of_selected_grids(self, selected_grid_map: dict) -> tuple:
        """
        Computes statistics for selected grids.

        Returns:
            tuple: Total grids, total samples, total 0-label and 1-label samples.
        """
        total_grids = len(selected_grid_map)
        total_samples = 0
        total_zeros = 0
        total_ones = 0

        for sample_ids in selected_grid_map.values():
            labels = self.data.loc[sample_ids, 'Output']  # Assuming 'Output' column contains the labels
            total_samples += len(sample_ids)
            total_zeros += (labels == 0).sum()
            total_ones += (labels == 1).sum()

        return total_grids, total_samples, total_zeros, total_ones

    def create_new_data_from_selected_grids(self, selected_grid_map: dict) -> pd.DataFrame:
        """
        Creates a new DataFrame from selected grid samples.

        Returns:
            pd.DataFrame: New DataFrame containing the selected samples.
        """
        selected_sample_ids = [sample_id for sample_ids in selected_grid_map.values() for sample_id in sample_ids]
        return self.data.loc[selected_sample_ids]

    def train_and_evaluate_nn(self, selected_grid_map: dict) -> float:
        """
        Trains and evaluates a neural network on samples from selected grids.

        Returns:
            float: Model accuracy on the test dataset.
        """
        new_data = self.create_new_data_from_selected_grids(selected_grid_map)
        X = new_data.drop('Output', axis=1)
        y = new_data['Output']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.2)

        _, accuracy = model.evaluate(X_test_scaled, y_test)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        return accuracy
