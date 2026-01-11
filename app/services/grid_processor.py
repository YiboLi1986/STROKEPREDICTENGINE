import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from grid_merger import GridMerger
from interval_divider import IntervalDivider

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

class GridProcessor:  # Consider changing from GridSelector to GridProcessor to reflect its broader functionality
    def __init__(self, final_grid_map, histograms, file_path, len_threshold, len_threshold_0, len_threshold_1, num_threshold_0, num_threshold_1):
        """
        Initializes the GridProcessor class for selecting and analyzing grids based on sample counts and thresholds.

        Args:
            final_grid_map (dict): A dictionary where grid IDs map to lists of sample IDs after merging operations.
            histograms (dict): A dictionary mapping grid IDs to histograms of 0s and 1s sample counts.
            file_path (str): Path to the original dataset file.
            len_threshold (int): The sample count threshold for grid selection.
            len_threshold_0 (int): The threshold for the number of 0-label samples.
            len_threshold_1 (int): The threshold for the number of 1-label samples.
            num_threshold_0 (float): The proportion threshold for 0-label samples.
            num_threshold_1 (float): The proportion threshold for 1-label samples.
        """
        self.final_grid_map = final_grid_map
        self.histograms = histograms
        self.data = pd.read_excel(file_path)
        self.len_threshold = len_threshold
        self.len_threshold_0 = len_threshold_0
        self.len_threshold_1 = len_threshold_1
        self.num_threshold_0 = num_threshold_0
        self.num_threshold_1 = num_threshold_1

    def select_grids(self):
        """
        Selects grids based on the sample count threshold.

        Iterates over final_grid_map and selects grids where the number of samples
        is greater than or equal to the specified len_threshold.

        Returns:
            dict: A dictionary of selected grid IDs and their sample lists.
        """
        selected_grids = {}
        for grid_id, sample_ids in self.final_grid_map.items():
            if len(sample_ids) >= self.len_threshold:
                selected_grids[grid_id] = sample_ids
        return selected_grids

    def select_majority_grids(self):
        """
        Selects grids based on the majority of 0s or 1s in the histograms and filters based on thresholds.

        Applies thresholds for both the proportion and count of 0-label and 1-label samples in each grid.

        Returns:
            dict: A dictionary of selected grid IDs and their sample lists that meet the conditions.
        """
        new_final_grid_map = {}
        for grid_id, sample_ids in self.final_grid_map.items():
            if not sample_ids:  # Check if sample_ids is empty
                continue

            histogram = self.histograms.get(grid_id, {0: 0, 1: 0})  # Retrieve histogram for the grid
            total_samples = len(sample_ids)
            num_zeros = histogram.get(0, 0)
            num_ones = histogram.get(1, 0)

            # Calculate the proportion of zeros and ones
            zero_proportion = num_zeros / total_samples if total_samples > 0 else 0
            one_proportion = num_ones / total_samples if total_samples > 0 else 0

            # Apply thresholds to select grids
            if zero_proportion >= self.num_threshold_0 and num_zeros >= self.len_threshold_0:
                new_final_grid_map[grid_id] = sample_ids
            elif one_proportion >= self.num_threshold_1 and num_ones >= self.len_threshold_1:
                new_final_grid_map[grid_id] = sample_ids

        return new_final_grid_map

    def statistics_of_selected_grids(self, selected_grid_map):
        """
        Computes statistics for selected grids including:
        - Total number of grids
        - Total number of samples
        - Total number of samples labeled as 0
        - Total number of samples labeled as 1

        Args:
            selected_grid_map (dict): A dictionary of selected grid IDs and their sample lists.

        Returns:
            tuple: Total number of grids, total samples, total 0-label samples, and total 1-label samples.
        """
        total_grids = len(selected_grid_map)
        total_samples = 0
        total_zeros = 0
        total_ones = 0

        for sample_ids in selected_grid_map.values():
            sample_labels = self.data.loc[sample_ids, 'Output']  # Assuming 'Output' column holds the label

            total_samples += len(sample_ids)
            total_zeros += (sample_labels == 0).sum()
            total_ones += (sample_labels == 1).sum()

        return total_grids, total_samples, total_zeros, total_ones

    def create_new_data_from_selected_grids(self, selected_grid_map):
        """
        Creates a new DataFrame from the selected grid samples.

        Args:
            selected_grid_map (dict): A dictionary of selected grid IDs and their sample lists.

        Returns:
            pd.DataFrame: A new DataFrame containing the selected samples.
        """
        selected_sample_ids = [sample_id for sample_ids in selected_grid_map.values() for sample_id in sample_ids]

        # Filter the data using selected sample IDs
        new_data = self.data.loc[selected_sample_ids]

        return new_data

    def train_and_evaluate_nn(self, selected_grid_map):
        """
        Trains and evaluates a neural network on the samples from the selected grids.

        Args:
            selected_grid_map (dict): A dictionary of selected grid IDs and their sample lists.
        
        Returns:
            float: The accuracy of the model on the test dataset.
        """
        # Create new dataset from the selected grids
        new_data = self.create_new_data_from_selected_grids(selected_grid_map)

        # Prepare the data, assuming 'Output' column contains labels
        X = new_data.drop('Output', axis=1)
        y = new_data['Output']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build a simple neural network model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Sigmoid for binary classification
        ])

        # Compile the model
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.2)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_scaled, y_test)
        print(f"Test accuracy: {accuracy * 100:.2f}%")

        return accuracy


# Initialize GridProcessor and run the main workflow
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), 'data_clean.xlsx')
    divider = 2
    interval_divider = IntervalDivider(file_path, divider)

    intervals = interval_divider.intervals
    grid_map = interval_divider.grid_map
    diff_threshold = 0.3

    merger = GridMerger(grid_map, diff_threshold)
    final_grid_map, histograms = merger.merge_until_threshold(interval_divider.data)

    # Set threshold values
    len_threshold = 2
    len_threshold_0 = 2
    len_threshold_1 = 3
    num_threshold_0 = 0.7
    num_threshold_1 = 0.7

    processor = GridProcessor(final_grid_map, histograms, file_path, len_threshold, len_threshold_0, len_threshold_1, num_threshold_0, num_threshold_1)

    # Print statistics of the initial grid selection
    new_final_grid_map = processor.select_majority_grids()
    total_grids, total_samples, total_zeros, total_ones = processor.statistics_of_selected_grids(new_final_grid_map)
    print(f"Total selected grids: {total_grids}")
    print(f"Total selected samples: {total_samples}")
    print(f"Total 0-label samples: {total_zeros}")
    print(f"Total 1-label samples: {total_ones}")

    # Train and evaluate the neural network
    accuracy = processor.train_and_evaluate_nn(new_final_grid_map)
    print(f"Model accuracy on the test set: {accuracy * 100:.2f}%")
