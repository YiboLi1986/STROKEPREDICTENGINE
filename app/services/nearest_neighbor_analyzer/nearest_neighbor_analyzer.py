import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import faiss

from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


class NearestNeighborAnalyzer:
    """
    A class for finding the nearest neighbors of a given data point and performing statistical analysis.

    This class is designed to handle datasets loaded from an xlsx file. It supports three different
    methods for finding nearest neighbors:
      1. sklearn's NearestNeighbors with configurable distance metrics (e.g., Euclidean, Cosine).
      2. FAISS (Facebook AI Similarity Search) for fast approximate nearest neighbor search.
      3. Precomputed dot products for efficient neighbor search in specific use cases.

    Attributes:
        data (np.ndarray): The feature data extracted from the dataset (excluding the last column).
        labels (np.ndarray): The labels (last column) of the dataset.
        nbrs (NearestNeighbors): Instance of sklearn's NearestNeighbors for nearest neighbor search.
        faiss_index (faiss.Index): FAISS index for nearest neighbor search.
        precomputed_dot_products (np.ndarray): Precomputed dot product matrix for the dataset.

    Methods:
        find_nearest_neighbors_sklearn: Find neighbors using sklearn's NearestNeighbors.
        find_nearest_neighbors_faiss: Find neighbors using FAISS.
        find_nearest_neighbors_dot_product: Find neighbors using precomputed dot products.
        print_nearest_neighbors: Print distances, indices, and data of the nearest neighbors.
    """

    def __init__(self, file_path):
        """
        Initialize the class by reading the dataset from the given xlsx file path.
        :param file_path: Path to the xlsx file containing the dataset.
        """
        dataset = pd.read_excel(file_path).values
        self.data = dataset[:, :-1]  # All columns except the last one as features
        self.labels = dataset[:, -1]  # Last column as labels
        
        self.nbrs = None  # Placeholder for NearestNeighbors model
        self.faiss_index = None  # Placeholder for FAISS index
        self.precomputed_dot_products = None  # Placeholder for precomputed dot products

    def find_nearest_neighbors_sklearn(self, new_data, n_neighbors, metric='euclidean'):
        """
        Find the n nearest neighbors of the given data point using sklearn's NearestNeighbors.
        :param new_data: The new data point as a NumPy array.
        :param n_neighbors: Number of nearest neighbors to find.
        :param metric: Distance metric to use ('euclidean' or 'cosine').
        :return: Distances and indices of the nearest neighbors.
        """
        if self.nbrs is None or self.nbrs.effective_metric_ != metric:
            self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
            self.nbrs.fit(self.data)
        distances, indices = self.nbrs.kneighbors(new_data.reshape(1, -1))
        return distances.flatten(), indices.flatten()

    def find_nearest_neighbors_faiss(self, new_data, n_neighbors):
        """
        Find the n nearest neighbors of the given data point using FAISS.
        :param new_data: The new data point as a NumPy array.
        :param n_neighbors: Number of nearest neighbors to find.
        :return: Distances and indices of the nearest neighbors.
        """
        if self.faiss_index is None:
            d = self.data.shape[1]
            self.faiss_index = faiss.IndexFlatL2(d)  # L2 distance for nearest neighbors
            self.faiss_index.add(self.data.astype('float32'))  # Add data to the index
        distances, indices = self.faiss_index.search(new_data.reshape(1, -1).astype('float32'), n_neighbors)
        return distances.flatten(), indices.flatten()

    def precompute_dot_products(self):
        """
        Precompute dot products for the entire dataset for fast nearest neighbor search.
        """
        self.precomputed_dot_products = np.dot(self.data, self.data.T)

    def find_nearest_neighbors_dot_product(self, new_data, n_neighbors):
        """
        Find the n nearest neighbors using precomputed dot products.
        :param new_data: The new data point as a NumPy array.
        :param n_neighbors: Number of nearest neighbors to find.
        :return: Distances and indices of the nearest neighbors.
        """
        if self.precomputed_dot_products is None:
            self.precompute_dot_products()

        new_dot_product = np.dot(self.data, new_data)  # Dot product of new data with the dataset
        distances = -new_dot_product  # Negative to mimic distance-like behavior
        indices = np.argsort(distances)[:n_neighbors]  # Find indices of n smallest distances
        return distances[indices], indices

    def print_nearest_neighbors(self, method, new_data, n_neighbors, metric='euclidean'):
        """
        Print the nearest neighbors' distances, indices, original data (with feature names), and labels.
        :param method: The method to use ('sklearn', 'faiss', 'dot_product').
        :param new_data: The new data point as a NumPy array.
        :param n_neighbors: Number of nearest neighbors to find.
        :param metric: Distance metric for sklearn method.
        """
        if method == 'sklearn':
            distances, indices = self.find_nearest_neighbors_sklearn(new_data, n_neighbors, metric=metric)
        elif method == 'faiss':
            distances, indices = self.find_nearest_neighbors_faiss(new_data, n_neighbors)
        elif method == 'dot_product':
            distances, indices = self.find_nearest_neighbors_dot_product(new_data, n_neighbors)
        else:
            raise ValueError("Unsupported method. Choose from 'sklearn', 'faiss', or 'dot_product'.")

        feature_names = ['age', 'race', 'sex', 'HTN', 'DM', 'HLD', 'Smoking', 
                        'HxOfStroke', 'HxOfAfib', 'HxOfPsychIllness', 'HxOfESRD', 
                        'HxOfSeizure', 'SBP', 'DBP', 'BloodSugar', 'NIHSS', 'FacialDroop']
        
        print(f"Method: {method}, Metric: {metric}")
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")
        print(f"Nearest Neighbors (data with features):")
        for i, idx in enumerate(indices):
            print(f"Neighbor {i+1}:")
            for feature, value in zip(feature_names, self.data[idx]):
                print(f"  {feature}: {value}")
        print(f"Nearest Neighbors (labels): {self.labels[indices]}")


    def evaluate_density_from_neighbors(self, new_data, neighbors, bandwidth=0.5, threshold=0.01):
        """
        Evaluate if the new data is in a high-density region using KDE.
        :param new_data: The new data point as a NumPy array.
        :param neighbors: The neighbor data points as a NumPy array.
        :param bandwidth: Bandwidth for KDE.
        :param threshold: Density threshold to detect outliers.
        :return: A message indicating if the new data is an outlier based on density.
        """
        # Kernel Density Estimation
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(neighbors)
        log_density = kde.score_samples(new_data.reshape(1, -1))
        density = np.exp(log_density[0])

        if density < threshold:
            return f"New data is likely an outlier (low density: {density:.4f})."
        else:
            return f"New data is in a high-density region (density: {density:.4f})."

    def evaluate_relative_distance_from_neighbors(self, new_data, neighbors):
        """
        Compare the relative distance of the new data with its neighbors.
        :param new_data: The new data point as a NumPy array.
        :param neighbors: The neighbor data points as a NumPy array.
        :return: A message indicating if the new data is far from its neighbors.
        """
        # Calculate distances
        new_to_neighbors = np.mean(np.linalg.norm(neighbors - new_data, axis=1))
        neighbors_to_neighbors = np.mean(pairwise_distances(neighbors))

        ratio = new_to_neighbors / neighbors_to_neighbors
        if ratio > 2:
            return f"New data is far from its neighbors (relative distance ratio: {ratio:.2f})."
        else:
            return f"New data is close to its neighbors (relative distance ratio: {ratio:.2f})."

    def evaluate_label_consistency_from_neighbors(self, neighbor_labels):
        """
        Check the consistency of labels among the nearest neighbors.
        :param neighbor_labels: The labels of the nearest neighbors as a NumPy array.
        :return: A message indicating the consistency of labels.
        """
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        dominant_label_ratio = counts.max() / counts.sum()

        if dominant_label_ratio < 0.6:
            return "Nearest neighbors have diverse labels, suggesting low reference value."
        else:
            return "Nearest neighbors have consistent labels, suggesting high reference value."

    def evaluate_distribution_pca_from_neighbors(self, new_data, neighbors):
        """
        Use PCA to check if the new data falls within the main distribution of its neighbors.
        :param new_data: The new data point as a NumPy array.
        :param neighbors: The neighbor data points as a NumPy array.
        :return: A message indicating if the new data aligns with its neighbors in PCA space.
        """
        # Perform PCA
        pca = PCA(n_components=2)
        transformed_neighbors = pca.fit_transform(neighbors)
        transformed_new = pca.transform(new_data.reshape(1, -1))

        # Check if new data lies within PCA bounds
        x_min, x_max = transformed_neighbors[:, 0].min(), transformed_neighbors[:, 0].max()
        y_min, y_max = transformed_neighbors[:, 1].min(), transformed_neighbors[:, 1].max()

        if (x_min <= transformed_new[0, 0] <= x_max) and (y_min <= transformed_new[0, 1] <= y_max):
            return "New data lies within the main PCA distribution of its neighbors."
        else:
            return "New data is outside the main PCA distribution of its neighbors."

if __name__ == "__main__":
    file_path = "app/services/data/data_clean.xlsx"
    processor = NearestNeighborAnalyzer(file_path)
    new_data = np.array([0.64, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.835, 0.87155963, 0.81651376, 0.4, 0.0])

    n_neighbors = 20 
    processor.print_nearest_neighbors(method='sklearn', new_data=new_data, n_neighbors=n_neighbors, metric='cosine')
    #processor.print_nearest_neighbors(method='faiss', new_data=new_data, n_neighbors=n_neighbors)
    #processor.print_nearest_neighbors(method='dot_product', new_data=new_data, n_neighbors=n_neighbors)

    distances, indices = processor.find_nearest_neighbors_sklearn(new_data, n_neighbors, metric='euclidean')
    neighbors = processor.data[indices]
    neighbor_labels = processor.labels[indices]

    # Evaluate density
    print("Density Evaluation:")
    print(processor.evaluate_density_from_neighbors(new_data, neighbors))

    # Evaluate relative distance
    print("\nRelative Distance Evaluation:")
    print(processor.evaluate_relative_distance_from_neighbors(new_data, neighbors))

    # Evaluate label consistency
    print("\nLabel Consistency Evaluation:")
    print(processor.evaluate_label_consistency_from_neighbors(neighbor_labels))

    # Evaluate PCA distribution
    print("\nPCA Distribution Evaluation:")
    print(processor.evaluate_distribution_pca_from_neighbors(new_data, neighbors))
