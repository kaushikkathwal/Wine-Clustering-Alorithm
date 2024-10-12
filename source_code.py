#Importing Necessary Libraries
import numpy as np
import pandas as pd


class Matrix:
    # Constructor that loads data from a CSV file and optionally standardizes it

    def __init__(self, filename, standardize=True):
        self.array_2d = self.load_from_csv(filename)  # Load data into a 2D array
        if standardize and self.array_2d is not None:
            self.standardize()  # Standardize the data if needed

    # Load CSV data and convert it into a NumPy array

    def load_from_csv(self, filename):
        try:
            data = pd.read_csv(filename).to_numpy()  # Convert the data into a NumPy array
            return data  # Return the loaded data
        except Exception as e:
            print(f"Error loading file: {e}")  # Print an error message if loading fails
            return None
        
    # Standardize each column of data by adjusting the values so they are more comparable

    def standardize(self):
        for j in range(self.array_2d.shape[1]):  # Go through each column
            column = self.array_2d[:, j]  # Get all the values in that column
            avg = np.mean(column)  # Find the average (mean) of the column
            max_val = np.max(column)  # Get the maximum value
            min_val = np.min(column)  # Get the minimum value
            self.array_2d[:, j] = (column - avg) / (max_val - min_val)  # Standardize the column values

    # Calculate the Euclidean distance from one row to all rows in another matrix

    def get_distance(self, other_matrix, row_index):
        row = self.array_2d[row_index]  # Get the row for which we're calculating distances
        distances = np.sqrt(np.sum((other_matrix.array_2d - row) ** 2, axis=1))  # Calculate distances
        return distances.reshape(-1, 1)  # Return distances as a column

    # Calculate the Euclidean distance but with weights applied to each feature

    def get_weighted_distance(self, other_matrix, weights, row_index):
        row = self.array_2d[row_index]  # Get the specific row to compare
        weighted_diffs = (other_matrix.array_2d - row) * weights.array_2d  # Apply the weights
        distances = np.sqrt(np.sum(weighted_diffs ** 2, axis=1))  # Calculate weighted distances
        return distances.reshape(-1, 1)  # Return as a column of distances

# Count how many times each value appears in S (used for counting group sizes)

def get_count_frequency(S):
    unique, counts = np.unique(S, return_counts=True)  # Get unique values and their counts
    return dict(zip(unique.flatten(), counts))  # Return as a dictionary of counts

# Generate random weights that sum to 1 (used for clustering)

def get_initial_weights(num_columns):
    weights = np.random.rand(1, num_columns)  # Create random weights
    return weights / np.sum(weights)  # Normalize them so they sum to 1

# Find the centroids (average points) of the clusters

def get_centroids(data, S, K):
    centroids = np.zeros((K, data.array_2d.shape[1]))  # Initialize centroids
    for k in range(K):
        # Calculate the mean (centroid) for each cluster
        centroids[k] = np.mean(data.array_2d[S.flatten() == (k + 1)], axis=0)
    return centroids  # Return the centroids

# Calculate how spread out (separation) points are within each cluster

def get_separation_within(data, centroids, S, K):
    separation = np.zeros((1, data.array_2d.shape[1]))  # Initialize separation array
    for j in range(data.array_2d.shape[1]):  # Go through each column (feature)
        for k in range(K):
            # Find all the points in cluster k and calculate their squared distance from the centroid
            cluster_points = data.array_2d[S.flatten() == (k + 1)]
            separation[0, j] += np.sum((cluster_points[:, j] - centroids[k, j]) ** 2)
    return separation  # Return the separation values

# Calculate how far apart the centroids of different clusters are

def get_separation_between(data, centroids, S, K):
    separation = np.zeros((1, data.array_2d.shape[1]))  # Initialize separation array
    counts = np.array([np.sum(S.flatten() == (k + 1)) for k in range(K)])  # Get the number of points in each cluster
    for j in range(data.array_2d.shape[1]):  # Go through each column
        for k in range(K):
            # Calculate the separation between the cluster centroid and the global mean
            separation[0, j] += counts[k] * (centroids[k, j] - np.mean(data.array_2d[:, j])) ** 2
    return separation  # Return the separation values

# Assign each data point to a group (cluster) based on the nearest centroid

def get_groups(data, K, verbose=False):
    if K < 2 or K >= data.array_2d.shape[0]:
        raise ValueError(f"K must be in the range [2, {data.array_2d.shape[0] - 1}]")  # Ensure valid number of clusters

    weights = get_initial_weights(data.array_2d.shape[1])  # Get initial weights for clustering
    S = np.zeros((data.array_2d.shape[0], 1))  # Initialize group assignments

    # Randomly select initial centroids from the data
    initial_indices = np.random.choice(data.array_2d.shape[0], K, replace=False)
    centroids = data.array_2d[initial_indices]  # Set initial centroids

    if verbose:
        print(f"Initial centroids:\n{centroids}")  # Print centroids if verbose mode is on

    while True:
        # Assign each point to the closest centroid
        for i in range(data.array_2d.shape[0]):
            distances = np.sqrt(np.sum((centroids - data.array_2d[i]) ** 2, axis=1))  # Calculate distances to centroids
            S[i] = np.argmin(distances) + 1  # Assign to the nearest centroid

        new_centroids = get_centroids(data, S, K)  # Recalculate centroids
        if np.array_equal(new_centroids, centroids):  # Stop if centroids don't change
            break
        centroids = new_centroids  # Update centroids for the next iteration
        if verbose:
            print(f"Updated centroids:\n{centroids}")  # Print new centroids if verbose mode is on

    return S  # Return the group assignments

# Update the weights based on how spread out (separated) the clusters are

def get_new_weights(data, centroids, old_weights, S, K):
    a = get_separation_within(data, centroids, S, K)  # Get separation within clusters
    b = get_separation_between(data, centroids, S, K)  # Get separation between clusters
    new_weights = old_weights + b / a  # Adjust weights based on separations
    return new_weights / np.sum(new_weights)  # Normalize the weights to sum to 1

# Test the clustering with different numbers of groups (K) and iterations

def run_test(): 
    m = Matrix('Data.csv')  # Load data from 'Data.csv'
    for k in range(2, 11):  # Test for K = 2 to 10 clusters
        for i in range(20):  # Repeat clustering 20 times for each K
            S = get_groups(m, k)  # Get the group assignments
            print(str(k) + " = " + str(get_count_frequency(S)))  # Print the number of points in each group


run_test()
