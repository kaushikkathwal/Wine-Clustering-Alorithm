# Wine-Clustering-Alorithm

## Introduction
In this project, I wrote code to group similar wines based on their characteristics, such as alcohol content, acidity, and color intensity. The goal was to implement a clustering algorithm that uses a dataset containing various attributes of wines to effectively identify and group similar entities.

## Features
- Loads wine data from a CSV file.
- Standardizes data to ensure comparability of features.
- Calculates Euclidean and weighted distances between wines.
- Implements a clustering algorithm to group similar wines.
- Provides insights into the distribution of wines across different groups.
## Technologies Used
- Python
- NumPy
- Pandas

Functions Overview
Matrix Class: Core implementation for loading and processing wine data.
    -__init__(filename, standardize=True): Initializes the class and loads data.
    -load_from_csv(filename): Loads data from a CSV file.
    -standardize(): Standardizes the data for better clustering performance.
    -get_distance(other_matrix, row_index): Calculates Euclidean distance.
    -get_weighted_distance(other_matrix, weights, row_index): Calculates weighted distance.
Standalone Functions:
    -get_count_frequency(S): Counts frequency of unique values in the dataset.
    -get_initial_weights(num_columns): Generates random weights for clustering.
    -get_centroids(data, S, K): Computes cluster centroids.
    -get_groups(data, K): Assigns data points to clusters.
Conclusion
The code effectively groups similar wines based on their attributes using a clustering algorithm. The structured approach allows for data loading, standardization, distance calculations, and clustering, providing valuable insights into wine characteristics.

