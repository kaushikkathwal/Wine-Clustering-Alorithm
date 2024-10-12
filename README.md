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

## Functions Overview

### Matrix Class
The **Matrix** class is the core of the implementation, providing methods for loading, processing, and analyzing wine data.

- **`__init__(filename, standardize=True)`**
  - Initializes the class and loads data from a specified CSV file into a 2D NumPy array.
  - **Parameters**: 
    - `filename`: The path to the CSV file.
    - `standardize`: If set to `True`, standardizes the data upon loading.

- **`load_from_csv(filename)`**
  - Reads data from the specified CSV file and populates the internal array.
  - **Parameters**:
    - `filename`: The path to the CSV file.
  - **Return Value**: A NumPy array containing the data.

- **`standardize()`**
  - Standardizes the internal data array to have a mean of 0 and a standard deviation of 1.
  - **Return Value**: None (modifies the data in place).

- **`get_distance(other_matrix, row_index)`**
  - Calculates the Euclidean distance between a specified row and all rows of another matrix.
  - **Parameters**:
    - `other_matrix`: An instance of the Matrix class.
    - `row_index`: The index of the row in the current matrix.
  - **Return Value**: A column matrix of distances.

- **`get_weighted_distance(other_matrix, weights, row_index)`**
  - Computes the weighted Euclidean distance between a specified row and all rows of another matrix using provided weights.
  - **Parameters**:
    - `other_matrix`: An instance of the Matrix class.
    - `weights`: A matrix containing weights for distance calculation.
    - `row_index`: The index of the row in the current matrix.
  - **Return Value**: A column matrix of weighted distances.

### Standalone Functions
The project also includes several standalone functions that facilitate various tasks in the clustering process:

- **`get_count_frequency(S)`**
  - Counts the frequency of unique values in a dataset.
  - **Parameters**: None (operates on S).
  - **Return Value**: A dictionary mapping each unique element to its frequency count.

- **`get_initial_weights(num_columns)`**
  - Generates random weights that sum to one.
  - **Parameters**:
    - `num_columns`: The number of columns for the weights matrix.
  - **Return Value**: A matrix with one row and `num_columns` that sums to one.

- **`get_centroids(data, S, K)`**
  - Computes the centroids of clusters based on current assignments.
  - **Parameters**:
    - `data`: An instance of the Matrix class containing the data.
    - `S`: The current group assignments.
    - `K`: The number of clusters.
  - **Return Value**: A matrix containing the centroids of the clusters.

- **`get_groups(data, K)`**
  - Assigns each data point to a group based on the nearest centroid.
  - **Parameters**:
    - `data`: An instance of the Matrix class containing the data.
    - `K`: The number of groups to create.
  - **Return Value**: A matrix S containing group assignments.

- **`run_test()`**
  - Runs tests to evaluate the clustering algorithm for different values of K.
  - **Operation**: Loads data, performs clustering for K values ranging from 2 to 10, and prints frequency of group assignments.
 
## Conclusion
The code I wrote successfully implements a clustering algorithm that effectively groups similar wines based on their attributes. By leveraging a structured approach with the Matrix class and various standalone functions, the code can efficiently load data, standardize it, compute distances, and categorize wines into distinct groups. Additionally, the integration of frequency counting provides valuable insights into the distribution of wines across different clusters, enabling a better understanding of the underlying data patterns. This project not only demonstrates the practical application of clustering algorithms but also serves as a foundation for future enhancements and analyses in the field of wine classification.
## Linkedin Post Link:-https://www.linkedin.com/posts/kaushik-kathwal-372769290_datascience-machinelearning-wineindustry-activity-7250860756361994240-P7FS?utm_source=share&utm_medium=member_desktop
