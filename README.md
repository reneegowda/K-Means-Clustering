# K-Means-Clustering
This repository provides a Python implementation of a dataset class tailored for use in k-Means clustering algorithms. The dataset is designed to handle multi-dimensional data points while ensuring efficient data management, robust validation, and proper attribute encapsulation.

Features:
1. Data Validation
Helper Functions: Utility functions like is_point and is_point_list validate data points, ensuring they consist of integers or floats and have consistent dimensions.
2. Encapsulation
Attributes are encapsulated with getter methods (getDimension, getSize, getContents, etc.), maintaining data integrity and preventing unintended modifications.
3. Dataset Manipulation
Add Points: Easily add new data points using the addPoint method. Points are copied before addition to prevent external modification of the original data.
Access Points: Retrieve individual points via the getPoint method, which ensures a safe, independent copy of the data.
4. String Representation
The __str__ method generates a clean string representation of the dataset, formatting points with indices and converting all integers to floats for uniformity.
5. Robust Initialization
Supports initialization with a predefined dataset or an empty dataset. Validates all input to ensure consistency and correctness.

Usage
This class can serve as the foundation for k-Means clustering or other clustering algorithms requiring organized, multi-dimensional data storage and manipulation.

