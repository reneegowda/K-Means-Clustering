"""
Cluster class for k-Means clustering. With this done, the visualization can
display the centroid of a single cluster.

Renee Gowda (rsg276) and Muskan Gupta (mg2479)
November 15th, 2024
"""

import math
import random
import numpy

# For accessing the previous parts of the assignment
import a6dataset

# TASK 2: CLUSTER
class Cluster(object):
    """
    A class representing a cluster, a subset of the points in a dataset.

    A cluster is represented as a list of integers that give the indices in the
    dataset of the points contained in the cluster.
    """

    # Part A
    def getIndices(self):
        """
        Returns the indices of points in this cluster.

        This method allows direct access to the indices of points within the cluster.
        Note that changes to the returned list will reflect on the cluster directly.
        """
        return self._indices

    def getCentroid(self):
        """
        Returns a COPY of the centroid of this cluster.

        This ensures that external modifications to the returned list do not affect
        the actual centroid stored within the cluster.
        """
        return self._centroid.copy()

    def __init__(self, dset, centroid):
        """
        Initializes a new empty cluster with a given dataset and centroid.

        - Ensures that the centroid and dataset meet the required preconditions.
        - Creates an empty list for storing indices of points within the cluster.
        """
        # Preconditions: Validating input parameters
        assert a6dataset.is_point(centroid)
        assert isinstance(dset, a6dataset.Dataset)
        assert len(centroid) == dset.getDimension()

        # Attribute initialization
        self._indices = []  # Stores indices of points in the cluster
        self._dataset = dset  # Reference to the dataset
        self._centroid = centroid.copy()  # Copy of the provided centroid

    def addIndex(self, index):
        """
        Adds the given dataset index to this cluster.

        - Checks if the index is already present in the cluster to avoid duplicates.
        - Appends the index if it is not already present.
        """
        assert isinstance(index, int)
        assert index >= 0
        assert index < self._dataset.getSize()

        if index not in self._indices:
            self._indices.append(index)

    def clear(self):
        """
        Removes all points from this cluster while keeping the centroid unchanged.
        """
        self._indices = []  # Reset the list of indices

    def getContents(self):
        """
        Returns a new list containing copies of the points in this cluster.

        - Uses the indices to fetch points from the dataset.
        - Ensures that each point is a copy to prevent unintended modifications.
        """
        contents = []
        for index in self._indices:
            contents.append(self._dataset.getPoint(index).copy())
        return contents

    # Part B
    def distance(self, point):
        """
        Returns the Euclidean distance between the given point and the cluster's centroid.

        - Iterates over each dimension to calculate the squared difference.
        - Uses the square root of the sum of squared differences to compute the distance.
        """
        assert a6dataset.is_point(point)
        assert len(point) == len(self.getCentroid())

        cent = self.getCentroid()
        sum = 0
        for i in range(len(point)):
            temp = float(cent[i]) - float(point[i])  # Difference in each dimension
            temp = pow(temp, 2)  # Squared difference
            sum += temp
        return math.sqrt(sum)  # Euclidean distance

    def getRadius(self):
        """
        Returns the maximum distance from any point in the cluster to the centroid.

        - Iterates over all points in the cluster.
        - Computes the distance to the centroid for each point.
        - Tracks the maximum distance encountered.
        """
        max = 0  # Initialize maximum distance
        contents = self.getContents()
        for i in range(len(contents)):
            temp = self.distance(contents[i])
            if temp > max:  # Update maximum if the current distance is greater
                max = temp
        return max

    def update(self):
        """
        Updates the cluster's centroid to the average of its points.

        - Computes the average for each coordinate separately.
        - Uses numpy.allclose to determine whether the centroid has changed.
        - If there are no points in the cluster, the centroid remains unchanged.
        """
        old_centroid = self.getCentroid()  # Current centroid before updating
        dim = len(old_centroid)  # Number of dimensions
        new_centroid = [0.0] * dim  # Initialize a new centroid with zeros

        ind = self.getIndices()  # Get indices of points in the cluster
        dset = self._dataset  # Reference to the dataset

        if ind == []:  # If the cluster is empty, no update is performed
            return True

        # Compute the average for each dimension
        for i in range(dim):
            total = 0
            for j in range(len(ind)):
                index = ind[j]
                point = dset.getPoint(index)
                total += point[i]
            new_centroid[i] = total / len(ind)

        # Update the centroid and check for stability
        self._centroid = new_centroid
        return numpy.allclose(old_centroid, new_centroid)

    # PROVIDED METHODS: Do not modify!
    def __str__(self):
        """
        Returns a String representation of the centroid of this cluster.
        """
        return str(self._centroid) + ':' + str(self._indices)

    def __repr__(self):
        """
        Returns an unambiguous representation of this cluster.
        """
        return str(self.__class__) + str(self)
