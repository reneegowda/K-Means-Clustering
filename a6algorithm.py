"""
Primary algorithm for k-Means clustering

Authors: Renee Gowda (rsg276) and Muskan Gupta (mg2479)
Date: November 15th, 2024
"""

import math
import random
import numpy

# For accessing the previous parts of the assignment
import a6dataset
import a6cluster

# TASK 3: ALGORITHM

# Part A: Seed Validation
def valid_seeds(value, size):
    """
    Validates if the given value is a list of seeds suitable for clustering.

    A valid seed list must:
    - Contain k unique integers.
    - Ensure each integer is in the range [0, size-1].
    - Not allow duplicate integers.

    Parameters:
        value: A value to check, can be of any type.
        size: The size of the dataset (int > 0).

    Returns:
        bool: True if the value is a valid list of seeds, False otherwise.
    """
    # Validate size precondition
    assert isinstance(size, int)
    assert size > 0

    # Check if value is a list
    if not isinstance(value, list):
        return False

    # Ensure all elements in the list meet seed requirements
    for i in range(len(value)):
        if not isinstance(value[i], int):  # Check if element is an integer
            return False
        if value[i] < 0 or value[i] > size - 1:  # Check range
            return False
        for j in range(i + 1, len(value)):  # Ensure no duplicates
            if value[i] == value[j]:
                return False

    return True


class Algorithm(object):
    """
    A class to manage and execute the k-means clustering algorithm.

    This class implements the core k-means logic, including:
    - Assigning points to the nearest cluster.
    - Updating cluster centroids.
    - Checking for convergence.
    """
    # Immutable Attributes:
    # - _dataset: The dataset used in the algorithm (Instance of Dataset).
    # - _cluster: A list of Cluster instances representing the clusters.

    # Part B: Accessing Clusters
    def getClusters(self):
        """
        Retrieves the list of clusters.

        The returned list is mutable; any changes to it will directly
        affect the internal state of the Algorithm object.

        Returns:
            list: A list of Cluster instances.
        """
        return self._cluster

    def __init__(self, dset, k, seeds=None):
        """
        Initializes the k-means algorithm with a dataset and cluster count.

        If seed indices are provided, they are used to set initial cluster
        centroids. Otherwise, k random points from the dataset are chosen.

        Parameters:
            dset: The dataset (Instance of Dataset).
            k: Number of clusters (int, 0 < k <= dset.getSize()).
            seeds: Optional list of seed indices (default is None).
        """
        # Validate input preconditions
        assert isinstance(dset, a6dataset.Dataset)
        assert isinstance(k, int)
        assert 0 < k <= dset.getSize()
        assert seeds is None or valid_seeds(seeds, dset.getSize())

        self._dataset = dset
        cent = []

        # Initialize centroids from seeds or randomly
        if seeds is not None:
            for i in range(k):
                index = seeds[i]
                cent.append(dset.getPoint(index))  # Retrieve point at seed index
        else:
            cent = random.sample(dset.getContents(), k)  # Randomly sample k points

        # Create cluster objects
        clust = []
        for i in range(k):
            c = a6cluster.Cluster(dset, cent[i])
            clust.append(c)
        self._cluster = clust

    # Part C: Nearest Cluster Assignment
    def _nearest(self, point):
        """
        Finds the nearest cluster to the given point based on distance.

        Uses the Cluster's distance method to compute the distance
        between the point and cluster centroids.

        Parameters:
            point: A list of numerical values (int or float), with the same
                   dimension as the dataset.

        Returns:
            Cluster: The nearest cluster instance.
        """
        # Validate input preconditions
        assert a6dataset.is_point(point)
        assert len(point) == self._dataset.getDimension()

        clust = self.getClusters()
        min = clust[0].distance(point)  # Initialize with the first cluster
        index = 0

        # Iterate over clusters to find the minimum distance
        for i in range(1, len(clust)):
            c = clust[i]
            dis = c.distance(point)
            if min > dis:  # Update if a closer cluster is found
                index = i
                min = dis

        return clust[index]

    def _partition(self):
        """
        Reassigns all points in the dataset to their nearest cluster.

        This method clears all clusters and then assigns each dataset point
        to the cluster whose centroid is nearest.
        """
        clust = self.getClusters()

        # Clear all clusters before reassignment
        for j in range(len(clust)):
            clust[j].clear()

        dset = self._dataset

        # Assign each point to its nearest cluster
        for i in range(dset.getSize()):
            point = dset.getPoint(i)
            c = self._nearest(point)
            c.addIndex(i)

        # Part D: Updating Centroids
    def _update(self):
        """
        Returns True if all centroids are unchanged after an update; False otherwise.

        This method first updates the centroids of all clusters'. When it is done,
        it checks whether any of them have changed. It returns False if just one
        has changed. Otherwise, it returns True.
        """

        # Initialize a variable to track whether centroids have changed
        change = True

        # Iterate over each cluster to update its centroid
        for clust in self.getClusters():
            # Call the update method for the cluster
            # If any cluster's update method returns False, change is set to False
            change = clust.update()

        # Return the final result: True if no centroids changed, False otherwise
        return change

    def step(self):
        """
        Returns True if the algorithm converges after one step; False otherwise.

        This method performs one cycle of the k-means algorithm. It then checks
        if the algorithm has converged and returns the appropriate result (True
        if converged, false otherwise).
        """

        # Partition the dataset by assigning each point to its nearest cluster
        self._partition()

        # Update the centroids of the clusters and check for convergence
        self._update()

    def run(self, maxstep):
        """
        Continues clustering until either it converges or performs maxstep steps.

        After the maxstep call to step, if this calculation did not converge,
        this method will stop.

        Parameter maxstep: The maximum number of steps to perform
        Precondition: maxstep is an int >= 0
        """

        # Validate the input precondition for maxstep
        assert isinstance(maxstep, int)
        assert maxstep >= 0

        # Perform up to maxstep iterations of the algorithm
        for i in range(maxstep):
            # Perform a single step of the algorithm
            s = self.step()

            # If the algorithm converges during this step, terminate early
            if s:
                return

        # If the loop completes without convergence, simply return
        return
