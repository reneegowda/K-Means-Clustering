"""
Dataset for k-Means clustering

Renee Gowda (rsg276) and Muskan Gupta (mg2379)
November 15th, 2024
"""

import math
import random
import numpy

# TASK 0: HELPERS TO CHECK PRECONDITIONS
def is_point(value):
    """
    Returns True if value is a list that only contains ints or floats

    Parameter value: a value to check
    Precondition: value can be anything
    """
    # Ensure the input is a list
    if not isinstance(value, list):
        return False
    # Check that all elements in the list are int or float
    for i in range(len(value)):
        if not isinstance(value[i], (int, float)):
            return False
    return True

def is_point_list(value):
    """
    Returns True if value is a list of points (int/float lists)

    This function also checks that all points in value have the same dimension.

    Parameter value: a value to check
    Precondition: value can be anything
    """
    # Ensure the input is a list
    if not isinstance(value, list):
        return False

    # Check that all elements in the list are valid points
    for i in range(len(value)):
        if not is_point(value[i]):
            return False

    # Ensure all points have the same length (dimension)
    length = len(value[0])
    for i in range(len(value)):
        if len(value[i]) != length:
            return False
    return True

# TASK 1: DATASET
class Dataset(object):
    """
    A class representing a dataset for k-means clustering.

    The data is stored as a list of points (int/float lists). All points have
    the same number of elements which is the dimension of the dataset.

    None of the attributes should be accessed directly outside of the class
    Dataset (e.g. in the methods of class Cluster or KMeans). Instead, this class
    has getter and setter style methods (with the appropriate preconditions) for
    modifying these values.
    """
    # IMMUTABLE ATTRIBUTES (Fixed after initialization)
    # Attribute _dimension: The point dimension for this dataset
    # Invariant: _dimension is an int > 0.
    #
    # MUTABLE ATTRIBUTES (Can be changed at any time, via addPoint)
    # Attribute _contents: The dataset contents
    # Invariant: _contents is a table of numbers (float or int), possibly empty.
    # Each row of _contents is a list of size _dimension

    # Part A
    # Getters for encapsulated attributes
    def getDimension(self):
        """
        Returns the point dimension of this dataset
        """
        return self._dimension

    def getSize(self):
        """
        Returns the number of points in this dataset.
        """
        if self._contents is None:
            return 0
        return len(self._contents)

    def getContents(self):
        """
        Returns the contents of this dataset as a list of points.

        This method returns the contents directly (not a copy). Any changes made
        to this list will modify the dataset. If you want to access the dataset
        but want to protect yourself from modifying the data, use getPoint()
        instead.
        """
        if self._contents is None:
            return []
        return self._contents.copy()

    def __init__(self, dim, contents=None):
        """
        Initializes a dataset for the given point dimension.

        The optional parameter contents is the initial value of the dataset.
        When initializing the dataset, it creates a COPY of the list contents.
        If contents is None, the dataset starts off empty. The parameter contents
        is None by default.

        Parameter dim: The dimension of the dataset
        Precondition: dim is an int > 0

        Parameter contents: the dataset contents
        Precondition: contents is either None or it is a table of numbers (int
        or float). If contents is not None, then contents is not empty, and the
        number of columns is equal to dim.
        """
        # Validate the dimension parameter
        assert isinstance(dim, int) and dim > 0
        # Validate the contents parameter, if provided
        if contents is not None:
            assert is_point_list(contents)

        self._dimension = dim
        # Initialize the dataset contents
        if contents is None:
            self._contents = []
        else:
            self._contents = contents.copy()

    def getPoint(self, i):
        """
        Returns a COPY of the point at index i in this dataset.

        Often, we want to access a point in the dataset but want a copy to
        ensure that we do not accidentally modify the dataset. That is the
        purpose of this method.

        If you actually want to modify the dataset, use the method getContents().
        That returns the list storing the dataset, and any changes to that list
        will alter the dataset.

        Parameter i: the index position of the point
        Precondition: i is an int that refers to a valid position in 0..getSize()-1
        """
        # Validate the index parameter
        assert isinstance(i, int)
        assert i < self.getSize()

        # Return a copy of the requested point
        temp = self._contents
        return temp[i].copy()

    def addPoint(self, point):
        """
        Adds a COPY of point at the end of _contents.

        This method does not add the point directly. It adds a copy of the point.

        Parameter point: The point to add to the dataset
        Precondition: point is a list of int/float. The length of point is equal
        to getDimension().
        """
        # Validate the point parameter
        assert is_point(point)
        assert len(point) == self._dimension

        # Add a copy of the point to the dataset
        new_point = point.copy()
        self._contents = self.getContents()
        self._contents.append(new_point)

    # Part B
    def __str__(self):
        """
        Returns a string representation of this dataset.

        The string returned should be formatted with each point on a line (so
        there is a newline between each point), with the index of each point
        at the start of the line. The index and the point are separated by a
        colon and a space. Finally, there should be NO spaces after any of the
        commas in the point (this is not the default).

        In addition, any ints should be cast to a float before conversion
        to a string.

        Example: Suppose the contents of this dataset is

            [[1.0, 2], [3.0, 4.0], [5, 6.0]]

        In that case, this method would produce the string

            '0: [1.0,2.0]\n1: [3.0,4.0]\n2: [5.0,6.0]'

        See the assignment instructions for more details.
        """
        # Handle the case of an empty dataset
        if self._contents == []:
            return ''
        # Build the string representation
        total = '0: ['
        for i in range(self.getSize()):
            for j in range(len(self.getPoint(i))):
                temp = self.getPoint(i)
                total = total + str(float(temp[j]))
                if j < (len(temp) - 1):
                    total += ','
            total += ']'
            if i < (self.getSize() - 1):
                total += '\n' + str(i + 1) + ': ['
        return total
