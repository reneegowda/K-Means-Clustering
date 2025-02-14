"""
Unit tests for k-means clustering
"""

import introcs
import random
import numpy
import tools
import os, os.path

# The modules to test.
import a6dataset
import a6cluster
import a6algorithm

# Helper function for latter tests
TEST_FILE = 'data/candy.csv'


def test_point():
    print('  Testing function is_point')

    # TEST CASE 1
    item = [0.0,0.0,0.0]
    introcs.assert_true(a6dataset.is_point(item))

    # TEST CASE 2
    item = [0.0,2.0]
    introcs.assert_true(a6dataset.is_point(item))

    # TEST CASE 3
    item = [1,0.5,-2]
    introcs.assert_true(a6dataset.is_point(item))

    # TEST CASE 4
    item = [0.0,1.0, '0.0']
    introcs.assert_false(a6dataset.is_point(item))

    # TEST CASE 5
    item = 4
    introcs.assert_false(a6dataset.is_point(item))

    print('  function is_point appears correct')
    print()

def test_point_list():
    print('  Testing function is_point_list')

    # TEST CASE 1
    items = [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    introcs.assert_true(a6dataset.is_point_list(items))

    # TEST CASE 2
    items = [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0],[0.0,0.0,1.0]]
    introcs.assert_false(a6dataset.is_point_list(items))

    # TEST CASE 3
    items = [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0, '0.0'], [0.0,0.0,1.0]]
    introcs.assert_false(a6dataset.is_point_list(items))

    # TEST CASE 4
    items = [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0], 2, [0.0,0.0,1.0]]
    introcs.assert_false(a6dataset.is_point_list(items))

    # TEST CASE 5
    items = [[0.0,0.0,0.0],[1.0,0.0,0.0],'(0.0,1.0,0.0)', [0.0,0.0,1.0]]
    introcs.assert_false(a6dataset.is_point_list(items))

    # TEST CASE 6
    items = 4
    introcs.assert_false(a6dataset.is_point_list(items))
    print('  function is_point_list appears correct')
    print()


def test_dataset_a():
    """
    Tests Part A of the Dataset class.
    """
    print('  Testing Part A of class Dataset')

    # TEST CASE 1
    # Create and test an empty dataset
    dset1 = a6dataset.Dataset(3)
    introcs.assert_equals(3,dset1.getDimension())
    introcs.assert_equals(0,dset1.getSize())

    # We use this assert function to compare lists
    assert_point_sets_equal([],dset1.getContents())

    print('    Default initialization looks okay')

    # TEST CASE 2
    # Create and test a non-empty dataset
    items = [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    dset2 = a6dataset.Dataset(3,items)
    introcs.assert_equals(3,dset2.getDimension())
    introcs.assert_equals(4,dset2.getSize())

    # Check that contents is initialized correctly
    # Make sure items is COPIED
    assert_point_sets_equal(items,dset2.getContents())
    introcs.assert_false(dset2.getContents() is items)

    print('    User-provided initialization looks okay')

    # Check that getPoint() is correct AND that it copies
    assert_points_equal([0.0,1.0,0.0],dset2.getPoint(2))

    print('    Method Dataset.getPoint looks okay')

    # Add something to the dataset (and check it was added)
    dset1.addPoint([0.0,0.5,4.2])
    assert_point_sets_equal([[0.0,0.5,4.2]],dset1.getContents())
    assert_points_equal([0.0,0.5,4.2],dset1.getPoint(0))

    extra = [0.0,0.5,4.2]
    dset2.addPoint(extra)
    items.append(extra)
    assert_point_sets_equal(items,dset2.getContents())
    introcs.assert_false(dset2.getPoint(-1) is extra)
    print('    Method Dataset.addPoint looks okay')
    print('  Part A of class Dataset appears correct')
    print()


def test_dataset_b():
    """
    Tests Part B of the Dataset class.
    """
    print('  Testing Part B of class Dataset')

    # TEST CASE 1
    items = [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    dset1 = a6dataset.Dataset(3,items)
    output = '0: [0.0,0.0,0.0]\n1: [1.0,0.0,0.0]\n2: [0.0,1.0,0.0]\n3: [0.0,0.0,1.0]'
    introcs.assert_equals(output,str(dset1))

    # TEST CASE 2
    items = [[0.0,0.0],[0.0,1.0],[0.0,0.0],[1.0,0.0],[0.0,0.0],[0.0,1.0]]
    dset2 = a6dataset.Dataset(2,items)
    output = '0: [0.0,0.0]\n1: [0.0,1.0]\n2: [0.0,0.0]\n3: [1.0,0.0]\n4: [0.0,0.0]\n5: [0.0,1.0]'
    introcs.assert_equals(output,str(dset2))

    # TEST CASE 2
    items = [[0.0],[0.5],[1.0],[1.5]]
    dset2 = a6dataset.Dataset(1,items)
    output = '0: [0.0]\n1: [0.5]\n2: [1.0]\n3: [1.5]'
    introcs.assert_equals(output,str(dset2))

    # TEST CASE 4
    items = [[1,0.5,2],[1.5,-3,4.0]]
    dset4 = a6dataset.Dataset(3,items)
    output = '0: [1.0,0.5,2.0]\n1: [1.5,-3.0,4.0]'
    introcs.assert_equals(output,str(dset4))

    # TEST CASE 5
    # Create and test an empty dataset
    dset5 = a6dataset.Dataset(3)
    introcs.assert_equals('',str(dset5))

    print('    Method Dataset.__str__ looks okay')
    print('  Part B of class Dataset appears correct')
    print()


def test_cluster_a():
    """
    Tests Part A of the Cluster class assignment.
    """
    print('  Testing Part A of class Cluster')

    # TEST CASE 1
    # Create and test a cluster (always empty)
    dset = a6dataset.Dataset(3)
    point = [0.0,1.0,0.0]
    cluster1 = a6cluster.Cluster(dset, point)

    # Compare centroid and contents
    assert_points_equal(point,cluster1.getCentroid())
    introcs.assert_equals([],cluster1.getIndices())

    print('    Basic cluster methods look okay')

    # Add something to cluster (and check it was added)
    extra = [[0.0,0.5,4.2],[0.0,1.0,0.0]]
    dset.addPoint(extra[0])
    dset.addPoint(extra[1])
    cluster1.addIndex(1)
    introcs.assert_equals([1],cluster1.getIndices())
    cluster1.addIndex(0)
    introcs.assert_equals([1,0],cluster1.getIndices())
    # Make sure we can handle duplicates!
    cluster1.addIndex(1)
    introcs.assert_equals([1,0],cluster1.getIndices())

    print('    Method Cluster.addIndex look okay')

    # And clear it
    contents = cluster1.getContents()
    introcs.assert_equals(2,len(contents))
    assert_points_equal(extra[1],contents[0])
    assert_points_equal(extra[0],contents[1])

    print('    Method Cluster.getContents look okay')

    # And clear it
    cluster1.clear()
    introcs.assert_equals([],cluster1.getIndices())

    print('    Method Cluster.clear look okay')
    print('  Part A of class Cluster appears correct')
    print()


def test_cluster_b():
    """
    Tests Part B of the Cluster class assignment.
    """
    print('  Testing Part B of class Cluster')

    # A dataset with four points
    items = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]]
    dset = a6dataset.Dataset(3, items)

    # Create some clusters
    cluster1 = a6cluster.Cluster(dset, [0.0,0.0,0.2])
    cluster2 = a6cluster.Cluster(dset, [0.5,0.5,0.0])
    cluster3 = a6cluster.Cluster(dset, [0.0,0.0,0.5])

    # TEST CASE 1 (distance)
    dist = cluster2.distance([1.0,0.0,-1.0])
    introcs.assert_floats_equal(1.22474487139,dist)

    # TEST CASE 2 (distance)
    dist = cluster2.distance([0.5,0.5,0.0])
    introcs.assert_floats_equal(0.0,dist)

    # TEST CASE 3 (distance)
    dist = cluster3.distance([0.5,0.0,0.5])
    introcs.assert_floats_equal(0.5,dist)
    print('    Method Cluster.distance() looks okay')

    # Add some indices
    cluster1.addIndex(0)
    cluster1.addIndex(1)
    cluster2.addIndex(0)
    cluster2.addIndex(1)
    cluster3.addIndex(0)
    cluster3.addIndex(1)
    cluster3.addIndex(2)

    # TEST CASE 1 (radius)
    rads = cluster1.getRadius()
    introcs.assert_floats_equal(1.0198039,rads)

    # TEST CASE 2 (radius)
    rads = cluster2.getRadius()
    introcs.assert_floats_equal(0.7071068,rads)

    # TEST CASE 3 (radius)
    rads = cluster3.getRadius()
    introcs.assert_floats_equal(1.1180340,rads)
    print('    Method Cluster.getRadius() looks okay')

    # TEST CASE 1 (updateCentroid): centroid remains the same
    stable = cluster2.update()
    assert_points_equal([0.5, 0.5, 0.0], cluster2.getCentroid())
    introcs.assert_true(stable)

    # TEST CASE 2 (updateCentroid): centroid changes
    cluster2.addIndex(2)
    cluster2.addIndex(3)
    stable = cluster2.update()
    assert_points_equal([0.25, 0.25, 0.25], cluster2.getCentroid())
    introcs.assert_false(stable)
    # updating again without changing points: centroid stable
    stable = cluster2.update()
    assert_points_equal([0.25, 0.25, 0.25], cluster2.getCentroid())
    introcs.assert_true(stable)

    print('    Method Cluster.update() looks okay')
    print('  Part B of class Cluster appears correct')
    print()


def test_valid_seeds():
    print('  Testing function valid_seeds')

    # TEST CASE 1
    items = [0,3,7,5,2]
    introcs.assert_true(a6algorithm.valid_seeds(items,8))

    # TEST CASE 2
    introcs.assert_false(a6algorithm.valid_seeds(items,7))

    # TEST CASE 3
    items = [0,3,7,5,3]
    introcs.assert_false(a6algorithm.valid_seeds(items,8))

    # TEST CASE 4
    items = [0,3,7,5,2.0]
    introcs.assert_false(a6algorithm.valid_seeds(items,8))

    # TEST CASE 5
    items = [0,3,7,'5',2]
    introcs.assert_false(a6algorithm.valid_seeds(items,8))

    # TEST CASE 6
    items = 4
    introcs.assert_false(a6algorithm.valid_seeds(items,8))
    print('  function valid_seeds appears correct')
    print()


def test_algorithm_a():
    """
    Tests Part A of the Algorithm class.
    """
    print('  Testing Part A of class Algorithm')

    # A dataset with four points almost in a square
    items = [[0.,0.], [10.,1.], [10.,10.], [0.,9.]]
    dset = a6dataset.Dataset(2, items)

    # Test creating a clustering with random seeds
    km = a6algorithm.Algorithm(dset, 3)
    # Should have 3 clusters
    introcs.assert_equals(len(km.getClusters()), 3)
    for clust in km.getClusters():
        # cluster centroids should have been chosen from items
        introcs.assert_true(clust.getCentroid() in items)
        # cluster centroids should be distinct (since items are)
        for clust2 in km.getClusters():
            if clust2 is not clust:
                assert_points_not_equal(clust.getCentroid(), clust2.getCentroid())

    print('    Random Algorithm initialization looks okay')

    # Clusterings of that dataset, with two and three deterministic clusters
    km = a6algorithm.Algorithm(dset, 2, [0,2])
    assert_points_equal(items[0], km.getClusters()[0].getCentroid())
    assert_points_equal(items[2], km.getClusters()[1].getCentroid())
    km = a6algorithm.Algorithm(dset, 3, [0,2,3])
    assert_points_equal(items[0], km.getClusters()[0].getCentroid())
    assert_points_equal(items[2], km.getClusters()[1].getCentroid())
    assert_points_equal(items[3], km.getClusters()[2].getCentroid())

    # Try it on a file
    file = os.path.join(os.path.split(__file__)[0],TEST_FILE)
    data = a6dataset.Dataset(4, tools.data_for_file(file))
    km2  = a6algorithm.Algorithm(data, 3, [23, 54, 36])
    assert_points_equal([0.38, 0.94, 0.53, 0.07], km2.getClusters()[0].getCentroid())
    assert_points_equal([0.84, 0.88, 0.04, 0.86], km2.getClusters()[1].getCentroid())
    assert_points_equal([0.8, 0.4, 0.23, 0.33],   km2.getClusters()[2].getCentroid())

    print('    Seeded Algorithm initialization looks okay')
    print('  Part A of class Algorithm appears correct')
    print()


def test_algorithm_b():
    """
    Tests Part B of the Algorithm class.
    s"""
    # This function tests the methods _nearest and _partition, both of which are hidden
    # methods. Normally it's not good form to directly call these methods from outside
    # the class, but we make an exception for testing code, which often has to be more
    # tightly integrated with the implementation of a class than other code that just
    # uses the class.
    print('  Testing Part B of class Algorithm')
    # Reinitialize data set
    items = [[0.,0.], [10.,1.], [10.,10.], [0.,9.]]
    dset = a6dataset.Dataset(2, items)
    km1 = a6algorithm.Algorithm(dset, 2, [0,2])
    km2 = a6algorithm.Algorithm(dset, 3, [0,2,3])

    nearest = km1._nearest([1.,1.])
    introcs.assert_true(nearest is km1.getClusters()[0])

    nearest = km1._nearest([1.,10.])
    introcs.assert_true(nearest is km1.getClusters()[1])

    nearest = km2._nearest([1.,1.])
    introcs.assert_true(nearest is km2.getClusters()[0])

    nearest = km2._nearest([1.,10.])
    introcs.assert_true(nearest is km2.getClusters()[2])
    print('    Method Algorithm._nearest() looks okay')

    # Testing partition()
    # For this example points 0 and 3 are closer, as are 1 and 2
    km1._partition()
    introcs.assert_equals(set([0,3]), set(km1.getClusters()[0].getIndices()))
    introcs.assert_equals(set([1,2]), set(km1.getClusters()[1].getIndices()))
    # partition and repeat -- should not change clusters.
    km1._partition()
    introcs.assert_equals(set([0,3]), set(km1.getClusters()[0].getIndices()))
    introcs.assert_equals(set([1,2]), set(km1.getClusters()[1].getIndices()))

    # Reset the cluster centroids; now it changes
    cluster = km1.getClusters()
    cluster[0]._centroid = [5.0, 10.0]
    cluster[1]._centroid = [0.0, 2.0]
    km1._partition()
    introcs.assert_equals(set([2,3]), set(km1.getClusters()[0].getIndices()))
    introcs.assert_equals(set([0,1]), set(km1.getClusters()[1].getIndices()))

    # Try it on a file
    index1 = [2, 3, 5, 9, 11, 15, 16, 18, 19, 20, 22, 23, 29, 30, 32, 33, 37, 40, 41, 42,
              44, 45, 50, 60, 61, 62, 64, 69, 71, 73, 75, 76, 78, 80, 85, 88, 90, 94, 97]
    index2 = [0, 34, 8, 43, 66, 46, 77, 84, 54]
    index3 = [1, 4, 6, 7, 10, 12, 13, 14, 17, 21, 24, 25, 26, 27, 28, 31, 35, 36, 38, 39,
              47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 63, 65, 67, 68, 70, 72, 74, 79,
              81, 82, 83, 86, 87, 89, 91, 92, 93, 95, 96, 98, 99]

    file = os.path.join(os.path.split(__file__)[0],TEST_FILE)
    data = a6dataset.Dataset(4, tools.data_for_file(file))
    km3  = a6algorithm.Algorithm(data, 3, [23, 54, 36])
    km3._partition()
    introcs.assert_equals(set(index1), set(km3.getClusters()[0].getIndices()))
    introcs.assert_equals(set(index2), set(km3.getClusters()[1].getIndices()))
    introcs.assert_equals(set(index3), set(km3.getClusters()[2].getIndices()))

    print('    Method Algorithm._partition() looks okay')
    print('  Part B of class Algorithm appears correct')
    print()


def test_algorithm_c():
    """
    Tests Part C of the Algorithm class.
    """
    print('  Testing Part C of class Algorithm')
    items = [[0.,0.], [10.,1.], [10.,10.], [0.,9.]]
    dset = a6dataset.Dataset(2, items)
    km1 = a6algorithm.Algorithm(dset, 2, [0,2])
    km1._partition()

    # Test update()
    stable = km1._update()
    assert_points_equal([0,4.5], km1.getClusters()[0].getCentroid())
    assert_points_equal([10.0,5.5], km1.getClusters()[1].getCentroid())
    introcs.assert_false(stable)

    # updating again should not change anything, but should return stable
    stable = km1._update()
    assert_points_equal([0,4.5], km1.getClusters()[0].getCentroid())
    assert_points_equal([10.0,5.5], km1.getClusters()[1].getCentroid())
    introcs.assert_true(stable)

    print('    Method Algorithm._update() looks okay')

    # Now test the k-means process itself.

    # FOR ALL TEST CASES
    # Create and initialize a non-empty dataset
    items = [[0.5,0.5,0.5],[0.5,0.6,0.6],[0.6,0.5,0.6],[0.5,0.6,0.5],[0.5,0.4,0.5],[0.5,0.4,0.4]]
    dset = a6dataset.Dataset(3,items)

    # Create a clustering, providing non-random seed indices so the test is deterministic
    km2 = a6algorithm.Algorithm(dset, 2, [1, 3])

    # PRE-TEST: Check first cluster (should be okay if passed part D)
    cluster1 = km2.getClusters()[0]
    assert_points_equal([0.5, 0.6, 0.6], cluster1.getCentroid())
    introcs.assert_equals(set([]), set(cluster1.getIndices()))

    # PRE-TEST: Check second cluster (should be okay if passed part D)
    cluster2 = km2.getClusters()[1]
    assert_points_equal([0.5, 0.6, 0.5], cluster2.getCentroid())
    introcs.assert_equals(set([]), set(cluster2.getIndices()))

    # Make a fake cluster to test update_centroid() method
    clustertest = a6cluster.Cluster(dset, [0.5, 0.6, 0.6])
    for ind in [1, 2]:
        clustertest.addIndex(ind)

    # TEST CASE 1 (update)
    stable = clustertest.update()
    assert_points_equal([0.55, 0.55, 0.6],clustertest.getCentroid())
    introcs.assert_false(stable) # Not yet stable

    # TEST CASE 2 (update)
    stable = clustertest.update()
    assert_points_equal([0.55, 0.55, 0.6],clustertest.getCentroid())
    introcs.assert_true(stable) # Now it is stable

    # TEST CASE 3 (step)
    km2.step()

    # Check first cluster (WHICH HAS CHANGED!)
    cluster1 = km2.getClusters()[0]
    assert_points_equal([0.55, 0.55, 0.6], cluster1.getCentroid())
    introcs.assert_equals(set([1, 2]), set(cluster1.getIndices()))

    # Check second cluster (WHICH HAS CHANGED!)
    cluster2 = km2.getClusters()[1]
    assert_points_equal([0.5, 0.475, 0.475],cluster2.getCentroid())
    introcs.assert_equals(set([0, 3, 4, 5]), set(cluster2.getIndices()))

    # TEST CASE 3 (step)
    km2.step()

    # Check first cluster (WHICH HAS CHANGED!)
    cluster1 = km2.getClusters()[0]
    assert_points_equal([8./15, 17./30, 17./30], cluster1.getCentroid())
    introcs.assert_equals(set([1, 2, 3]), set(cluster1.getIndices()))

    # Check second cluster (WHICH HAS CHANGED!)
    cluster2 = km2.getClusters()[1]
    assert_points_equal([0.5, 13./30, 14./30],cluster2.getCentroid())
    introcs.assert_equals(set([0, 4, 5]), set(cluster2.getIndices()))

    # Try it on a file
    file = os.path.join(os.path.split(__file__)[0],TEST_FILE)
    data = a6dataset.Dataset(4, tools.data_for_file(file))
    km3  = a6algorithm.Algorithm(data, 3, [23, 54, 36])
    km3.step()

    # The actual results
    cluster0 = km3.getClusters()[0]
    cluster1 = km3.getClusters()[1]
    cluster2 = km3.getClusters()[2]

    # The "correct" answers
    contents0 = [[0.88, 0.84, 0.8, 0.3], [0.02, 0.67, 0.75, 0.61], [0.2, 0.54, 0.73, 0.85],
                 [0.62, 0.75, 0.65, 0.43], [0.35, 0.63, 0.65, 0.12], [0.61, 0.85, 0.81, 0.44],
                 [0.95, 0.94, 0.98, 0.69], [0.04, 0.69, 0.38, 0.39], [0.04, 0.52, 0.99, 0.75],
                 [0.28, 0.91, 0.63, 0.08], [0.14, 0.55, 0.67, 0.63], [0.38, 0.94, 0.53, 0.07],
                 [0.08, 0.62, 0.32, 0.27], [0.69, 0.82, 0.75, 0.65], [0.84, 0.89, 0.91, 0.38],
                 [0.22, 0.88, 0.39, 0.33], [0.39, 0.38, 0.85, 0.32], [0.26, 0.39, 0.95, 0.63],
                 [0.15, 0.87, 0.62, 0.22], [0.65, 0.81, 0.69, 0.55], [0.27, 0.63, 0.69, 0.39],
                 [0.35, 0.7, 0.41, 0.15], [0.2, 0.48, 0.98, 0.84], [0.76, 0.86, 0.74, 0.61],
                 [0.27, 0.65, 0.52, 0.28], [0.86, 0.91, 0.88, 0.62], [0.1, 0.79, 0.5, 0.12],
                 [0.09, 0.85, 0.55, 0.21], [0.79, 0.94, 0.83, 0.48], [0.73, 0.92, 0.74, 0.39],
                 [0.31, 0.5, 0.87, 0.85], [0.39, 0.9, 0.52, 0.26], [0.46, 0.35, 0.96, 0.05],
                 [0.21, 0.62, 0.33, 0.09], [0.58, 0.37, 0.9, 0.08], [0.54, 0.92, 0.36, 0.35],
                 [0.36, 0.64, 0.57, 0.26], [0.09, 0.47, 0.63, 0.8], [0.4, 0.69, 0.74, 0.7]]
    contents1 = [[0.32, 0.87, 0.14, 0.68], [0.87, 0.99, 0.2, 0.8], [0.86, 0.86, 0.32, 0.88],
                 [0.81, 0.66, 0.26, 0.82], [0.91, 0.98, 0.61, 0.58], [0.84, 0.88, 0.04, 0.86],
                 [0.8, 0.62, 0.09, 0.65], [0.72, 0.88, 0.02, 0.95], [0.88, 0.96, 0.09, 0.88]]
    contents2 = [[0.4, 0.21, 0.78, 0.68], [0.54, 0.06, 0.81, 0.98], [0.73, 0.31, 0.15, 0.08],
                 [0.81, 0.69, 0.65, 0.65], [0.14, 0.31, 0.86, 0.74], [0.77, 0.45, 0.31, 0.31],
                 [0.39, 0.14, 0.99, 0.24], [0.23, 0.32, 0.7, 0.75], [0.65, 0.05, 0.39, 0.49],
                 [0.96, 0.09, 0.49, 0.3], [0.86, 0.03, 0.3, 0.39], [0.5, 0.2, 0.69, 0.95],
                 [0.79, 0.09, 0.41, 0.69], [0.4, 0.3, 0.78, 0.74], [0.65, 0.24, 0.63, 0.27],
                 [0.35, 0.3, 0.94, 0.92], [0.71, 0.78, 0.64, 0.57], [0.8, 0.4, 0.23, 0.33],
                 [0.38, 0.07, 0.82, 0.01], [0.66, 0.09, 0.69, 0.46], [0.54, 0.06, 0.74, 0.86],
                 [0.95, 0.62, 0.28, 0.01], [0.35, 0.71, 0.01, 0.32], [0.62, 0.24, 0.77, 0.17],
                 [0.73, 0.65, 0.23, 0.02], [0.27, 0.38, 0.76, 0.63], [0.9, 0.63, 0.83, 0.6],
                 [0.7, 0.04, 0.7, 0.82], [0.95, 0.83, 0.64, 0.5], [0.41, 0.11, 0.61, 0.78],
                 [0.22, 0.44, 0.67, 0.99], [0.51, 0.05, 0.95, 0.66], [0.99, 0.68, 0.8, 0.42],
                 [0.72, 0.55, 0.1, 0.17], [0.44, 0.1, 0.61, 0.98], [0.31, 0.16, 0.95, 0.9],
                 [0.61, 0.42, 0.24, 0.33], [0.89, 0.72, 0.78, 0.38], [0.5, 0.09, 0.84, 0.78],
                 [0.62, 0.01, 0.88, 0.1], [0.44, 0.28, 0.88, 0.99], [0.57, 0.23, 0.6, 0.85],
                 [0.9, 0.05, 0.34, 0.41], [0.9, 0.41, 0.27, 0.36], [0.67, 0.32, 0.66, 0.2],
                 [0.72, 0.14, 0.63, 0.37], [0.39, 0.08, 0.77, 0.96], [0.9, 0.7, 0.74, 0.63],
                 [0.63, 0.05, 0.52, 0.63], [0.62, 0.27, 0.67, 0.77], [0.35, 0.04, 0.85, 0.86],
                 [0.36, 0.34, 0.75, 0.37]]
    centroid0 = [0.3987179487179487, 0.7097435897435899, 0.6864102564102561, 0.4164102564102565]
    centroid1 = [0.7788888888888889, 0.8555555555555555, 0.19666666666666668, 0.788888888888889]
    centroid2 = [0.6038461538461538, 0.29865384615384616, 0.6217307692307692, 0.5455769230769231]

    assert_points_equal(centroid0,cluster0.getCentroid())
    assert_points_equal(centroid1,cluster1.getCentroid())
    assert_points_equal(centroid2,cluster2.getCentroid())
    assert_point_sets_equal(contents0,cluster0.getContents())
    assert_point_sets_equal(contents1,cluster1.getContents())
    assert_point_sets_equal(contents2,cluster2.getContents())


    print('    Method Algorithm.step looks okay')
    print('  Part C of class Algorithm appears correct')
    print()


def test_algorithm_d():
    """
    Tests Part D of the Algorithm class.
    """
    print('  Testing Part D of class Algorithm')
    items = [[0.5,0.5,0.5],[0.5,0.6,0.6],[0.6,0.5,0.6],[0.5,0.6,0.5],[0.5,0.4,0.5],[0.5,0.4,0.4]]
    dset = a6dataset.Dataset(3,items)

    # Try the same test case straight from the top using perform_k_means
    km1 = a6algorithm.Algorithm(dset, 2, [1, 3])
    km1.run(10)

    # Check first cluster
    cluster1 = km1.getClusters()[0]
    assert_points_equal([8./15, 17./30, 17./30], cluster1.getCentroid())
    introcs.assert_equals(set([1, 2, 3]), set(cluster1.getIndices()))

    # Check second cluster
    cluster2 = km1.getClusters()[1]
    assert_points_equal([0.5, 13./30, 14./30],cluster2.getCentroid())
    introcs.assert_equals(set([0, 4, 5]), set(cluster2.getIndices()))
    print('    Method run looks okay')

    # Test on a real world data set
    file = os.path.join(os.path.split(__file__)[0],TEST_FILE)
    data = a6dataset.Dataset(4, tools.data_for_file(file))
    km2  = a6algorithm.Algorithm(data, 3, [23, 54, 36])
    km2.run(20)

    # The actual results
    cluster0 = km2.getClusters()[0]
    cluster1 = km2.getClusters()[1]
    cluster2 = km2.getClusters()[2]

    # The "correct" answers
    contents0 = [[0.88, 0.84, 0.8, 0.3], [0.02, 0.67, 0.75, 0.61], [0.81, 0.69, 0.65, 0.65],
                 [0.62, 0.75, 0.65, 0.43], [0.35, 0.63, 0.65, 0.12], [0.61, 0.85, 0.81, 0.44],
                 [0.95, 0.94, 0.98, 0.69], [0.04, 0.69, 0.38, 0.39], [0.28, 0.91, 0.63, 0.08],
                 [0.38, 0.94, 0.53, 0.07], [0.08, 0.62, 0.32, 0.27], [0.69, 0.82, 0.75, 0.65],
                 [0.84, 0.89, 0.91, 0.38], [0.22, 0.88, 0.39, 0.33], [0.71, 0.78, 0.64, 0.57],
                 [0.15, 0.87, 0.62, 0.22], [0.65, 0.81, 0.69, 0.55], [0.27, 0.63, 0.69, 0.39],
                 [0.35, 0.7, 0.41, 0.15], [0.91, 0.98, 0.61, 0.58], [0.9, 0.63, 0.83, 0.6],
                 [0.95, 0.83, 0.64, 0.5], [0.76, 0.86, 0.74, 0.61], [0.27, 0.65, 0.52, 0.28],
                 [0.86, 0.91, 0.88, 0.62], [0.1, 0.79, 0.5, 0.12], [0.99, 0.68, 0.8, 0.42],
                 [0.09, 0.85, 0.55, 0.21], [0.79, 0.94, 0.83, 0.48], [0.73, 0.92, 0.74, 0.39],
                 [0.89, 0.72, 0.78, 0.38], [0.39, 0.9, 0.52, 0.26], [0.46, 0.35, 0.96, 0.05],
                 [0.21, 0.62, 0.33, 0.09], [0.58, 0.37, 0.9, 0.08], [0.54, 0.92, 0.36, 0.35],
                 [0.67, 0.32, 0.66, 0.2], [0.36, 0.64, 0.57, 0.26], [0.9, 0.7, 0.74, 0.63],
                 [0.4, 0.69, 0.74, 0.7]]
    contents1 = [[0.32, 0.87, 0.14, 0.68], [0.73, 0.31, 0.15, 0.08], [0.87, 0.99, 0.2, 0.8],
                 [0.77, 0.45, 0.31, 0.31], [0.96, 0.09, 0.49, 0.3], [0.86, 0.03, 0.3, 0.39],
                 [0.86, 0.86, 0.32, 0.88], [0.8, 0.4, 0.23, 0.33], [0.81, 0.66, 0.26, 0.82],
                 [0.95, 0.62, 0.28, 0.01], [0.35, 0.71, 0.01, 0.32], [0.73, 0.65, 0.23, 0.02],
                 [0.84, 0.88, 0.04, 0.86], [0.8, 0.62, 0.09, 0.65], [0.72, 0.55, 0.1, 0.17],
                 [0.61, 0.42, 0.24, 0.33], [0.72, 0.88, 0.02, 0.95], [0.88, 0.96, 0.09, 0.88],
                 [0.9, 0.05, 0.34, 0.41], [0.9, 0.41, 0.27, 0.36]]
    contents2 = [[0.4, 0.21, 0.78, 0.68], [0.54, 0.06, 0.81, 0.98], [0.2, 0.54, 0.73, 0.85],
                 [0.14, 0.31, 0.86, 0.74], [0.39, 0.14, 0.99, 0.24], [0.23, 0.32, 0.7, 0.75],
                 [0.65, 0.05, 0.39, 0.49], [0.04, 0.52, 0.99, 0.75], [0.14, 0.55, 0.67, 0.63],
                 [0.5, 0.2, 0.69, 0.95], [0.79, 0.09, 0.41, 0.69], [0.4, 0.3, 0.78, 0.74],
                 [0.65, 0.24, 0.63, 0.27], [0.35, 0.3, 0.94, 0.92], [0.39, 0.38, 0.85, 0.32],
                 [0.38, 0.07, 0.82, 0.01], [0.66, 0.09, 0.69, 0.46], [0.26, 0.39, 0.95, 0.63],
                 [0.54, 0.06, 0.74, 0.86], [0.2, 0.48, 0.98, 0.84], [0.62, 0.24, 0.77, 0.17],
                 [0.27, 0.38, 0.76, 0.63], [0.7, 0.04, 0.7, 0.82], [0.41, 0.11, 0.61, 0.78],
                 [0.22, 0.44, 0.67, 0.99], [0.51, 0.05, 0.95, 0.66], [0.44, 0.1, 0.61, 0.98],
                 [0.31, 0.16, 0.95, 0.9], [0.31, 0.5, 0.87, 0.85], [0.5, 0.09, 0.84, 0.78],
                 [0.62, 0.01, 0.88, 0.1], [0.44, 0.28, 0.88, 0.99], [0.57, 0.23, 0.6, 0.85],
                 [0.72, 0.14, 0.63, 0.37], [0.39, 0.08, 0.77, 0.96], [0.09, 0.47, 0.63, 0.8],
                 [0.63, 0.05, 0.52, 0.63], [0.62, 0.27, 0.67, 0.77], [0.35, 0.04, 0.85, 0.86],
                 [0.36, 0.34, 0.75, 0.37]]
    centroid0 = [0.54125, 0.7545, 0.66125, 0.3775]
    centroid1 = [0.76900, 0.5705, 0.20550, 0.4775]
    centroid2 = [0.42325, 0.2330, 0.75775, 0.6765]

    introcs.assert_float_lists_equal(centroid0,cluster0.getCentroid())
    introcs.assert_float_lists_equal(centroid1,cluster1.getCentroid())
    introcs.assert_float_lists_equal(centroid2,cluster2.getCentroid())
    assert_point_sets_equal(contents0,cluster0.getContents())
    assert_point_sets_equal(contents1,cluster1.getContents())
    assert_point_sets_equal(contents2,cluster2.getContents())
    print('    File analysis test looks okay')
    print('  Part D of class ClusterGroup appears correct')
    print()


def test_all():
    """
    Invokes all tests
    """
    print('Starting unit test\n')
    test_point()
    test_point_list()
    test_dataset_a()
    test_dataset_b()
    test_cluster_a()
    test_cluster_b()
    test_valid_seeds()
    test_algorithm_a()
    test_algorithm_b()
    test_algorithm_c()
    test_algorithm_d()
    print('All test cases passed!')


# COMPARISON FUNCTIONS (DO NOT MODIFY)
def assert_points_equal(expected,received):
    """
    Quits if the lists of points ``expected`` and ``received`` differ

    This function takes two points (list of int/float) and compares them using
    functions from the numerical  package ``numpy``. This is a scientific
    computing package that allows us to test if numbers are "close enough".

    :param expected: The value you expect the test to have
    :type expected:  ``list``

    :param received: The value the test actually had
    :type received:  ``list``
    """
    import numpy
    if not type(expected) == list:
        msg = ('assert_points_equal: first argument %s is not a list' % repr(expected))
        introcs.quit_with_error(msg)
    elif not type(received)  == list:
        msg = ('assert_points_equal: second argument %s is not a list' % repr(received))
    elif sum(map(lambda x : 0 if type(x) in [int,float] else 1,expected)) > 0 :
        msg = ( 'assert_points_equal: first argument %s has non-numeric values' % repr(expected))
        introcs.quit_with_error(msg)
    elif sum(map(lambda x : 0 if type(x) in [int,float] else 1,received)) > 0:
        msg = ( 'assert_points_equal: second argument %s has non-numeric values' % repr(received))
        introcs.quit_with_error(msg)
    elif len(expected) != len(received):
        msg = ( 'assert_points_equal: sequences %s and %s have different sizes' %
               (repr(expected),repr(received)))
        introcs.quit_with_error(msg)

    test = True
    try:
        test = numpy.allclose(expected,received)
    except Exception as e:
        msg = 'assert_points_equal: sequences %s and %s are not comparable' % (repr(expected),repr(received))
        introcs.quit_with_error(msg)
    if (not test):
        msg = 'assert_points_equal: expected %s but instead got %s' % (repr(expected),repr(received))
        introcs.quit_with_error(msg)


def assert_points_not_equal(expected,received):
    """
    Quits if the lists of points ``expected`` and ``received`` are the same

    This function takes two points (tuples of int/float) and compares them using
    functions from the numerical  package ``numpy``. This is a scientific
    computing package that allows us to test if numbers are "close enough".

    :param expected: The value you expect the test to have
    :type expected:  ``list``

    :param received: The value the test actually had
    :type received:  ``list``
    """
    import numpy
    if not type(expected) == list:
        msg = ('assert_points_equal: first argument %s is not a list' % repr(expected))
        introcs.quit_with_error(msg)
    elif not type(received) == list:
        msg = ('assert_points_equal: second argument %s is not a list' % repr(received))
    elif sum(map(lambda x : 0 if type(x) in [int,float] else 1,expected)) > 0 :
        msg = ( 'assert_points_equal: first argument %s has non-numeric values' % repr(expected))
        introcs.quit_with_error(msg)
    elif sum(map(lambda x : 0 if type(x) in [int,float] else 1,received)) > 0:
        msg = ( 'assert_points_equal: second argument %s has non-numeric values' % repr(received))
        introcs.quit_with_error(msg)

    test = True
    try:
        test = numpy.allclose(expected,received)
    except Exception as e:
        msg = 'assert_points_equal: sequences %s and %s are not comparable' % (repr(expected),repr(received))
        introcs.quit_with_error(msg)
    if (test):
        msg = 'assert_points_equal: values %s and %s are the same' % (repr(expected),repr(received))
        introcs.quit_with_error(msg)


def assert_point_sets_equal(expected,received):
    """
    Quits if the lists of points ``expected`` and ``received`` differ

    This function takes two lists of points and compares them using functions
    from the numerical  package ``numpy``. This is a scientific computing
    package that allows us to test if numbers are "close enough".

    The order of the two lists is ignored.  They are the same so the lists
    contain the same points.

    :param expected: The value you expect the test to have
    :type expected:  ``list``

    :param received: The value the test actually had
    :type received:  ``list``
    """
    import numpy
    if not type(expected) == list:
        msg = ('assert_point_sets_equal: first argument %s is not a list' % repr(expected))
        introcs.quit_with_error(msg)
    elif not type(received) == list:
        msg = ('assert_point_sets_equal: second argument %s is not a list' % repr(received))
        introcs.quit_with_error(msg)
    elif sum(map(lambda x : 0 if type(x) in [list,tuple] else 1,expected)) > 0:
        msg = ( 'assert_point_sets_equal: first argument %s is not 2-dimensional' % repr(expected))
        introcs.quit_with_error(msg)
    elif sum(map(lambda x : 0 if type(x) == list else 1,received)) > 0:
        msg = ( 'assert_point_sets_equal: second argument %s is not 2-dimensional' % repr(received))
        introcs.quit_with_error(msg)
    elif any([sum(map(lambda x : 0 if type(x) in [int,float] else 1,item)) > 0 for item in expected]) :
        msg = ( 'assert_point_sets_equal: first argument %s has non-numeric values' % repr(expected))
        introcs.quit_with_error(msg)
    elif any([sum(map(lambda x : 0 if type(x) in [int,float] else 1,item)) > 0 for item in received]) :
        msg = ( 'assert_point_sets_equal: second argument %s has non-numeric values' % repr(received))
        introcs.quit_with_error(msg)
    elif len(expected) != len(received):
        msg = ( 'assert_point_sets_equal: sequences %s and %s have different sizes' %
               (repr(expected),repr(received)))
        introcs.quit_with_error(msg)

    test = True

    # Let's compute the symmetric difference
    try:
        diff = []
        for item1 in expected:
            found = False
            for item2 in received:
                found = found or numpy.allclose(item1,item2)
            if not found:
                diff.append(item1)

        for item1 in received:
            found = False
            for item2 in expected:
                found = found or numpy.allclose(item1,item2)
            if not found:
                diff.append(item1)

        if len(diff) > 0:
            msg = 'assert_point_sets_equal: the points %s are not in both sets' % repr(diff)
            introcs.quit_with_error(msg)
    except Exception as e:
        msg = 'assert_point_sets_equal: sequences %s and %s are not comparable' % (repr(expected),repr(received))
        introcs.quit_with_error(msg)
