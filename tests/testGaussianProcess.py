#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see  < http://www.lsstcorp.org/LegalNotices/ > .
#

from __future__ import absolute_import, division, print_function
import os
import unittest

from builtins import range
import numpy as np

import lsst.utils.tests
import lsst.afw.math as gp
import lsst.pex.exceptions as pex

testPath = os.path.abspath(os.path.dirname(__file__))


class KdTreeTestCase_GaussianProcess(lsst.utils.tests.TestCase):

    def testKdTreeGetData(self):
        """
        Test that KdTree.getData() throws exceptions when it should and returns
        the correct data when it should.
        """
        rng = np.random.RandomState(112)
        data = rng.random_sample((10,10))
        kd = gp.KdTreeD()
        kd.Initialize(data)

        # test that an exception is thrown if you ask for a point with
        # a negative index
        with self.assertRaises(RuntimeError) as context:
            kd.getData(-1, 0)

        # test that an exception is thrown if you ask for a point beyond
        # the number of points stored in the tree
        with self.assertRaises(RuntimeError) as context:
            kd.getData(10, 0)

        # test that an exception is thrown if you ask for dimensions that
        # don't exist
        with self.assertRaises(RuntimeError) as context:
            kd.getData(0, -1)

        with self.assertRaises(RuntimeError) as context:
            kd.getData(0, 10)

        # test that the correct values are returned when they should be
        for ix in range(10):
            for iy in range(10):
                self.assertAlmostEqual(data[ix][iy], kd.getData(ix, iy), places=10)

    def testKdTreeGetDataVec(self):
        """
        Test that KdTree.getData(int) throws exceptions when it should and returns
        the correct data when it should.
        """
        rng = np.random.RandomState(112)
        data = rng.random_sample((10,10))
        kd = gp.KdTreeD()
        kd.Initialize(data)

        # test that an exception is thrown if you ask for a point with
        # a negative index
        with self.assertRaises(RuntimeError) as context:
            kd.getData(-1)

        # test that an exception is thrown if you ask for a point beyond
        # the number of points stored in the tree
        with self.assertRaises(RuntimeError) as context:
            kd.getData(10)

        # test that the correct values are returned when they should be
        for ix in range(10):
            vv = kd.getData(ix)
            for iy in range(10):
                self.assertAlmostEqual(data[ix][iy], vv[iy], places=10)

    def testKdTreeNeighborExceptions(self):
        """
        This test will test that KdTree throws exceptions when you ask it for
        nonsensical number of nearest neighbors.
        """
        rng = np.random.RandomState(112)
        data = rng.random_sample((10,10))
        kd = gp.KdTreeD()
        kd.Initialize(data)
        pt = rng.random_sample(10).astype(float)
        neighdex = np.zeros((5), dtype=np.int32)
        distances = np.zeros((5), dtype=float)

        # ask for a negative number of neighbors
        with self.assertRaises(RuntimeError) as context:
            kd.findNeighbors(neighdex, distances, pt, -2)

        # ask for zero neighbors
        with self.assertRaises(RuntimeError) as context:
            kd.findNeighbors(neighdex, distances, pt, 0)

        # ask for more neighbors than you have data
        with self.assertRaises(RuntimeError) as context:
            kd.findNeighbors(neighdex, distances, pt, 11)

        # try sending neighdex of wrong size
        neighdex_bad = np.zeros((1), dtype=np.int32)
        with self.assertRaises(RuntimeError) as context:
            kd.findNeighbors(neighdex_bad, distances, pt, 5)

        # try sending distances array of wrong size
        distances_bad = np.zeros((1), dtype=float)
        with self.assertRaises(RuntimeError) as context:
            kd.findNeighbors(neighdex, distances_bad, pt, 5)

        # run something that works
        kd.findNeighbors(neighdex, distances, pt, 5)

    def testKdTree(self):
        """
        This test will test the construction of KdTree in the pathological case
        where many of the input data points are identical.
        """
        pp = 100
        dd = 5
        data = np.zeros((pp, dd), dtype=float)
        tol = 1.0e-10

        f = open(os.path.join(testPath, "data", "kd_test_data.sav"))
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(dd):
                data[i][j] = float(s[j])

        kd = gp.KdTreeD()
        try:
            kd.Initialize(data)
        except pex.Exception as e:
            print(e.what())

        kds = gp.KdTreeD()
        try:
            kds.Initialize(data)
        except pex.Exception as e:
            print(e.what())

        try:
            kds.removePoint(2)
        except pex.Exception as e:
            print(e.what())

        worstErr = -1.0
        for i in range(100):
            if i > 2:
                dd = 0.0
                for j in range(5):
                    dd = dd+(kd.getData(i, j)-kds.getData(i-1, j))*(kd.getData(i, j)-kds.getData(i-1, j))
                if dd > worstErr:
                    worstErr = dd
        self.assertLess(worstErr, tol)

        try:
            kd.removePoint(2)
        except pex.Exception as e:
            print(e.what())

        try:
            kds.removePoint(10)
        except pex.Exception as e:
            print(e.what())

        for i in range(99):
            if i > 10:
                dd = 0.0
                for j in range(5):
                    dd = dd+(kd.getData(i, j)-kds.getData(i-1, j))*(kd.getData(i, j)-kds.getData(i-1, j))
                if dd > worstErr:
                    worstErr = dd
        self.assertLess(worstErr, tol)

        try:
            kd.removePoint(10)
        except pex.Exception as e:
            print(e.what())

        try:
            kds.removePoint(21)
        except pex.Exception as e:
            print(e.what())

        for i in range(98):
            if i > 21:
                dd = 0.0
                for j in range(5):
                    dd = dd+(kd.getData(i, j)-kds.getData(i-1, j))*(kd.getData(i, j)-kds.getData(i-1, j))
                if dd > worstErr:
                    worstErr = dd
        self.assertLess(worstErr, tol)
        print("\nworst distance error in kdTest ", worstErr, "\n")

    def testKdTreeNeighbors(self):
        """
        Test that KdTree.findNeighbors() does find the nearest neighbors
        """
        rng = np.random.RandomState(112)
        data = rng.random_sample((10,10))
        kd = gp.KdTreeD()
        kd.Initialize(data)
        pt = rng.random_sample(10).astype(float)
        neighdex = np.zeros((5), dtype=np.int32)
        distances = np.zeros((5), dtype=float)
        kd.findNeighbors(neighdex, distances, pt,5)

        # check that the distances to the nearest neighbors are
        # correctly reported
        for ix in range(len(neighdex)):
            dd_true = np.sqrt(np.power(pt - data[neighdex[ix]],2).sum())
            self.assertAlmostEqual(dd_true, distances[ix], places=10)

        # check that the distances are returned in ascending order
        for ix in range(len(distances)-1):
            self.assertGreaterEqual(distances[ix+1], distances[ix])

        # check that the actual nearest neighbors were found
        dd_true = np.sqrt(np.power(pt-data,2).sum(axis=1))
        sorted_dexes = np.argsort(dd_true)
        for ix in range(len(neighdex)):
            self.assertEqual(neighdex[ix], sorted_dexes[ix])

    def testKdTreeAddPoint(self):
        """
        Test the behavior of KdTree.addPoint
        """
        data = np.array([[1.0, 0.0, 2.0], [1.1, 2.5, 0.0], [4.5, 6.1, 0.0]])
        kd = gp.KdTreeD()
        kd.Initialize(data)

        # test that, if you try to add an improperly-sized point, an exception is thrown
        with self.assertRaises(RuntimeError):
            kd.addPoint(np.array([1.1]*2))

        with self.assertRaises(RuntimeError):
            kd.addPoint(np.array([1.1]*4))

        # test that adding a correctly sized-point works
        # (i.e. that the new point is added to the tree's data)
        kd.addPoint(np.array([1.1]*3))

        self.assertEqual(kd.getPoints(), 4)
        for ix in range(3):
            for iy in range(3):
                self.assertAlmostEqual(data[ix][iy], kd.getData(ix, iy), places=10)

        for ix in range(3):
            self.assertAlmostEqual(kd.getData(3, ix), 1.1, places=10)

        # check that the new point is accounted for in findNeighbors()
        neighdex = np.ones(2, dtype=np.int32)
        distances = np.ones(2, dtype=float)
        vv = np.array([1.2]*3)
        kd.findNeighbors(neighdex, distances, vv, 2)
        self.assertEqual(neighdex[0], 3)
        self.assertEqual(neighdex[1], 0)

    def testKdTreeRemovePoint(self):
        """
        Test the behavior of KdTree.removePoint()
        """
        data = np.array([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]])
        kd = gp.KdTreeD()
        kd.Initialize(data)
        self.assertEqual(kd.getPoints(), 4)

        # test that an exception is raised if you try to remove a non-existent point
        with self.assertRaises(RuntimeError) as context:
            kd.removePoint(-1)

        with self.assertRaises(RuntimeError) as context:
            kd.removePoint(4)

        # test that things work correctly when you do remove a point
        kd.removePoint(1)
        self.assertEqual(kd.getPoints(), 3)
        for ix in range(3):
            self.assertAlmostEqual(kd.getData(0, ix), 1.5, places=10)
            self.assertAlmostEqual(kd.getData(1, ix), 4.0, places=10)
            self.assertAlmostEqual(kd.getData(2, ix), 3.0, places=10)

        neighdex = np.zeros(2, dtype=np.int32)
        distances = np.zeros(2, dtype=float)
        kd.findNeighbors(neighdex, distances, np.array([2.0, 2.0, 2.0]), 2)
        self.assertEqual(neighdex[0], 0)
        self.assertEqual(neighdex[1], 2)

        # test that an exception is raised when you try to remove the last point
        kd.removePoint(0)
        kd.removePoint(0)
        with self.assertRaises(RuntimeError) as context:
            kd.removePoint(0)

    def testKdTreeGetTreeNode(self):
        """
        Test that KdTree.GetTreeNode raises exceptions if you give it bad inputs
        """
        data = np.array([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [3.0, 3.0, 3.0]])
        kd = gp.KdTreeD()
        kd.Initialize(data)

        vv = np.ones(4, dtype=np.int32)
        vv_small = np.ones(1, dtype=np.int32)
        vv_large = np.ones(5, dtype=np.int32)

        with self.assertRaises(RuntimeError):
            kd.getTreeNode(vv_small, 0)

        with self.assertRaises(RuntimeError):
            kd.getTreeNode(vv_large, 0)

        with self.assertRaises(RuntimeError):
            kd.getTreeNode(vv, -1)

        with self.assertRaises(RuntimeError):
            kd.getTreeNode(vv, 4)

        # make sure that a call with good inputs passes
        kd.getTreeNode(vv, 0)


class GaussianProcessTestCase(lsst.utils.tests.TestCase):

    def testConstructorExceptions(self):
        """
        Test that exceptions are raised when constructor is given
        improper inputs
        """
        rng = np.random.RandomState(22)

        # when you pass a different number of data points and function
        # values in
        data = rng.random_sample((10, 4))
        fn_values = rng.random_sample(11)
        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, fn_values, gp.SquaredExpCovariogramD())

        max_val = np.ones(4)
        min_val = np.ones(4)
        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, min_val, max_val, fn_values,
                                gp.SquaredExpCovariogramD())

        many_fn_values = rng.random_sample((11,3))
        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, many_fn_values, gp.SquaredExpCovariogramD())

        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, min_val, max_val, many_fn_values,
                                gp.SquaredExpCovariogramD())

        fn_values = rng.random_sample(data.shape[0])
        many_fn_values = rng.random_sample((10,3))

        # when you pass in improperly sized min and max val arrays
        bad_max_val = np.ones(3)
        bad_min_val = np.zeros(3)
        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, bad_min_val, max_val,
                                fn_values, gp.SquaredExpCovariogramD())

        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, min_val, bad_max_val,
                                fn_values, gp.SquaredExpCovariogramD())

        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, bad_min_val, max_val,
                                many_fn_values, gp.SquaredExpCovariogramD())

        with self.assertRaises(RuntimeError) as context:
            gp.GaussianProcessD(data, min_val, bad_max_val,
                                many_fn_values, gp.SquaredExpCovariogramD())

        # check that the constructor runs when it should
        gp.GaussianProcessD(data, fn_values, gp.SquaredExpCovariogramD())

        gp.GaussianProcessD(data, min_val, max_val, fn_values,
                            gp.SquaredExpCovariogramD())

        gp.GaussianProcessD(data, many_fn_values, gp.SquaredExpCovariogramD())

        gp.GaussianProcessD(data, min_val, max_val, many_fn_values,
                            gp.SquaredExpCovariogramD())

    def testInterpolateExceptions(self):
        """
        Test that interpolate() raises exceptions when given improper
        arguments
        """
        rng = np.random.RandomState(88)
        data = rng.random_sample((13, 5))
        fn = rng.random_sample(13)
        many_fn = rng.random_sample((13, 3))
        gg = gp.GaussianProcessD(data, fn, gp.SquaredExpCovariogramD())
        gg_many = gp.GaussianProcessD(data, many_fn, gp.SquaredExpCovariogramD())

        var = np.zeros(1)

        many_var = np.zeros(3)
        many_mu = np.zeros(3)

        # test that an exception is raised if you try to interpolate at
        # a point with an incorrect number of dimensions
        bad_pt = rng.random_sample(3)
        with self.assertRaises(RuntimeError):
            gg.interpolate(var, bad_pt, 5)

        with self.assertRaises(RuntimeError):
            gg_many.interpolate(many_mu, many_var, bad_pt, 5)

        good_pt = rng.random_sample(5)

        # test that an exception is raised if you try to ask for an improper
        # number of nearest neighbors when interpolating
        with self.assertRaises(RuntimeError) as context:
            gg.interpolate(var, good_pt, 0)

        with self.assertRaises(RuntimeError) as context:
            gg.interpolate(var, good_pt, -1)

        with self.assertRaises(RuntimeError) as context:
            gg.interpolate(var, good_pt, 14)

        with self.assertRaises(RuntimeError) as context:
            gg_many.interpolate(many_mu, many_var, good_pt, 0)

        with self.assertRaises(RuntimeError) as context:
            gg_many.interpolate(many_mu, many_var, good_pt, -1)

        with self.assertRaises(RuntimeError) as context:
            gg_many.interpolate(many_mu, many_var, good_pt, 14)

        # make sure that a Gaussian Process interpolating many functions
        # does not let you call the interpolate() method that returns
        # a scalar
        with self.assertRaises(RuntimeError) as context:
            gg_many.interpolate(many_var, good_pt, 4)

        # test that the many-function interpolate throws an exception
        # on improperly-sized mu and variance arrays
        bad_var = np.zeros(6)
        bad_mu = np.zeros(6)

        with self.assertRaises(RuntimeError) as context:
            gg_many.interpolate(bad_mu, many_var, good_pt, 5)

        with self.assertRaises(RuntimeError) as context:
            gg_many.interpolate(many_mu, bad_var, good_pt, 5)

        # if you try to pass a variance array with len != 1 to
        # the single value interpolate()
        with self.assertRaises(RuntimeError) as context:
            gg.interpolate(many_var, good_pt, 5)

        gg.interpolate(var, good_pt, 5)
        gg_many.interpolate(many_mu, many_var, good_pt, 5)

    def testSelfInterpolateExceptions(self):
        """
        Test that selfInterpolate raises exceptions on bad arguments.
        """
        rng = np.random.RandomState(632)
        data = rng.random_sample((15, 4))
        fn = rng.random_sample(15)
        gg = gp.GaussianProcessD(data, fn, gp.SquaredExpCovariogramD())
        many_fn = rng.random_sample((15, 3))
        gg_many = gp.GaussianProcessD(data, many_fn, gp.SquaredExpCovariogramD())

        var_good = np.zeros(3)
        mu_good = np.zeros(3)

        var_bad = np.zeros(5)
        mu_bad = np.zeros(5)

        # test that an exception is raised when you try to use scalar
        # selfInterpolation() on a many-function GaussianProcess
        with self.assertRaises(RuntimeError) as context:
            gg_many.selfInterpolate(var_good, 11, 6)

        # test that an exception is raised when you pass a var_array that is
        # too large into a scalar GaussianProcess
        with self.assertRaises(RuntimeError) as context:
            gg.selfInterpolate(var_good, 11, 6)

        # test that an exception is thrown when you pass in improperly-sized
        # mu and/or var arrays
        with self.assertRaises(RuntimeError) as context:
            gg_many.selfInterpolate(mu_good, var_bad, 11, 6)

        with self.assertRaises(RuntimeError) as context:
            gg_many.selfInterpolate(mu_bad, var_good, 11, 6)

        # make surethat selfInterpolate runs when it should
        gg_many.selfInterpolate(mu_good, var_good, 11, 6)
        var_one = np.zeros(1)
        gg.selfInterpolate(var_one, 11, 6)

    def testBatchInterpolateExceptions(self):
        """
        Test that batchInterpolate() throws exceptions on bad input
        """
        rng = np.random.RandomState(88)
        rng = np.random.RandomState(632)
        data = rng.random_sample((15, 4))
        fn = rng.random_sample(15)
        gg = gp.GaussianProcessD(data, fn, gp.SquaredExpCovariogramD())

        pts_good = rng.random_sample((11, 4))
        pts_bad = rng.random_sample((11, 3))
        mu_good = np.zeros(11)
        mu_bad = np.zeros(9)
        var_good = np.zeros(11)
        var_bad = np.zeros(9)

        # test for exception on points of incorrect size
        with self.assertRaises(RuntimeError) as context:
            gg.batchInterpolate(mu_good, var_good, pts_bad)

        with self.assertRaises(RuntimeError) as context:
            gg.batchInterpolate(mu_good, pts_bad)

        # test for exception on output arrays of incorrect size
        with self.assertRaises(RuntimeError) as context:
            gg.batchInterpolate(mu_bad, var_good, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg.batchInterpolate(mu_bad, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg.batchInterpolate(mu_good, var_bad, pts_good)

        # test that it runs properly with good inputs
        gg.batchInterpolate(mu_good, var_good, pts_good)
        gg.batchInterpolate(mu_good, pts_good)

        fn_many = rng.random_sample((15,6))
        gg_many = gp.GaussianProcessD(data, fn_many, gp.SquaredExpCovariogramD())

        # test that a GaussianProcess on many functions raises an exception
        # when you call batchInterpolate with output arrays that only have
        # room for one function
        with self.assertRaises(RuntimeError) as cnotext:
            gg_many.batchInterpolate(mu_good, var_good, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_good, pts_good)

        mu_good = np.zeros((11, 6))
        mu_bad_fn = np.zeros((11, 5))
        mu_bad_pts = np.zeros((10, 6))
        var_good = np.zeros((11, 6))
        var_bad_fn = np.zeros((11, 5))
        var_bad_pts = np.zeros((10, 6))

        # test that a Gaussian Process on many functions raises an exception
        # when you try to interpolate on points of the wrong dimensionality
        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_good, var_good, pts_bad)

        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_good, pts_bad)

        # test that a Gaussian Process on many functions rases an exception
        # when the output arrays are of the wrong size
        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_bad_fn, var_good, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_bad_pts, var_good, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_bad_pts, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_bad_fn, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_good, var_bad_fn, pts_good)

        with self.assertRaises(RuntimeError) as context:
            gg_many.batchInterpolate(mu_good, var_bad_pts, pts_good)

        # check that a Gaussian Process on many functions runs properly
        # when given good inputs
        gg_many.batchInterpolate(mu_good, var_good, pts_good)
        gg_many.batchInterpolate(mu_good, pts_good)

    def testTooManyNeighbors(self):
        """
        Test that GaussianProcess checks if too many neighbours are requested
        """
        nData = 100                        # number of data points
        dimen = 10                         # dimension of each point
        data = np.zeros((nData, dimen))
        fn = np.zeros(nData)
        gg = gp.GaussianProcessD(data, fn, gp.SquaredExpCovariogramD())
        test = np.zeros(dimen)
        sigma = np.empty(1)
        mu_arr = np.empty(1)

        with self.assertRaises(pex.Exception):
             gg.interpolate(sigma, test, 2*nData)
        with self.assertRaises(pex.Exception):
             gg.interpolate(sigma, test, -5)
        with self.assertRaises(pex.Exception):
             gg.selfInterpolate(sigma, 0, 2*nData)
        with self.assertRaises(pex.Exception):
             gg.selfInterpolate(sigma, 0, -5)
        with self.assertRaises(pex.Exception):
             gg.selfInterpolate(sigma, -1, nData-1)
        # the following segfaults, for unknown reasons, so run directly instead
        # self.assertRaises(pex.Exception,gg.selfInterpolate,sigma,nData,nData-1)
        try:
            gg.interpolate(mu_arr, sigma, 2*nData)
            self.fail("gg.interpolate(mu_arr,sigma,2*nData) did not fail")
        except pex.Exception:
            pass
        with self.assertRaises(pex.Exception):
             gg.interpolate(mu_arr, sigma, 2*nData)
        with self.assertRaises(pex.Exception):
             gg.interpolate(mu_arr, sigma, -5)

    def testInterpolate(self):
        """
        This will test GaussianProcess.interpolate using both the squared
        exponential covariogram  and the neural network covariogram on data
        that was generated with known answers.

        The test will check that the code returns the correct values of both
        mu (interpolated function value) and sig2 (the variance)

        This test uses the GaussianProcess constructor that does not normalize
        coordinate values with minima and maxima.
        """

        pp = 2000  # number of data points
        dd = 10  # number of dimensions
        kk = 15  # number of nearest neighbors being used
        tol = 1.0e-3  # the largest relative error that will be tolerated

        data = np.zeros((pp, dd), dtype=float)  # input data points
        fn = np.zeros((pp), dtype=float)  # input function values
        test = np.zeros((dd), dtype=float)  # query points
        sigma = np.zeros((1), dtype=float)  # variance

        xx = gp.SquaredExpCovariogramD()
        xx.setEllSquared(100.0)

        # read in the input data
        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"))
        ff = f.readlines()
        f.close()

        for i in range(len(ff)):
            s = ff[i].split()
            fn[i] = float(s[10])
            for j in range(10):
                data[i][j] = float(s[j])

        # first try the squared exponential covariogram (the default)
        try:
            gg = gp.GaussianProcessD(data, fn, xx)
        except pex.Exception as e:
            print(e.what())

        gg.setLambda(0.001)

        # now, read in the test points and their corresponding known solutions
        f = open(os.path.join(testPath,"data", "gp_exp_covar_solutions.sav"))
        ff = f.readlines()
        f.close()

        worstMuErr = -1.0  # keep track of the worst fractional error in mu
        worstSigErr = -1.0  # keep track of the worst fractional error in the variance

        for z in range(len(ff)):
            s = ff[z].split()  # s will store the zth line of the solution file
            for i in range(dd):
                test[i] = float(s[i])  # read in the test point coordinates

            mushld = float(s[dd + kk])  # read in what mu should be
            sigshld = float(s[dd + kk + 1])  # read in what the variance should be

            mu = gg.interpolate(sigma, test, kk)

            err = (mu - mushld)
            if mushld != 0.0:
                err = err/mushld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstMuErr:
                worstMuErr = err

            err = (sigma[0] - sigshld)
            if sigshld != 0.0:
                err = err/sigshld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstSigErr:
                worstSigErr = err

        print("\nThe errors for squared exponent covariogram\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)

        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

        # now try with the Neural Network covariogram

        kk = 50

        nn = gp.NeuralNetCovariogramD()
        nn.setSigma0(1.23)
        nn.setSigma1(0.452)

        gg.setCovariogram(nn)
        gg.setLambda(0.0045)

        f = open(os.path.join(testPath,"data", "gp_nn_solutions.sav"))
        ff = f.readlines()
        f.close()

        worstMuErr = -1.0
        worstSigErr = -1.0

        for z in range(len(ff)):
            s = ff[z].split()
            for i in range(dd):
                test[i] = float(s[i])

            mushld = float(s[dd + kk])
            sigshld = float(s[dd + kk + 1])

            mu = gg.interpolate(sigma, test, kk)

            err = (mu - mushld)
            if mushld != 0.0:
                err = err/mushld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstMuErr:
                worstMuErr = err

            err = (sigma[0] - sigshld)
            if sigshld != 0.0:
                err = err/sigshld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstSigErr:
                worstSigErr = err

        print("\nThe errors for neural net covariogram\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)

        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

    def testMinMax(self):
        """
        This test will test GaussianProcess.interpolate using the constructor
        that normalizes data point coordinates by minima and maxima.

        It will only use the squared exponential covariogram (since testInterpolate() presumably
        tested the performance of the neural network covariogram; this test is only concerned
        with the alternate constructor)

        This test proceeds largely like testInterpolate above
        """
        pp = 2000
        dd = 10
        kk = 50
        tol = 1.0e-4
        data = np.zeros((pp, dd), dtype=float)
        fn = np.zeros((pp), dtype=float)
        test = np.zeros((dd), dtype=float)
        sigma = np.zeros((1), dtype=float)

        mins = np.zeros((dd), dtype=float)
        maxs = np.zeros((dd), dtype=float)

        nn = gp.NeuralNetCovariogramD()
        nn.setSigma0(0.555)
        nn.setSigma1(0.112)

        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"))
        ff = f.readlines()
        f.close()

        for i in range(len(ff)):
            s = ff[i].split()
            fn[i] = float(s[10])
            for j in range(10):
                data[i][j] = float(s[j])

        for i in range(pp):
            for j in range(dd):
                if (i == 0) or (data[i][j] < mins[j]):
                    mins[j] = data[i][j]
                if (i == 0) or (data[i][j] > maxs[j]):
                    maxs[j] = data[i][j]

        mins[2] = 0.0
        maxs[2] = 10.0
        try:
            gg = gp.GaussianProcessD(data, mins, maxs, fn, nn)
        except pex.Exception as e:
            print(e.what())

        gg.setLambda(0.0045)

        f = open(os.path.join(testPath, "data", "gp_minmax_solutions.sav"))
        ff = f.readlines()
        f.close()

        worstMuErr = -1.0
        worstSigErr = -1.0
        for z in range(len(ff)):
            s = ff[z].split()
            for i in range(dd):
                test[i] = float(s[i])

            mushld = float(s[dd + kk])
            sigshld = float(s[dd + kk + 1])

            mu = gg.interpolate(sigma, test, kk)

            err = (mu - mushld)
            if mushld != 0.0:
                err = err/mushld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstMuErr:
                worstMuErr = err

            err = (sigma[0] - sigshld)
            if sigshld != 0.0:
                err = err/sigshld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstSigErr:
                worstSigErr = err

        print("\nThe errors for Gaussian process using min-max normalization\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)

        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

    def testAddPointExceptions(self):
        """
        Test that addPoint() raises exceptions when it should
        """
        rng = np.random.RandomState(44)
        data = rng.random_sample((10, 3))
        fn = rng.random_sample(10)
        gg = gp.GaussianProcessD(data, fn, gp.SquaredExpCovariogramD())
        fn_many = rng.random_sample((10, 5))
        gg_many = gp.GaussianProcessD(data, fn_many, gp.SquaredExpCovariogramD())

        pt_good = rng.random_sample(3)
        pt_bad = rng.random_sample(6)
        fn_good = rng.random_sample(5)

        # test that, when you add a point of the wrong dimensionality,
        # an exception is raised
        with self.assertRaises(RuntimeError) as context:
            gg.addPoint(pt_bad, 5.0)

        with self.assertRaises(RuntimeError) as context:
            gg.addPoint(pt_bad, fn_good)

        # test that a GaussianProcess on many functions raises an exception
        # when you try to add a point with just one function value
        with self.assertRaises(RuntimeError) as context:
            gg_many.addPoint(pt_good, 5.0)

        # check that, given good inputs, addPoint will run
        gg.addPoint(pt_good, 5.0)
        gg_many.addPoint(pt_good,fn_good)

    def testAddition(self):
        """
        This will test the performance of interpolation after adding new points
        to GaussianProcess' data set
        """
        pp = 1000
        dd = 10
        kk = 15
        tol = 1.0e-4

        data = np.zeros((pp, dd), dtype=float)
        fn = np.zeros((pp), dtype=float)
        test = np.zeros((dd), dtype=float)
        sigma = np.zeros((1), dtype=float)

        xx = gp.SquaredExpCovariogramD()
        xx.setEllSquared(5.0)

        f = open(os.path.join(testPath, "data", "gp_additive_test_root.sav"))
        ff = f.readlines()
        f.close()

        for i in range(len(ff)):
            s = ff[i].split()
            fn[i] = float(s[10])
            for j in range(10):
                data[i][j] = float(s[j])

        # establish the Gaussian Process
        try:
            gg = gp.GaussianProcessD(data, fn, xx)
        except pex.Exception as e:
            print(e.what())

        gg.setLambda(0.002)

        # now add new points to it and see if GaussianProcess.interpolate performs
        # correctly
        f = open(os.path.join(testPath, "data", "gp_additive_test_data.sav"))
        ff = f.readlines()
        f.close()
        for z in range(len(ff)):
            s = ff[z].split()
            for i in range(dd):
                test[i] = float(s[i])
                mushld = float(s[dd])
            try:
                gg.addPoint(test, mushld)
            except pex.Exception as e:
                print(e.what())

        f = open(os.path.join(testPath, "data", "gp_additive_test_solutions.sav"))
        ff = f.readlines()
        f.close()

        worstMuErr = -1.0
        worstSigErr = -1.0

        for z in range(len(ff)):
            s = ff[z].split()
            for i in range(dd):
                test[i] = float(s[i])

            mushld = float(s[dd + kk])
            sigshld = float(s[dd + kk + 1])

            mu = gg.interpolate(sigma, test, kk)

            err = (mu - mushld)
            if mushld != 0:
                err = err/mushld
            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstMuErr:
                worstMuErr = err

            err = (sigma[0] - sigshld)
            if sigshld != 0:
                err = err/sigshld
            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstSigErr:
                worstSigErr = err

        print("\nThe errors for the test of adding points to the Gaussian process\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)

        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

    def testBatch(self):
        """
        This test will test GaussianProcess.batchInterpolate both with
        and without variance calculation
        """
        pp = 100
        dd = 10
        tol = 1.0e-3

        data = np.zeros((pp, dd), dtype=float)
        fn = np.zeros((pp), dtype=float)

        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(100):
            s = ff[i].split()
            for j in range(dd):
                data[i][j] = float(s[j])
            fn[i] = float(s[dd])

        xx = gp.SquaredExpCovariogramD()
        xx.setEllSquared(2.0)

        try:
            gg = gp.GaussianProcessD(data, fn, xx)
        except pex.Exception as e:
            print(e.what())

        gg.setLambda(0.0032)

        f = open(os.path.join(testPath, "data", "gp_batch_solutions.sav"), "r")
        ff = f.readlines()
        f.close()

        ntest = len(ff)
        mushld = np.zeros((ntest), dtype=float)
        varshld = np.zeros((ntest), dtype=float)
        mu = np.zeros((ntest), dtype=float)
        var = np.zeros((ntest), dtype=float)

        queries = np.zeros((ntest, dd), dtype=float)

        for i in range(ntest):
            s = ff[i].split()
            for j in range(dd):
                queries[i][j] = float(s[j])
            mushld[i] = float(s[dd])
            varshld[i] = float(s[dd + 1])

        # test with variance calculation
        gg.batchInterpolate(mu, var, queries)

        worstMuErr = -1.0
        worstVarErr = -1.0
        for i in range(ntest):
            err = mu[i]-mushld[i]
            if mushld[i] != 0.0:
                err = err/mushld[i]
            if err < 0.0:
                err = -1.0 * err
            if err > worstMuErr:
                worstMuErr = err

            err = var[i]-varshld[i]
            if varshld[i] != 0.0:
                err = err/varshld[i]
            if err < 0.0:
                err = -1.0 * err
            if err > worstVarErr:
                worstVarErr = err

        # test without variance interpolation
        # continue keeping track of worstMuErr
        gg.batchInterpolate(mu, queries)
        for i in range(ntest):
            err = mu[i]-mushld[i]
            if mushld[i] != 0.0:
                err = err/mushld[i]
            if err < 0.0:
                err = -1.0 * err
            if err > worstMuErr:
                worstMuErr = err

        self.assertLess(worstMuErr, tol)
        self.assertLess(worstVarErr, tol)

        print("\nThe errors for batch interpolation\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstVarErr)

    def testSelf(self):
        """
        This test will test GaussianProcess.selfInterpolation
        """
        pp = 2000
        dd = 10
        tol = 1.0e-3
        kk = 20

        data = np.zeros((pp, dd), dtype=float)
        fn = np.zeros((pp), dtype=float)

        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(pp):
            s = ff[i].split()
            for j in range(dd):
                data[i][j] = float(s[j])
            fn[i] = float(s[dd])

        xx = gp.SquaredExpCovariogramD()
        xx.setEllSquared(20.0)
        try:
            gg = gp.GaussianProcessD(data, fn, xx)
        except pex.Exception as e:
            print(e.what())

        gg.setKrigingParameter(30.0)
        gg.setLambda(0.00002)

        f = open(os.path.join(testPath, "data", "gp_self_solutions.sav"), "r")
        ff = f.readlines()
        f.close()
        variance = np.zeros((1), dtype=float)

        worstMuErr = -1.0
        worstSigErr = -1.0

        for i in range(pp):
            s = ff[i].split()
            mushld = float(s[0])
            sig2shld = float(s[1])

            try:
                mu = gg.selfInterpolate(variance, i, kk)
            except pex.Exception as e:
                print(e.what())

            err = mu - mushld
            if mushld != 0.0:
                err = err/mushld
            if err < 0.0:
                err = err * (-1.0)
            if i == 0 or err > worstMuErr:
                worstMuErr = err

            err = variance[0] - sig2shld
            if sig2shld != 0.0:
                err = err/sig2shld
            if err < 0.0:
                err = err * (-1.0)
            if i == 0 or err > worstSigErr:
                worstSigErr = err

        print("\nThe errors for self interpolation\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)
        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

    def testVector(self):
        """
        This will test interpolate using a vector of functions
        """

        tol = 1.0e-3
        data = np.zeros((2000, 10), dtype=float)
        fn = np.zeros((2000, 4), dtype=float)
        mu = np.zeros((4), dtype=float)
        sig = np.zeros((4), dtype=float)
        mushld = np.zeros((4), dtype=float)
        vv = np.zeros((10), dtype=float)
        vvf = np.zeros((4), dtype=float)

        kk = 30

        f = open(os.path.join(testPath, "data", "gp_vector_data.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(10):
                data[i][j] = float(s[j])
            for j in range(4):
                fn[i][j] = float(s[j+10])

        nn = gp.NeuralNetCovariogramD()
        nn.setSigma0(2.25)
        nn.setSigma1(0.76)
        try:
            gg = gp.GaussianProcessD(data, fn, nn)
        except pex.Exception as e:
            print(e.what())

        gg.setLambda(0.0045)

        worstMuErr = -1.0
        worstSigErr = -1.0
        f = open(os.path.join(testPath, "data", "gp_vector_solutions.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(10):
                vv[j] = float(s[j])
            for j in range(4):
                mushld[j] = float(s[j+10])
            sigshld = float(s[14])

            gg.interpolate(mu, sig, vv, kk)

            for j in range(4):
                muErr = (mu[j]-mushld[j])/mushld[j]
                sigErr = (sig[j]-sigshld)/sigshld
                if muErr < 0.0:
                    muErr = -1.0 * muErr
                if sigErr < 0.0:
                    sigErr = -1.0 * sigErr
                if (muErr > worstMuErr):
                    worstMuErr = muErr
                if (sigErr > worstSigErr):
                    worstSigErr = sigErr

        print("\nThe errors for vector interpolation\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)
        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

        worstMuErr = -1.0
        worstSigErr = -1.0

        f = open(os.path.join(testPath, "data", "gp_vector_selfinterpolate_solutions.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            try:
                gg.selfInterpolate(mu, sig, i, kk)
            except pex.Exception as e:
                print(e.what())

            for j in range(4):
                mushld[j] = float(s[j])
                sigshld = float(s[4])

            for j in range(4):
                muErr = (mu[j]-mushld[j])/mushld[j]
                if muErr < -1.0:
                    muErr = -1.0 * muErr
                if muErr > worstMuErr:
                    worstMuErr = muErr

                sigErr = (sig[j]-sigshld)/sigshld
                if sigErr < -1.0:
                    sigErr = -1.0*sigErr
                if sigErr > worstSigErr:
                    worstSigErr = sigErr

        print("\nThe errors for vector self interpolation\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)
        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

        queries = np.zeros((100, 10), dtype=float)
        batchMu = np.zeros((100, 4), dtype=float)
        batchSig = np.zeros((100, 4), dtype=float)
        batchMuShld = np.zeros((100, 4), dtype=float)
        batchSigShld = np.zeros((100, 4), dtype=float)
        batchData = np.zeros((200, 10), dtype=float)
        batchFunctions = np.zeros((200, 4), dtype=float)

        f = open(os.path.join(testPath, "data", "gp_vectorbatch_data.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(10):
                batchData[i][j] = float(s[j])
            for j in range(4):
                batchFunctions[i][j] = float(s[j+10])

        try:
            ggbatch = gp.GaussianProcessD(batchData, batchFunctions, nn)
        except pex.Exception as e:
            print(e.what())

        ggbatch.setLambda(0.0045)

        f = open(os.path.join(testPath, "data", "gp_vectorbatch_solutions.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(10):
                queries[i][j] = float(s[j])
            for j in range(4):
                batchMuShld[i][j] = float(s[j+10])
            sigShld = float(s[14])
            for j in range(4):
                batchSigShld[i][j] = sigShld

        ggbatch.batchInterpolate(batchMu, batchSig, queries)
        worstMuErr = -1.0
        worstSigErr = -1.0
        for i in range(100):
            for j in range(4):
                muErr = (batchMu[i][j]-batchMuShld[i][j])/batchMuShld[i][j]
                sigErr = (batchSig[i][j]-batchSigShld[i][j])/batchSigShld[i][j]
                if muErr < 0.0:
                    muErr = muErr * (-1.0)
                if sigErr < 0.0:
                    sigErr = sigErr * (-1.0)

                if muErr > worstMuErr:
                    worstMuErr = muErr
                if sigErr > worstSigErr:
                    worstSigErr = sigErr

        print("\nThe errors for vector batch interpolation with variance\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)
        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

        ggbatch.batchInterpolate(batchMu, queries)
        worstMuErr = -1.0
        worstSigErr = -1.0
        for i in range(100):
            for j in range(4):
                muErr = (batchMu[i][j]-batchMuShld[i][j])/batchMuShld[i][j]
                if muErr < 0.0:
                    muErr = muErr * (-1.0)

                if muErr > worstMuErr:
                    worstMuErr = muErr

        print("\nThe errors for vector batch interpolation without variance\n")
        print("worst mu error ", worstMuErr)
        self.assertLess(worstMuErr, tol)

        f = open(os.path.join(testPath, "data", "gp_vector_add_data.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(10):
                vv[j] = float(s[j])
            for j in range(4):
                vvf[j] = float(s[j+10])
            try:
                gg.addPoint(vv, vvf)
            except pex.Exception as e:
                print(e.what())

        f = open(os.path.join(testPath, "data", "gp_vector_add_solutions.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            try:
                gg.selfInterpolate(mu, sig, i, kk)
            except pex.Exception as e:
                print(e.what())

            for j in range(4):
                mushld[j] = float(s[j])
            sigshld = float(s[4])

            for j in range(4):
                muErr = (mu[j]-mushld[j])/mushld[j]
            if muErr < 0.0:
                muErr = -1.0 * muErr
            if muErr > worstMuErr:
                worstMuErr = muErr

            sigErr = (sig[j]-sigshld)/sigshld
            if sigErr < 0.0:
                sigErr = -1.0*sigErr
            if sigErr > worstSigErr:
                worstSigErr = sigErr

        print("\nThe errors for vector add interpolation\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)

        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

    def testSubtraction(self):
        """
        This will test interpolate after subtracting points
        """

        tol = 1.0e-3
        data = np.zeros((2000, 10), dtype=float)
        fn = np.zeros((2000, 4), dtype=float)
        mu = np.zeros((4), dtype=float)
        sig = np.zeros((4), dtype=float)
        mushld = np.zeros((4), dtype=float)
        vv = np.zeros((10), dtype=float)
        kk = 30

        f = open(os.path.join(testPath, "data", "gp_subtraction_data.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(10):
                data[i][j] = float(s[j])
            for j in range(4):
                fn[i][j] = float(s[j+10])

        xx = gp.SquaredExpCovariogramD()
        xx.setEllSquared(2.3)
        try:
            gg = gp.GaussianProcessD(data, fn, xx)
        except pex.Exception as e:
            print(e.what())

        gg.setLambda(0.002)

        j = 1
        for i in range(1000):
            try:
                gg.removePoint(j)
            except pex.Exception as e:
                print(e.what())

            j = j+1

        worstMuErr = -1.0
        worstSigErr = -1.0
        f = open(os.path.join(testPath, "data", "gp_subtraction_solutions.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(10):
                vv[j] = float(s[j])
            for j in range(4):
                mushld[j] = float(s[j+10])
            sigshld = float(s[14])

            gg.interpolate(mu, sig, vv, kk)

            for j in range(4):
                muErr = (mu[j]-mushld[j])/mushld[j]
                sigErr = (sig[j]-sigshld)/sigshld
                if muErr < 0.0:
                    muErr = -1.0 * muErr
                if sigErr < 0.0:
                    sigErr = -1.0 * sigErr
                if (muErr > worstMuErr):
                    worstMuErr = muErr
                if (sigErr > worstSigErr):
                    worstSigErr = sigErr

        print("\nThe errors for subtraction interpolation\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)
        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)

        worstMuErr = -1.0
        worstSigErr = -1.0

        f = open(os.path.join(testPath, "data", "gp_subtraction_selfinterpolate_solutions.sav"), "r")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            try:
                gg.selfInterpolate(mu, sig, i, kk)
            except pex.Exception as e:
                print(e.what())

            for j in range(4):
                mushld[j] = float(s[j])
                sigshld = float(s[4])

            for j in range(4):
                muErr = (mu[j]-mushld[j])/mushld[j]
                if muErr < 0.0:
                    muErr = -1.0 * muErr
                if muErr > worstMuErr:
                    worstMuErr = muErr

                sigErr = (sig[j]-sigshld)/sigshld
                if sigErr < 0.0:
                    sigErr = -1.0*sigErr
                if sigErr > worstSigErr:
                    worstSigErr = sigErr

        print("\nThe errors for subtraction self interpolation\n")
        print("worst mu error ", worstMuErr)
        print("worst sig2 error ", worstSigErr)
        self.assertLess(worstMuErr, tol)
        self.assertLess(worstSigErr, tol)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
