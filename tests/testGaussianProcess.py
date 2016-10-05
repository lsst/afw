#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see  < http://www.lsstcorp.org/LegalNotices/ > .
#pybind11##
#pybind11#
#pybind11#import os
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.math as gp
#pybind11#import lsst.utils.tests as utilsTests
#pybind11#import lsst.pex.exceptions as pex
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11#
#pybind11#class GaussianProcessTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def testTooManyNeighbors(self):
#pybind11#        """
#pybind11#        Test that GaussianProcess checks if too many neighbours are requested
#pybind11#        """
#pybind11#        nData = 100                        # number of data points
#pybind11#        dimen = 10                         # dimension of each point
#pybind11#        data = np.zeros((nData, dimen))
#pybind11#        fn = np.zeros(nData)
#pybind11#        gg = gp.GaussianProcessD(data, fn, gp.SquaredExpCovariogramD())
#pybind11#        test = np.zeros(dimen)
#pybind11#        sigma = np.empty(1)
#pybind11#        mu_arr = np.empty(1)
#pybind11#
#pybind11#        with self.assertRaises(pex.Exception):
#pybind11#             gg.interpolate(sigma, test, 2*nData)
#pybind11#        with self.assertRaises(pex.Exception):
#pybind11#             gg.interpolate(sigma, test, -5)
#pybind11#        with self.assertRaises(pex.Exception):
#pybind11#             gg.selfInterpolate(sigma, 0, 2*nData)
#pybind11#        with self.assertRaises(pex.Exception):
#pybind11#             gg.selfInterpolate(sigma, 0, -5)
#pybind11#        with self.assertRaises(pex.Exception):
#pybind11#             gg.selfInterpolate(sigma, -1, nData-1)
#pybind11#        # the following segfaults, for unknown reasons, so run directly instead
#pybind11#        # self.assertRaises(pex.Exception,gg.selfInterpolate,sigma,nData,nData-1)
#pybind11#        try:
#pybind11#            gg.interpolate(mu_arr, sigma, 2*nData)
#pybind11#            self.fail("gg.interpolate(mu_arr,sigma,2*nData) did not fail")
#pybind11#        except pex.Exception:
#pybind11#            pass
#pybind11#        with self.assertRaises(pex.Exception):
#pybind11#             gg.interpolate(mu_arr, sigma, 2*nData)
#pybind11#        with self.assertRaises(pex.Exception):
#pybind11#             gg.interpolate(mu_arr, sigma, -5)
#pybind11#
#pybind11#    def testInterpolate(self):
#pybind11#        """
#pybind11#        This will test GaussianProcess.interpolate using both the squared
#pybind11#        exponential covariogram  and the neural network covariogram on data
#pybind11#        that was generated with known answers.
#pybind11#
#pybind11#        The test will check that the code returns the correct values of both
#pybind11#        mu (interpolated function value) and sig2 (the variance)
#pybind11#
#pybind11#        This test uses the GaussianProcess constructor that does not normalize
#pybind11#        coordinate values with minima and maxima.
#pybind11#        """
#pybind11#
#pybind11#        pp = 2000  # number of data points
#pybind11#        dd = 10  # number of dimensions
#pybind11#        kk = 15  # number of nearest neighbors being used
#pybind11#        tol = 1.0e-3  # the largest relative error that will be tolerated
#pybind11#
#pybind11#        data = np.zeros((pp, dd), dtype=float)  # input data points
#pybind11#        fn = np.zeros((pp), dtype=float)  # input function values
#pybind11#        test = np.zeros((dd), dtype=float)  # query points
#pybind11#        sigma = np.zeros((1), dtype=float)  # variance
#pybind11#
#pybind11#        xx = gp.SquaredExpCovariogramD()
#pybind11#        xx.setEllSquared(100.0)
#pybind11#
#pybind11#        # read in the input data
#pybind11#        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            fn[i] = float(s[10])
#pybind11#            for j in range(10):
#pybind11#                data[i][j] = float(s[j])
#pybind11#
#pybind11#        # first try the squared exponential covariogram (the default)
#pybind11#        try:
#pybind11#            gg = gp.GaussianProcessD(data, fn, xx)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        gg.setLambda(0.001)
#pybind11#
#pybind11#        # now, read in the test points and their corresponding known solutions
#pybind11#        f = open(os.path.join(testPath,"data", "gp_exp_covar_solutions.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        worstMuErr = -1.0  # keep track of the worst fractional error in mu
#pybind11#        worstSigErr = -1.0  # keep track of the worst fractional error in the variance
#pybind11#
#pybind11#        for z in range(len(ff)):
#pybind11#            s = ff[z].split()  # s will store the zth line of the solution file
#pybind11#            for i in range(dd):
#pybind11#                test[i] = float(s[i])  # read in the test point coordinates
#pybind11#
#pybind11#            mushld = float(s[dd + kk])  # read in what mu should be
#pybind11#            sigshld = float(s[dd + kk + 1])  # read in what the variance should be
#pybind11#
#pybind11#            mu = gg.interpolate(sigma, test, kk)
#pybind11#
#pybind11#            err = (mu - mushld)
#pybind11#            if mushld != 0.0:
#pybind11#                err = err/mushld
#pybind11#
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstMuErr:
#pybind11#                worstMuErr = err
#pybind11#
#pybind11#            err = (sigma[0] - sigshld)
#pybind11#            if sigshld != 0.0:
#pybind11#                err = err/sigshld
#pybind11#
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstSigErr:
#pybind11#                worstSigErr = err
#pybind11#
#pybind11#        print("\nThe errors for squared exponent covariogram\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#        # now try with the Neural Network covariogram
#pybind11#
#pybind11#        kk = 50
#pybind11#
#pybind11#        nn = gp.NeuralNetCovariogramD()
#pybind11#        nn.setSigma0(1.23)
#pybind11#        nn.setSigma1(0.452)
#pybind11#
#pybind11#        gg.setCovariogram(nn)
#pybind11#        gg.setLambda(0.0045)
#pybind11#
#pybind11#        f = open(os.path.join(testPath,"data", "gp_nn_solutions.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#
#pybind11#        for z in range(len(ff)):
#pybind11#            s = ff[z].split()
#pybind11#            for i in range(dd):
#pybind11#                test[i] = float(s[i])
#pybind11#
#pybind11#            mushld = float(s[dd + kk])
#pybind11#            sigshld = float(s[dd + kk + 1])
#pybind11#
#pybind11#            mu = gg.interpolate(sigma, test, kk)
#pybind11#
#pybind11#            err = (mu - mushld)
#pybind11#            if mushld != 0.0:
#pybind11#                err = err/mushld
#pybind11#
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstMuErr:
#pybind11#                worstMuErr = err
#pybind11#
#pybind11#            err = (sigma[0] - sigshld)
#pybind11#            if sigshld != 0.0:
#pybind11#                err = err/sigshld
#pybind11#
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstSigErr:
#pybind11#                worstSigErr = err
#pybind11#
#pybind11#        print("\nThe errors for neural net covariogram\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#    def testMinMax(self):
#pybind11#        """
#pybind11#        This test will test GaussianProcess.interpolate using the constructor
#pybind11#        that normalizes data point coordinates by minima and maxima.
#pybind11#
#pybind11#        It will only use the squared exponential covariogram (since testInterpolate() presumably
#pybind11#        tested the performance of the neural network covariogram; this test is only concerned
#pybind11#        with the alternate constructor)
#pybind11#
#pybind11#        This test proceeds largely like testInterpolate above
#pybind11#        """
#pybind11#        pp = 2000
#pybind11#        dd = 10
#pybind11#        kk = 50
#pybind11#        tol = 1.0e-4
#pybind11#        data = np.zeros((pp, dd), dtype=float)
#pybind11#        fn = np.zeros((pp), dtype=float)
#pybind11#        test = np.zeros((dd), dtype=float)
#pybind11#        sigma = np.zeros((1), dtype=float)
#pybind11#
#pybind11#        mins = np.zeros((dd), dtype=float)
#pybind11#        maxs = np.zeros((dd), dtype=float)
#pybind11#
#pybind11#        nn = gp.NeuralNetCovariogramD()
#pybind11#        nn.setSigma0(0.555)
#pybind11#        nn.setSigma1(0.112)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            fn[i] = float(s[10])
#pybind11#            for j in range(10):
#pybind11#                data[i][j] = float(s[j])
#pybind11#
#pybind11#        for i in range(pp):
#pybind11#            for j in range(dd):
#pybind11#                if (i == 0) or (data[i][j] < mins[j]):
#pybind11#                    mins[j] = data[i][j]
#pybind11#                if (i == 0) or (data[i][j] > maxs[j]):
#pybind11#                    maxs[j] = data[i][j]
#pybind11#
#pybind11#        mins[2] = 0.0
#pybind11#        maxs[2] = 10.0
#pybind11#        try:
#pybind11#            gg = gp.GaussianProcessD(data, mins, maxs, fn, nn)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        gg.setLambda(0.0045)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_minmax_solutions.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#        for z in range(len(ff)):
#pybind11#            s = ff[z].split()
#pybind11#            for i in range(dd):
#pybind11#                test[i] = float(s[i])
#pybind11#
#pybind11#            mushld = float(s[dd + kk])
#pybind11#            sigshld = float(s[dd + kk + 1])
#pybind11#
#pybind11#            mu = gg.interpolate(sigma, test, kk)
#pybind11#
#pybind11#            err = (mu - mushld)
#pybind11#            if mushld != 0.0:
#pybind11#                err = err/mushld
#pybind11#
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstMuErr:
#pybind11#                worstMuErr = err
#pybind11#
#pybind11#            err = (sigma[0] - sigshld)
#pybind11#            if sigshld != 0.0:
#pybind11#                err = err/sigshld
#pybind11#
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstSigErr:
#pybind11#                worstSigErr = err
#pybind11#
#pybind11#        print("\nThe errors for Gaussian process using min-max normalization\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#    def testAddition(self):
#pybind11#        """
#pybind11#        This will test the performance of interpolation after adding new points
#pybind11#        to GaussianProcess' data set
#pybind11#        """
#pybind11#        pp = 1000
#pybind11#        dd = 10
#pybind11#        kk = 15
#pybind11#        tol = 1.0e-4
#pybind11#
#pybind11#        data = np.zeros((pp, dd), dtype=float)
#pybind11#        fn = np.zeros((pp), dtype=float)
#pybind11#        test = np.zeros((dd), dtype=float)
#pybind11#        sigma = np.zeros((1), dtype=float)
#pybind11#
#pybind11#        xx = gp.SquaredExpCovariogramD()
#pybind11#        xx.setEllSquared(5.0)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_additive_test_root.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            fn[i] = float(s[10])
#pybind11#            for j in range(10):
#pybind11#                data[i][j] = float(s[j])
#pybind11#
#pybind11#        # establish the Gaussian Process
#pybind11#        try:
#pybind11#            gg = gp.GaussianProcessD(data, fn, xx)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        gg.setLambda(0.002)
#pybind11#
#pybind11#        # now add new points to it and see if GaussianProcess.interpolate performs
#pybind11#        # correctly
#pybind11#        f = open(os.path.join(testPath, "data", "gp_additive_test_data.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for z in range(len(ff)):
#pybind11#            s = ff[z].split()
#pybind11#            for i in range(dd):
#pybind11#                test[i] = float(s[i])
#pybind11#                mushld = float(s[dd])
#pybind11#            try:
#pybind11#                gg.addPoint(test, mushld)
#pybind11#            except pex.Exception as e:
#pybind11#                print(e.what())
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_additive_test_solutions.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#
#pybind11#        for z in range(len(ff)):
#pybind11#            s = ff[z].split()
#pybind11#            for i in range(dd):
#pybind11#                test[i] = float(s[i])
#pybind11#
#pybind11#            mushld = float(s[dd + kk])
#pybind11#            sigshld = float(s[dd + kk + 1])
#pybind11#
#pybind11#            mu = gg.interpolate(sigma, test, kk)
#pybind11#
#pybind11#            err = (mu - mushld)
#pybind11#            if mushld != 0:
#pybind11#                err = err/mushld
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstMuErr:
#pybind11#                worstMuErr = err
#pybind11#
#pybind11#            err = (sigma[0] - sigshld)
#pybind11#            if sigshld != 0:
#pybind11#                err = err/sigshld
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if z == 0 or err > worstSigErr:
#pybind11#                worstSigErr = err
#pybind11#
#pybind11#        print("\nThe errors for the test of adding points to the Gaussian process\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#    def testKdTree(self):
#pybind11#        """
#pybind11#        This test will test the construction of KdTree in the pathological case
#pybind11#        where many of the input data points are identical.
#pybind11#        """
#pybind11#        pp = 100
#pybind11#        dd = 5
#pybind11#        data = np.zeros((pp, dd), dtype=float)
#pybind11#        tol = 1.0e-10
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "kd_test_data.sav"))
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(dd):
#pybind11#                data[i][j] = float(s[j])
#pybind11#
#pybind11#        kd = gp.KdTreeD()
#pybind11#        try:
#pybind11#            kd.Initialize(data)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        kds = gp.KdTreeD()
#pybind11#
#pybind11#        try:
#pybind11#            kds.Initialize(data)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        try:
#pybind11#            kds.removePoint(2)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        worstErr = -1.0
#pybind11#        for i in range(100):
#pybind11#            if i > 2:
#pybind11#                dd = 0.0
#pybind11#                for j in range(5):
#pybind11#                    dd = dd+(kd.getData(i, j)-kds.getData(i-1, j))*(kd.getData(i, j)-kds.getData(i-1, j))
#pybind11#                if dd > worstErr:
#pybind11#                    worstErr = dd
#pybind11#        self.assertLess(worstErr, tol)
#pybind11#
#pybind11#        try:
#pybind11#            kd.removePoint(2)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        try:
#pybind11#            kds.removePoint(10)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        for i in range(99):
#pybind11#            if i > 10:
#pybind11#                dd = 0.0
#pybind11#                for j in range(5):
#pybind11#                    dd = dd+(kd.getData(i, j)-kds.getData(i-1, j))*(kd.getData(i, j)-kds.getData(i-1, j))
#pybind11#                if dd > worstErr:
#pybind11#                    worstErr = dd
#pybind11#        self.assertLess(worstErr, tol)
#pybind11#
#pybind11#        try:
#pybind11#            kd.removePoint(10)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        try:
#pybind11#            kds.removePoint(21)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        for i in range(98):
#pybind11#            if i > 21:
#pybind11#                dd = 0.0
#pybind11#                for j in range(5):
#pybind11#                    dd = dd+(kd.getData(i, j)-kds.getData(i-1, j))*(kd.getData(i, j)-kds.getData(i-1, j))
#pybind11#                if dd > worstErr:
#pybind11#                    worstErr = dd
#pybind11#        self.assertLess(worstErr, tol)
#pybind11#        print("\nworst distance error in kdTest ", worstErr, "\n")
#pybind11#
#pybind11#    def testBatch(self):
#pybind11#        """
#pybind11#        This test will test GaussianProcess.batchInterpolate both with
#pybind11#        and without variance calculation
#pybind11#        """
#pybind11#        pp = 100
#pybind11#        dd = 10
#pybind11#        tol = 1.0e-3
#pybind11#
#pybind11#        data = np.zeros((pp, dd), dtype=float)
#pybind11#        fn = np.zeros((pp), dtype=float)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(100):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(dd):
#pybind11#                data[i][j] = float(s[j])
#pybind11#            fn[i] = float(s[dd])
#pybind11#
#pybind11#        xx = gp.SquaredExpCovariogramD()
#pybind11#        xx.setEllSquared(2.0)
#pybind11#
#pybind11#        try:
#pybind11#            gg = gp.GaussianProcessD(data, fn, xx)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        gg.setLambda(0.0032)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_batch_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#
#pybind11#        ntest = len(ff)
#pybind11#        mushld = np.zeros((ntest), dtype=float)
#pybind11#        varshld = np.zeros((ntest), dtype=float)
#pybind11#        mu = np.zeros((ntest), dtype=float)
#pybind11#        var = np.zeros((ntest), dtype=float)
#pybind11#
#pybind11#        queries = np.zeros((ntest, dd), dtype=float)
#pybind11#
#pybind11#        for i in range(ntest):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(dd):
#pybind11#                queries[i][j] = float(s[j])
#pybind11#            mushld[i] = float(s[dd])
#pybind11#            varshld[i] = float(s[dd + 1])
#pybind11#
#pybind11#        # test with variance calculation
#pybind11#        gg.batchInterpolate(mu, var, queries)
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstVarErr = -1.0
#pybind11#        for i in range(ntest):
#pybind11#            err = mu[i]-mushld[i]
#pybind11#            if mushld[i] != 0.0:
#pybind11#                err = err/mushld[i]
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if err > worstMuErr:
#pybind11#                worstMuErr = err
#pybind11#
#pybind11#            err = var[i]-varshld[i]
#pybind11#            if varshld[i] != 0.0:
#pybind11#                err = err/varshld[i]
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if err > worstVarErr:
#pybind11#                worstVarErr = err
#pybind11#
#pybind11#        # test without variance interpolation
#pybind11#        # continue keeping track of worstMuErr
#pybind11#        gg.batchInterpolate(mu, queries)
#pybind11#        for i in range(ntest):
#pybind11#            err = mu[i]-mushld[i]
#pybind11#            if mushld[i] != 0.0:
#pybind11#                err = err/mushld[i]
#pybind11#            if err < 0.0:
#pybind11#                err = -1.0 * err
#pybind11#            if err > worstMuErr:
#pybind11#                worstMuErr = err
#pybind11#
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstVarErr, tol)
#pybind11#
#pybind11#        print("\nThe errors for batch interpolation\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstVarErr)
#pybind11#
#pybind11#    def testSelf(self):
#pybind11#        """
#pybind11#        This test will test GaussianProcess.selfInterpolation
#pybind11#        """
#pybind11#        pp = 2000
#pybind11#        dd = 10
#pybind11#        tol = 1.0e-3
#pybind11#        kk = 20
#pybind11#
#pybind11#        data = np.zeros((pp, dd), dtype=float)
#pybind11#        fn = np.zeros((pp), dtype=float)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_exp_covar_data.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(pp):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(dd):
#pybind11#                data[i][j] = float(s[j])
#pybind11#            fn[i] = float(s[dd])
#pybind11#
#pybind11#        xx = gp.SquaredExpCovariogramD()
#pybind11#        xx.setEllSquared(20.0)
#pybind11#        try:
#pybind11#            gg = gp.GaussianProcessD(data, fn, xx)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        gg.setKrigingParameter(30.0)
#pybind11#        gg.setLambda(0.00002)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_self_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        variance = np.zeros((1), dtype=float)
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#
#pybind11#        for i in range(pp):
#pybind11#            s = ff[i].split()
#pybind11#            mushld = float(s[0])
#pybind11#            sig2shld = float(s[1])
#pybind11#
#pybind11#            try:
#pybind11#                mu = gg.selfInterpolate(variance, i, kk)
#pybind11#            except pex.Exception as e:
#pybind11#                print(e.what())
#pybind11#
#pybind11#            err = mu - mushld
#pybind11#            if mushld != 0.0:
#pybind11#                err = err/mushld
#pybind11#            if err < 0.0:
#pybind11#                err = err * (-1.0)
#pybind11#            if i == 0 or err > worstMuErr:
#pybind11#                worstMuErr = err
#pybind11#
#pybind11#            err = variance[0] - sig2shld
#pybind11#            if sig2shld != 0.0:
#pybind11#                err = err/sig2shld
#pybind11#            if err < 0.0:
#pybind11#                err = err * (-1.0)
#pybind11#            if i == 0 or err > worstSigErr:
#pybind11#                worstSigErr = err
#pybind11#
#pybind11#        print("\nThe errors for self interpolation\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#    def testVector(self):
#pybind11#        """
#pybind11#        This will test interpolate using a vector of functions
#pybind11#        """
#pybind11#
#pybind11#        tol = 1.0e-3
#pybind11#        data = np.zeros((2000, 10), dtype=float)
#pybind11#        fn = np.zeros((2000, 4), dtype=float)
#pybind11#        mu = np.zeros((4), dtype=float)
#pybind11#        sig = np.zeros((4), dtype=float)
#pybind11#        mushld = np.zeros((4), dtype=float)
#pybind11#        vv = np.zeros((10), dtype=float)
#pybind11#        vvf = np.zeros((4), dtype=float)
#pybind11#
#pybind11#        kk = 30
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_vector_data.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(10):
#pybind11#                data[i][j] = float(s[j])
#pybind11#            for j in range(4):
#pybind11#                fn[i][j] = float(s[j+10])
#pybind11#
#pybind11#        nn = gp.NeuralNetCovariogramD()
#pybind11#        nn.setSigma0(2.25)
#pybind11#        nn.setSigma1(0.76)
#pybind11#        try:
#pybind11#            gg = gp.GaussianProcessD(data, fn, nn)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        gg.setLambda(0.0045)
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#        f = open(os.path.join(testPath, "data", "gp_vector_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(10):
#pybind11#                vv[j] = float(s[j])
#pybind11#            for j in range(4):
#pybind11#                mushld[j] = float(s[j+10])
#pybind11#            sigshld = float(s[14])
#pybind11#
#pybind11#            gg.interpolate(mu, sig, vv, kk)
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                muErr = (mu[j]-mushld[j])/mushld[j]
#pybind11#                sigErr = (sig[j]-sigshld)/sigshld
#pybind11#                if muErr < 0.0:
#pybind11#                    muErr = -1.0 * muErr
#pybind11#                if sigErr < 0.0:
#pybind11#                    sigErr = -1.0 * sigErr
#pybind11#                if (muErr > worstMuErr):
#pybind11#                    worstMuErr = muErr
#pybind11#                if (sigErr > worstSigErr):
#pybind11#                    worstSigErr = sigErr
#pybind11#
#pybind11#        print("\nThe errors for vector interpolation\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_vector_selfinterpolate_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            try:
#pybind11#                gg.selfInterpolate(mu, sig, i, kk)
#pybind11#            except pex.Exception as e:
#pybind11#                print(e.what())
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                mushld[j] = float(s[j])
#pybind11#                sigshld = float(s[4])
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                muErr = (mu[j]-mushld[j])/mushld[j]
#pybind11#                if muErr < -1.0:
#pybind11#                    muErr = -1.0 * muErr
#pybind11#                if muErr > worstMuErr:
#pybind11#                    worstMuErr = muErr
#pybind11#
#pybind11#                sigErr = (sig[j]-sigshld)/sigshld
#pybind11#                if sigErr < -1.0:
#pybind11#                    sigErr = -1.0*sigErr
#pybind11#                if sigErr > worstSigErr:
#pybind11#                    worstSigErr = sigErr
#pybind11#
#pybind11#        print("\nThe errors for vector self interpolation\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#        queries = np.zeros((100, 10), dtype=float)
#pybind11#        batchMu = np.zeros((100, 4), dtype=float)
#pybind11#        batchSig = np.zeros((100, 4), dtype=float)
#pybind11#        batchMuShld = np.zeros((100, 4), dtype=float)
#pybind11#        batchSigShld = np.zeros((100, 4), dtype=float)
#pybind11#        batchData = np.zeros((200, 10), dtype=float)
#pybind11#        batchFunctions = np.zeros((200, 4), dtype=float)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_vectorbatch_data.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(10):
#pybind11#                batchData[i][j] = float(s[j])
#pybind11#            for j in range(4):
#pybind11#                batchFunctions[i][j] = float(s[j+10])
#pybind11#
#pybind11#        try:
#pybind11#            ggbatch = gp.GaussianProcessD(batchData, batchFunctions, nn)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        ggbatch.setLambda(0.0045)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_vectorbatch_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(10):
#pybind11#                queries[i][j] = float(s[j])
#pybind11#            for j in range(4):
#pybind11#                batchMuShld[i][j] = float(s[j+10])
#pybind11#            sigShld = float(s[14])
#pybind11#            for j in range(4):
#pybind11#                batchSigShld[i][j] = sigShld
#pybind11#
#pybind11#        ggbatch.batchInterpolate(batchMu, batchSig, queries)
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#        for i in range(100):
#pybind11#            for j in range(4):
#pybind11#                muErr = (batchMu[i][j]-batchMuShld[i][j])/batchMuShld[i][j]
#pybind11#                sigErr = (batchSig[i][j]-batchSigShld[i][j])/batchSigShld[i][j]
#pybind11#                if muErr < 0.0:
#pybind11#                    muErr = muErr * (-1.0)
#pybind11#                if sigErr < 0.0:
#pybind11#                    sigErr = sigErr * (-1.0)
#pybind11#
#pybind11#                if muErr > worstMuErr:
#pybind11#                    worstMuErr = muErr
#pybind11#                if sigErr > worstSigErr:
#pybind11#                    worstSigErr = sigErr
#pybind11#
#pybind11#        print("\nThe errors for vector batch interpolation with variance\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#        ggbatch.batchInterpolate(batchMu, queries)
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#        for i in range(100):
#pybind11#            for j in range(4):
#pybind11#                muErr = (batchMu[i][j]-batchMuShld[i][j])/batchMuShld[i][j]
#pybind11#                if muErr < 0.0:
#pybind11#                    muErr = muErr * (-1.0)
#pybind11#
#pybind11#                if muErr > worstMuErr:
#pybind11#                    worstMuErr = muErr
#pybind11#
#pybind11#        print("\nThe errors for vector batch interpolation without variance\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_vector_add_data.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(10):
#pybind11#                vv[j] = float(s[j])
#pybind11#            for j in range(4):
#pybind11#                vvf[j] = float(s[j+10])
#pybind11#            try:
#pybind11#                gg.addPoint(vv, vvf)
#pybind11#            except pex.Exception as e:
#pybind11#                print(e.what())
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_vector_add_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            try:
#pybind11#                gg.selfInterpolate(mu, sig, i, kk)
#pybind11#            except pex.Exception as e:
#pybind11#                print(e.what())
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                mushld[j] = float(s[j])
#pybind11#            sigshld = float(s[4])
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                muErr = (mu[j]-mushld[j])/mushld[j]
#pybind11#            if muErr < 0.0:
#pybind11#                muErr = -1.0 * muErr
#pybind11#            if muErr > worstMuErr:
#pybind11#                worstMuErr = muErr
#pybind11#
#pybind11#            sigErr = (sig[j]-sigshld)/sigshld
#pybind11#            if sigErr < 0.0:
#pybind11#                sigErr = -1.0*sigErr
#pybind11#            if sigErr > worstSigErr:
#pybind11#                worstSigErr = sigErr
#pybind11#
#pybind11#        print("\nThe errors for vector add interpolation\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#    def testSubtraction(self):
#pybind11#        """
#pybind11#        This will test interpolate after subtracting points
#pybind11#        """
#pybind11#
#pybind11#        tol = 1.0e-3
#pybind11#        data = np.zeros((2000, 10), dtype=float)
#pybind11#        fn = np.zeros((2000, 4), dtype=float)
#pybind11#        mu = np.zeros((4), dtype=float)
#pybind11#        sig = np.zeros((4), dtype=float)
#pybind11#        mushld = np.zeros((4), dtype=float)
#pybind11#        vv = np.zeros((10), dtype=float)
#pybind11#        kk = 30
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_subtraction_data.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(10):
#pybind11#                data[i][j] = float(s[j])
#pybind11#            for j in range(4):
#pybind11#                fn[i][j] = float(s[j+10])
#pybind11#
#pybind11#        xx = gp.SquaredExpCovariogramD()
#pybind11#        xx.setEllSquared(2.3)
#pybind11#        try:
#pybind11#            gg = gp.GaussianProcessD(data, fn, xx)
#pybind11#        except pex.Exception as e:
#pybind11#            print(e.what())
#pybind11#
#pybind11#        gg.setLambda(0.002)
#pybind11#
#pybind11#        j = 1
#pybind11#        for i in range(1000):
#pybind11#            try:
#pybind11#                gg.removePoint(j)
#pybind11#            except pex.Exception as e:
#pybind11#                print(e.what())
#pybind11#
#pybind11#            j = j+1
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#        f = open(os.path.join(testPath, "data", "gp_subtraction_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            for j in range(10):
#pybind11#                vv[j] = float(s[j])
#pybind11#            for j in range(4):
#pybind11#                mushld[j] = float(s[j+10])
#pybind11#            sigshld = float(s[14])
#pybind11#
#pybind11#            gg.interpolate(mu, sig, vv, kk)
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                muErr = (mu[j]-mushld[j])/mushld[j]
#pybind11#                sigErr = (sig[j]-sigshld)/sigshld
#pybind11#                if muErr < 0.0:
#pybind11#                    muErr = -1.0 * muErr
#pybind11#                if sigErr < 0.0:
#pybind11#                    sigErr = -1.0 * sigErr
#pybind11#                if (muErr > worstMuErr):
#pybind11#                    worstMuErr = muErr
#pybind11#                if (sigErr > worstSigErr):
#pybind11#                    worstSigErr = sigErr
#pybind11#
#pybind11#        print("\nThe errors for subtraction interpolation\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#        worstMuErr = -1.0
#pybind11#        worstSigErr = -1.0
#pybind11#
#pybind11#        f = open(os.path.join(testPath, "data", "gp_subtraction_selfinterpolate_solutions.sav"), "r")
#pybind11#        ff = f.readlines()
#pybind11#        f.close()
#pybind11#        for i in range(len(ff)):
#pybind11#            s = ff[i].split()
#pybind11#            try:
#pybind11#                gg.selfInterpolate(mu, sig, i, kk)
#pybind11#            except pex.Exception as e:
#pybind11#                print(e.what())
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                mushld[j] = float(s[j])
#pybind11#                sigshld = float(s[4])
#pybind11#
#pybind11#            for j in range(4):
#pybind11#                muErr = (mu[j]-mushld[j])/mushld[j]
#pybind11#                if muErr < 0.0:
#pybind11#                    muErr = -1.0 * muErr
#pybind11#                if muErr > worstMuErr:
#pybind11#                    worstMuErr = muErr
#pybind11#
#pybind11#                sigErr = (sig[j]-sigshld)/sigshld
#pybind11#                if sigErr < 0.0:
#pybind11#                    sigErr = -1.0*sigErr
#pybind11#                if sigErr > worstSigErr:
#pybind11#                    worstSigErr = sigErr
#pybind11#
#pybind11#        print("\nThe errors for subtraction self interpolation\n")
#pybind11#        print("worst mu error ", worstMuErr)
#pybind11#        print("worst sig2 error ", worstSigErr)
#pybind11#        self.assertLess(worstMuErr, tol)
#pybind11#        self.assertLess(worstSigErr, tol)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
