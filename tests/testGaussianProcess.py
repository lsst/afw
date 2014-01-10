#!/usr/bin/env python

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

import os
import unittest
import warnings
import sys
import numpy as np
import lsst.afw.math as gp
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pex

class GaussianProcessTestCase(unittest.TestCase):
    
    def testTooManyNeighbors(self):
        """
        Test that GaussianProcess checks if too many neighbours are requested
        """
        nData = 100                        # number of data points
        dimen = 10                         # dimension of each point
        data = np.zeros((nData,dimen))
        fn = np.zeros(nData)
        gg = gp.GaussianProcessD(data, fn, gp.SquaredExpCovariogramD())
        test = np.zeros(dimen)
        sigma = np.empty(1)
	mu_arr = np.empty(1)
	
        try:
            mu = gg.interpolate(sigma, test, 2*nData)
            self.assertTrue(False, "Failed to catch using too many points")
        except pex.LsstCppException, e:
            self.assertTrue(True)
        
        try:
            gg.interpolate(mu_arr, sigma, 2*nData)
            self.assertTrue(False, "Failed to catch using too many points")
        except pex.LsstCppException, e:
            self.assertTrue(True)

        try:
            mu = gg.selfInterpolate(sigma, 0, 2*nData)
            self.assertTrue(False, "Failed to catch using too many points")
        except pex.LsstCppException, e:
            self.assertTrue(True)

        try:
            mu = gg.selfInterpolate(sigma, -1, 2*nData)
            self.assertTrue(False, 
            "Failed to catch selfInterpolating non-existent point")
   
        except pex.LsstCppException, e:
            self.assertTrue(True)

        try:
            mu = gg.selfInterpolate(sigma, nData, 2*nData)
            self.assertTrue(False, 
            "Failed to catch selfInterpolating non-existent point")

        except pex.LsstCppException, e:
            self.assertTrue(True)

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

        pp = 2000 #number of data points
        dd = 10 #number of dimensions
        kk = 15 #number of nearest neighbors being used
        tol = 1.0e-3 #the largest relative error that will be tolerated

        data = np.zeros((pp,dd),dtype = float) #input data points
        fn = np.zeros((pp),dtype = float) #input function values
        test = np.zeros((dd),dtype = float) #query points
        sigma = np.zeros((1),dtype = float) #variance

        xx=gp.SquaredExpCovariogramD()
        xx.setEllSquared(100.0)

        #read in the input data
        f = open("tests/data/gp_exp_covar_data.sav")
        ff = f.readlines()
        f.close()

        for i in range(len(ff)):
            s = ff[i].split()
            fn[i] = float(s[10])
            for j in range(10):
                data[i][j] = float(s[j])

        #first try the squared exponential covariogram (the default)
        try:
            gg = gp.GaussianProcessD(data,fn,xx)
        except pex.LsstCppException, e:
            print e.args[0].what()

        gg.setLambda(0.001)

        #now, read in the test points and their corresponding known solutions
        f = open("tests/data/gp_exp_covar_solutions.sav")
        ff = f.readlines()
        f.close()

        worstMuErr = -1.0 #keep track of the worst fractional error in mu
        worstSigErr = -1.0 #keep track of the worst fractional error in the variance

        for z in range(len(ff)):
            s = ff[z].split() #s will store the zth line of the solution file
            for i in range(dd):
                test[i] = float(s[i]) #read in the test point coordinates

            mushld = float(s[dd + kk]) #read in what mu should be
            sigshld = float(s[dd + kk + 1]) #read in what the variance should be

            mu = gg.interpolate(sigma,test,kk)

            err = (mu - mushld)
            if mushld !=  0.0:
                err = err/mushld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstMuErr:
                worstMuErr = err

            err = (sigma[0] - sigshld)
            if sigshld !=  0.0:
                err = err/sigshld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstSigErr:
                worstSigErr = err

        print "\nThe errors for squared exponent covariogram\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr

        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

        #now try with the Neural Network covariogram

        kk = 50

        nn=gp.NeuralNetCovariogramD()
        nn.setSigma0(1.23)
        nn.setSigma1(0.452)

        gg.setCovariogram(nn)
        gg.setLambda(0.0045)

        f = open("tests/data/gp_nn_solutions.sav")
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

            mu = gg.interpolate(sigma,test,kk)

            err = (mu - mushld)
            if mushld !=  0.0:
                err = err/mushld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstMuErr:
                worstMuErr = err

            err = (sigma[0] - sigshld)
            if sigshld !=  0.0:
                err = err/sigshld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstSigErr:
                worstSigErr = err

        print "\nThe errors for neural net covariogram\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr

        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

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
        data = np.zeros((pp,dd),dtype = float)
        fn = np.zeros((pp),dtype = float)
        test = np.zeros((dd),dtype = float)
        sigma = np.zeros((1),dtype = float)

        mins = np.zeros((dd),dtype = float)
        maxs = np.zeros((dd),dtype = float)

        nn=gp.NeuralNetCovariogramD()
        nn.setSigma0(0.555)
        nn.setSigma1(0.112)

        f = open("tests/data/gp_exp_covar_data.sav")
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
            gg = gp.GaussianProcessD(data,mins,maxs,fn,nn)
        except pex.LsstCppException, e:
            print e.args[0].what()

        gg.setLambda(0.0045)

        f = open("tests/data/gp_minmax_solutions.sav")
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

            mu = gg.interpolate(sigma,test,kk)

            err = (mu - mushld)
            if mushld !=  0.0:
                err = err/mushld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstMuErr:
                 worstMuErr = err

            err = (sigma[0] - sigshld)
            if sigshld !=  0.0:
                err = err/sigshld

            if err < 0.0:
                err = -1.0 * err
            if z == 0 or err > worstSigErr:
                worstSigErr = err

        print "\nThe errors for Gaussian process using min-max normalization\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr

        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

    def testAddition(self):
        """
        This will test the performance of interpolation after adding new points
        to GaussianProcess' data set
        """
        pp = 1000
        dd = 10
        kk = 15
        tol = 1.0e-4

        data = np.zeros((pp,dd),dtype = float)
        fn = np.zeros((pp),dtype = float)
        test = np.zeros((dd),dtype = float)
        sigma = np.zeros((1),dtype = float)

        xx=gp.SquaredExpCovariogramD()
        xx.setEllSquared(5.0)

        f = open("tests/data/gp_additive_test_root.sav")
        ff = f.readlines()
        f.close()


        for i in range(len(ff)):
            s=ff[i].split()
            fn[i] = float(s[10])
            for j in range(10):
                data[i][j] = float(s[j])

        #establish the Gaussian Process
        try:
            gg = gp.GaussianProcessD(data,fn,xx)
        except pex.LsstCppException, e:
            print e.args[0].what()

        gg.setLambda(0.002)

        #now add new points to it and see if GaussianProcess.interpolate performs
        #correctly
        f = open("tests/data/gp_additive_test_data.sav")
        ff = f.readlines()
        f.close()
        for z in range(len(ff)):
            s = ff[z].split()
            for i in range(dd):
                test[i] = float(s[i])
                mushld = float(s[dd])
            try:
                gg.addPoint(test,mushld)
            except pex.LsstCppException,e:
                print e.args[0].what()


        f = open("tests/data/gp_additive_test_solutions.sav")
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

            mu = gg.interpolate(sigma,test,kk)

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


        print "\nThe errors for the test of adding points to the Gaussian process\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr


        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

    def testKdTree(self):
        """
        This test will test the construction of KdTree in the pathological case
        where many of the input data points are identical.
        """
        pp = 100
        dd = 5
        data = np.zeros((pp,dd),dtype = float)
        tol=1.0e-10

        f = open("tests/data/kd_test_data.sav")
        ff = f.readlines()
        f.close()
        for i in range(len(ff)):
            s = ff[i].split()
            for j in range(dd):
                data[i][j] = float(s[j])

        kd = gp.KdTreeD()
        try:
            kd.Initialize(data)
        except pex.LsstCppException, e:
            print e.args[0].what()

        kds = gp.KdTreeD()

        try:
            kds.Initialize(data)
        except pex.LsstCppException, e:
            print e.args[0].what()

        try:
            kds.removePoint(2)
        except pex.LsstCppException, e:
            print e.args[0].what()

        worstErr=-1.0
        for i in range(100):
            if i > 2:
                dd=0.0
                for j in range(5):
                    dd = dd+(kd.getData(i,j)-kds.getData(i-1,j))*(kd.getData(i,j)-kds.getData(i-1,j))
                if dd>worstErr:
                    worstErr=dd
        self.assertTrue(worstErr<tol)

        try:
            kd.removePoint(2)
        except pex.LsstCppException, e:
            print e.args[0].what()

        try:
            kds.removePoint(10)
        except pex.LsstCppException, e:
            print e.args[0].what()

        for i in range(99):
            if i > 10:
                dd=0.0
                for j in range(5):
                    dd = dd+(kd.getData(i,j)-kds.getData(i-1,j))*(kd.getData(i,j)-kds.getData(i-1,j))
                if dd>worstErr:
                    worstErr=dd
        self.assertTrue(worstErr<tol)

        try:
            kd.removePoint(10)
        except pex.LsstCppException, e:
            print e.args[0].what()

        try:
            kds.removePoint(21)
        except pex.LsstCppException, e:
            print e.args[0].what()

        for i in range(98):
            if i > 21:
                dd=0.0
                for j in range(5):
                    dd = dd+(kd.getData(i,j)-kds.getData(i-1,j))*(kd.getData(i,j)-kds.getData(i-1,j))
                if dd>worstErr:
                    worstErr=dd
        self.assertTrue(worstErr<tol)
        print "\nworst distance error in kdTest ",worstErr,"\n"


    def testBatch(self):
        """
        This test will test GaussianProcess.batchInterpolate both with
        and without variance calculation
        """
        pp = 100
        dd = 10
        tol = 1.0e-3

        data = np.zeros((pp,dd),dtype = float)
        fn = np.zeros((pp),dtype = float)

        f = open("tests/data/gp_exp_covar_data.sav","r");
        ff = f.readlines()
        f.close()
        for i in range(100):
            s = ff[i].split()
            for j in range(dd):
                data[i][j] = float(s[j])
            fn[i] = float(s[dd])

        xx=gp.SquaredExpCovariogramD();
        xx.setEllSquared(2.0)

        try:
            gg = gp.GaussianProcessD(data,fn,xx)
        except pex.LsstCppException, e:
            print e.args[0].what()

        gg.setLambda(0.0032)

        f = open("tests/data/gp_batch_solutions.sav","r")
        ff = f.readlines()
        f.close()

        ntest = len(ff)
        mushld = np.zeros((ntest),dtype = float)
        varshld = np.zeros((ntest),dtype = float)
        mu = np.zeros((ntest),dtype = float)
        var = np.zeros((ntest),dtype = float)

        queries = np.zeros((ntest,dd),dtype = float)

        for i in range(ntest):
            s = ff[i].split()
            for j in range(dd):
                queries[i][j] = float(s[j])
            mushld[i] = float(s[dd])
            varshld[i] = float(s[dd + 1])

        #test with variance calculation
        gg.batchInterpolate(mu,var,queries)

        worstMuErr = -1.0
        worstVarErr = -1.0
        for i in range(ntest):
            err = mu[i]-mushld[i]
            if mushld[i] !=  0.0:
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

        #test without variance interpolation
        #continue keeping track of worstMuErr
        gg.batchInterpolate(mu,queries)
        for i in range(ntest):
            err = mu[i]-mushld[i]
            if mushld[i] !=  0.0:
                err = err/mushld[i]
            if err < 0.0:
                err = -1.0 * err
            if err > worstMuErr:
                worstMuErr = err

        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstVarErr < tol)

        print "\nThe errors for batch interpolation\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstVarErr

    def testSelf(self):
        """
        This test will test GaussianProcess.selfInterpolation
        """
        pp = 2000
        dd = 10
        tol = 1.0e-3
        kk = 20

        data = np.zeros((pp,dd),dtype = float)
        fn = np.zeros((pp),dtype = float)

        f = open("tests/data/gp_exp_covar_data.sav","r");
        ff = f.readlines()
        f.close()
        for i in range(pp):
            s = ff[i].split()
            for j in range(dd):
                data[i][j] = float(s[j])
            fn[i] = float(s[dd])


        xx=gp.SquaredExpCovariogramD()
        xx.setEllSquared(20.0)
        try:
            gg = gp.GaussianProcessD(data,fn,xx)
        except pex.LsstCppException, e:
            print e.args[0].what()

        gg.setKrigingParameter(30.0)
        gg.setLambda(0.00002)

        f = open("tests/data/gp_self_solutions.sav","r")
        ff = f.readlines()
        f.close()
        variance = np.zeros((1),dtype = float)

        worstMuErr = -1.0
        worstSigErr = -1.0

        for i in range(pp):
            s = ff[i].split()
            mushld = float(s[0])
            sig2shld = float(s[1])

            try:
                mu = gg.selfInterpolate(variance,i,kk)
            except pex.LsstCppException, e:
                print e.args[0].what()


            err = mu - mushld
            if mushld !=  0.0:
                err = err/mushld
            if err < 0.0:
                err = err * (-1.0)
            if i == 0 or err > worstMuErr:
                worstMuErr = err

            err = variance[0] - sig2shld
            if sig2shld !=  0.0:
                err = err/sig2shld
            if err < 0.0:
                err = err * (-1.0)
            if i == 0 or err > worstSigErr:
                worstSigErr = err

        print "\nThe errors for self interpolation\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr
        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

    def testVector(self):
        """
        This will test interpolate using a vector of functions
        """

        tol=1.0e-3
        data=np.zeros((2000,10), dtype = float)
        fn=np.zeros((2000,4), dtype = float)
        mu=np.zeros((4), dtype=float)
        sig=np.zeros((4), dtype = float)
        mushld=np.zeros((4), dtype = float)
        vv=np.zeros((10), dtype = float)
        vvf=np.zeros((4), dtype = float)

        kk=30

        ll=0.0045

        f=open("tests/data/gp_vector_data.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            for j in range(10):
                data[i][j]=float(s[j])
            for j in range(4):
                fn[i][j]=float(s[j+10])

        nn=gp.NeuralNetCovariogramD()
        nn.setSigma0(2.25)
        nn.setSigma1(0.76)
        try:
            gg=gp.GaussianProcessD(data,fn,nn);
        except pex.LsstCppException, e:
            print e.args[0].what()

        gg.setLambda(0.0045)

        worstMuErr=-1.0
        worstSigErr=-1.0
        f=open("tests/data/gp_vector_solutions.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            for j in range(10):
                vv[j]=float(s[j])
            for j in range(4):
                mushld[j]=float(s[j+10])
            sigshld=float(s[14])

            gg.interpolate(mu,sig,vv,kk)

            for j in range(4):
                muErr= (mu[j]-mushld[j])/mushld[j]
                sigErr = (sig[j]-sigshld)/sigshld
                if muErr < 0.0:
                    muErr = -1.0 * muErr
                if sigErr < 0.0:
                    sigErr = -1.0 * sigErr
                if (muErr > worstMuErr):
                    worstMuErr=muErr
                if (sigErr > worstSigErr):
                    worstSigErr=sigErr


        print "\nThe errors for vector interpolation\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr
        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

        worstMuErr=-1.0
        worstSigErr=-1.0

        f=open("tests/data/gp_vector_selfinterpolate_solutions.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            try:
                gg.selfInterpolate(mu,sig,i,kk);
            except pex.LsstCppException, e:
                print e.args[0].what()

            for j in range(4):
                mushld[j]=float(s[j])
                sigshld=float(s[4])

            for j in range(4):
                muErr=(mu[j]-mushld[j])/mushld[j]
                if muErr < -1.0:
                    muErr=-1.0 * muErr
                if muErr>worstMuErr:
                    worstMuErr=muErr

                sigErr=(sig[j]-sigshld)/sigshld
                if sigErr<-1.0:
                    sigErr = -1.0*sigErr
                if sigErr>worstSigErr:
                    worstSigErr=sigErr



        print "\nThe errors for vector self interpolation\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr
        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

        queries=np.zeros((100,10), dtype = float)
        batchMu=np.zeros((100,4), dtype = float)
        batchSig=np.zeros((100,4), dtype = float)
        batchMuShld=np.zeros((100,4), dtype = float)
        batchSigShld=np.zeros((100,4), dtype = float)
        batchData=np.zeros((200,10), dtype = float)
        batchFunctions=np.zeros((200,4), dtype = float)

        f=open("tests/data/gp_vectorbatch_data.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            for j in range(10):
                batchData[i][j]=float(s[j])
            for j in range(4):
                batchFunctions[i][j]=float(s[j+10])


        try:
            ggbatch=gp.GaussianProcessD(batchData,batchFunctions,nn)
        except pex.LsstCppException, e:
            print e.args[0].what()

        ggbatch.setLambda(0.0045)

        f=open("tests/data/gp_vectorbatch_solutions.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            for j in range(10):
                queries[i][j]=float(s[j])
            for j in range(4):
                batchMuShld[i][j]=float(s[j+10])
            sigShld=float(s[14])
            for j in range(4):
                batchSigShld[i][j]=sigShld


        ggbatch.batchInterpolate(batchMu,batchSig,queries)
        worstMuErr=-1.0
        worstSigErr=-1.0
        for i in range(100):
            for j in range(4):
                muErr=(batchMu[i][j]-batchMuShld[i][j])/batchMuShld[i][j]
                sigErr=(batchSig[i][j]-batchSigShld[i][j])/batchSigShld[i][j]
                if muErr < 0.0:
                    muErr = muErr * (-1.0)
                if sigErr < 0.0:
                    sigErr = sigErr * (-1.0)

                if muErr>worstMuErr:
                    worstMuErr=muErr
                if sigErr>worstSigErr:
                    worstSigErr=sigErr

        print "\nThe errors for vector batch interpolation with variance\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr
        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

        ggbatch.batchInterpolate(batchMu,queries)
        worstMuErr=-1.0
        worstSigErr=-1.0
        for i in range(100):
            for j in range(4):
                muErr=(batchMu[i][j]-batchMuShld[i][j])/batchMuShld[i][j]
                if muErr < 0.0:
                    muErr = muErr * (-1.0)

                if muErr>worstMuErr:
                    worstMuErr=muErr


        print "\nThe errors for vector batch interpolation without variance\n"
        print "worst mu error ",worstMuErr
        self.assertTrue(worstMuErr < tol)


        f=open("tests/data/gp_vector_add_data.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            for j in range(10):
                vv[j]=float(s[j])
            for j in range(4):
                vvf[j]=float(s[j+10])
            try:
                gg.addPoint(vv,vvf)
            except pex.LsstCppException, e:
                print e.args[0].what()

        f=open("tests/data/gp_vector_add_solutions.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            try:
                gg.selfInterpolate(mu,sig,i,kk);
            except pex.LsstCppException, e:
                print e.args[0].what()

            for j in range(4):
                mushld[j]=float(s[j])
            sigshld=float(s[4])

            for j in range(4):
                muErr=(mu[j]-mushld[j])/mushld[j]
            if muErr < 0.0:
                muErr=-1.0 * muErr
            if muErr>worstMuErr:
                worstMuErr=muErr
                worstMu=mu[j]

            sigErr=(sig[j]-sigshld)/sigshld
            if sigErr<0.0:
                sigErr = -1.0*sigErr
            if sigErr>worstSigErr:
                worstSigErr=sigErr

        print "\nThe errors for vector add interpolation\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr

        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)


    def testSubtraction(self):
        """
        This will test interpolate after subtracting points
        """

        tol=1.0e-3
        data=np.zeros((2000,10), dtype = float)
        fn=np.zeros((2000,4), dtype = float)
        mu=np.zeros((4), dtype=float)
        sig=np.zeros((4), dtype = float)
        mushld=np.zeros((4), dtype = float)
        vv=np.zeros((10), dtype = float)
        kk=30
        ll=0.002

        f=open("tests/data/gp_subtraction_data.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            for j in range(10):
                data[i][j]=float(s[j])
            for j in range(4):
                fn[i][j]=float(s[j+10])

        xx=gp.SquaredExpCovariogramD()
        xx.setEllSquared(2.3)
        try:
            gg=gp.GaussianProcessD(data,fn,xx);
        except pex.LsstCppException, e:
            print e.args[0].what()

        gg.setLambda(0.002)

        j=1
        for i in range(1000):
            try:
                gg.removePoint(j)
            except pex.LsstCppException, e:
                print e.args[0].what()

            j=j+1

        worstMuErr=-1.0
        worstSigErr=-1.0
        f=open("tests/data/gp_subtraction_solutions.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            for j in range(10):
                vv[j]=float(s[j])
            for j in range(4):
                mushld[j]=float(s[j+10])
            sigshld=float(s[14])

            gg.interpolate(mu,sig,vv,kk)

            for j in range(4):
                muErr= (mu[j]-mushld[j])/mushld[j]
                sigErr = (sig[j]-sigshld)/sigshld
                if muErr < 0.0:
                    muErr = -1.0 * muErr
                if sigErr < 0.0:
                    sigErr = -1.0 * sigErr
                if (muErr > worstMuErr):
                    worstMuErr=muErr
                if (sigErr > worstSigErr):
                    worstSigErr=sigErr


        print "\nThe errors for subtraction interpolation\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr
        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

        worstMuErr=-1.0
        worstSigErr=-1.0

        f=open("tests/data/gp_subtraction_selfinterpolate_solutions.sav","r")
        ff=f.readlines()
        f.close()
        for i in range(len(ff)):
            s=ff[i].split()
            try:
                gg.selfInterpolate(mu,sig,i,kk);
            except pex.LsstCppException, e:
                print e.args[0].what()

            for j in range(4):
                mushld[j]=float(s[j])
                sigshld=float(s[4])

            for j in range(4):
                muErr=(mu[j]-mushld[j])/mushld[j]
                if muErr < 0.0:
                    muErr=-1.0 * muErr
                if muErr>worstMuErr:
                    worstMuErr=muErr

                sigErr=(sig[j]-sigshld)/sigshld
                if sigErr<0.0:
                    sigErr = -1.0*sigErr
                if sigErr>worstSigErr:
                    worstSigErr=sigErr

        print "\nThe errors for subtraction self interpolation\n"
        print "worst mu error ",worstMuErr
        print "worst sig2 error ",worstSigErr
        self.assertTrue(worstMuErr < tol)
        self.assertTrue(worstSigErr < tol)

def suite():
    utilsTests.init()
    suites = []
    suites  +=  unittest.makeSuite(GaussianProcessTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)
