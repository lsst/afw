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

class GaussianProcessTestCase(unittest.TestCase):
 
  def testInterpolate(self):
    """This will test GaussianProcess.interpolate using both the squared 
    exponential covariogram  and the neural network covariogram on data 
    that was generated with known answers.
  
    The test will check that the code both returns the correct values of
    mu (interpolated function value) and sig2 (the variance) as well as
    the correct nearest neighbor points.
    
    After testing interpolate, this test will verify that the KdTree is
    properly constructed by running GaussianProcess.testKdTree
    
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
    neigh = np.zeros((kk),dtype = np.int32) #indices of nearest neighbors
    sigma = np.zeros((1),dtype = float) #variance
    neighshld = np.zeros((kk),dtype = np.int32) #correct indices of nearest neighbors
    hh = np.zeros((1),dtype = float); #hyper parameters
    
    hh[0] = 100.0
    
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
    gg = gp.GaussianProcessD(dd,pp,data,fn)
    gg.setLambda(0.001)
    gg.setHyperParameters(hh);

    #now, read in the test points and their corresponding known solutions
    f = open("tests/data/gp_exp_covar_solutions.sav")
    ff = f.readlines()
    f.close()

    worstMuErr = 1.0 #keep track of the worst fractional error in mu
    worstSigErr = 1.0 #keep track of the worst fractional error in the variance
    worstFbarErr = 1.0 #keep track of the worst fractional error in fbar,
                     #the mean of the function values at the nearest neighbor points
    for z in range(len(ff)):
      s = ff[z].split() #s will store the zth line of the solution file
      for i in range(dd):
        test[i] = float(s[i]) #read in the test point coordinates
    
      for i in range(kk):
        neighshld[i] = int(s[dd + i]) #read in what the nearest neighbor indices should be
    
      mushld = float(s[dd + kk]) #read in what mu should be
      sigshld = float(s[dd + kk + 1]) #read in what the variance should be
  
      mu = gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      #check that GaussianProcess found the right nearest neighbors
      for i in range(kk): 
        self.assertEqual(neigh[i],neighshld[i])
	
  
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
    
      #the test on fbar below is probably redundant, since we already
      #checked that Gaussian Process found the correct nearest neighbors
      fbar = 0.0
      for i in range(kk):
        fbar = fbar + fn[neigh[i]]
      fbar = fbar/float(kk)
      nn = float(s[dd + kk + 2])
      err = (fbar-nn)
      if nn !=  0.0:
        err = err/nn
      
      if err < 0.0:
        err = -1.0 * err
      if z == 0 or err > worstFbarErr:
        worstFbarErr = err
    
    print "\nThe errors for squared exponent covariogram\n"
    print "worst mu error ",worstMuErr
    print "worst sig2 error ",worstSigErr
    print "worst fbar error ",worstFbarErr
    
    self.assertTrue(worstMuErr < tol)
    self.assertTrue(worstSigErr < tol)
    self.assertTrue(worstFbarErr < tol)
    
    #test that the KdTree was properly constructed
    i = gg.testKdTree();
    self.assertEqual(i,1)
    
    #now try with the Neural Network covariogram
    
    kk = 50
    neigh = np.zeros((kk),dtype = np.int32)
    neighshld = np.zeros((kk),dtype = np.int32)
    hh = np.zeros((2),dtype = float)
    hh[0] = 1.23
    hh[1] = 0.452
    gg.setLambda(0.0045)
    
    gg.setCovariogramType(gg.neuralNetwork) #set the covariogram to the neural network
    gg.setHyperParameters(hh)
    f = open("tests/data/gp_nn_solutions.sav")
    ff = f.readlines()
    f.close()

    worstMuErr = 1.0
    worstSigErr = 1.0
    worstFbarErr = 1.0
    for z in range(len(ff)):
      s = ff[z].split()
      for i in range(dd):
        test[i] = float(s[i])
    
      for i in range(kk):
        neighshld[i] = int(s[dd + i])
    
      mushld = float(s[dd + kk])
      sigshld = float(s[dd + kk + 1])
  
      mu = gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      for i in range(kk):
        self.assertEqual(neigh[i],neighshld[i])
	
  
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
    
      fbar = 0.0
      for i in range(kk):
        fbar = fbar + fn[neigh[i]]
      fbar = fbar/float(kk)
      nn = float(s[dd + kk + 2])
      err = (fbar-nn)
      if nn !=  0.0:
        err = err/nn
	
      if err < 0.0:
        err = -1.0 * err
      if z == 0 or err > worstFbarErr:
        worstFbarErr = err
    
    print "\nThe errors for neural net covariogram\n"
    print "worst mu error ",worstMuErr
    print "worst sig2 error ",worstSigErr
    print "worst fbar error ",worstFbarErr
    
    self.assertTrue(worstMuErr < tol)
    self.assertTrue(worstSigErr < tol)
    self.assertTrue(worstFbarErr < tol)
  
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
    neigh = np.zeros((kk),dtype = np.int32)
    sigma = np.zeros((1),dtype = float)
    neighshld = np.zeros((kk),dtype = np.int32)
    hh = np.zeros((2),dtype = float);
    mins = np.zeros((dd),dtype = float)
    maxs = np.zeros((dd),dtype = float)
   
    hh[0] = 0.555
    hh[1] = 0.112

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
    gg = gp.GaussianProcessD(dd,pp,data,mins,maxs,fn)
    gg.setLambda(0.0045)
    gg.setCovariogramType(gg.neuralNetwork)
    gg.setHyperParameters(hh);
   
    f = open("tests/data/gp_minmax_solutions.sav")
    ff = f.readlines()
    f.close()

    worstMuErr = 1.0
    worstSigErr = 1.0
    worstFbarErr = 1.0
    for z in range(len(ff)):
      s = ff[z].split()
      for i in range(dd):
        test[i] = float(s[i])
    
      for i in range(kk):
        neighshld[i] = int(s[dd + i])
    
      mushld = float(s[dd + kk])
      sigshld = float(s[dd + kk + 1])
  
      mu = gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      for i in range(kk):
        self.assertEqual(neigh[i],neighshld[i])
	
  
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
    
      fbar = 0.0
      for i in range(kk):
        fbar = fbar + fn[neigh[i]]
      fbar = fbar/float(kk)
      nn = float(s[dd + kk + 2])
      err = (fbar-nn)
      if nn !=  0.0:
        err = err/nn
      
      if err < 0.0:
        err = -1.0 * err
      if z == 0 or err > worstFbarErr:
        worstFbarErr = err
    print "\nThe errors for Gaussian process using min-max normalization\n"
    print "worst mu error ",worstMuErr
    print "worst sig2 error ",worstSigErr
    print "worstf bar error ",worstFbarErr
    
    self.assertTrue(worstMuErr < tol)
    self.assertTrue(worstSigErr < tol)
    self.assertTrue(worstFbarErr < tol)
    
    #test that the KdTree was properly constructed
    i = gg.testKdTree();
    self.assertEqual(i,1)
 
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
    neigh = np.zeros((kk),dtype = np.int32)
    sigma = np.zeros((1),dtype = float)
    neighshld = np.zeros((kk),dtype = np.int32)
    hh = np.zeros((1),dtype = float);
    
    hh[0] = 5.0

    f = open("tests/data/gp_additive_test_root.sav")
    ff = f.readlines()
    f.close()
    
   
    for i in range(len(ff)):
     s=ff[i].split()
     fn[i] = float(s[10])
     for j in range(10):
       data[i][j] = float(s[j])
  
    #establish the Gaussian Process
    gg = gp.GaussianProcessD(dd,pp,data,fn)
    gg.setLambda(0.002)
    gg.setHyperParameters(hh)
    #print "build the gp"
    
    #now add new points to it and see if GaussianProcess.interpolate performs
    #correctly spock
    f = open("tests/data/gp_additive_test_data.sav")
    ff = f.readlines()
    f.close()
    for z in range(len(ff)):
      s = ff[z].split()
      for i in range(dd):
	test[i] = float(s[i])
      mushld = float(s[dd])
      gg.addPoint(test,mushld)

    f = open("tests/data/gp_additive_test_solutions.sav")
    ff = f.readlines()
    f.close()

    worstMuErr = 1.0 
    worstSigErr = 1.0 
    worstFbarErr = 1.0 
                     
    for z in range(len(ff)):
      s = ff[z].split()
      for i in range(dd):
        test[i] = float(s[i])
    
      for i in range(kk):
        neighshld[i] = int(s[dd + i])
    
      mushld = float(s[dd + kk])
      sigshld = float(s[dd + kk + 1])
  
      mu = gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      for i in range(kk):
        self.assertEqual(neigh[i],neighshld[i])
	
  
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
   
    #test that the KdTree was properly constructed
    i = gg.testKdTree();
    self.assertEqual(i,1)
  
  def testKdTree(self):
    """
    This test will test the construction of KdTree in the pathological case
    where many of the input data points are identical.
    """
    pp = 100
    dd = 5
    data = np.zeros((pp,dd),dtype = float)
    fn = np.zeros((pp),dtype = float)
    
    f = open("tests/data/kd_test_data.sav")
    ff = f.readlines()
    f.close()
    for i in range(len(ff)):
      s = ff[i].split()
      for j in range(dd):
        data[i][j] = float(s[j])
      fn[i] = float(s[dd])
    
    gg = gp.GaussianProcessD(dd,pp,data,fn)
    i = gg.testKdTree()
    self.assertEqual(i,1)

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
    
    gg = gp.GaussianProcessD(dd,pp,data,fn)
    gg.setLambda(0.0032)
    hh = np.zeros((1),dtype = float)
    hh[0] = 2.0
    gg.setHyperParameters(hh)
    
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
    gg.batchInterpolate(queries,mu,var,ntest)
    
    worstMuErr = -1.0
    worstvarerr = -1.0
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
      if err > worstvarerr:
        worstvarerr = err

    #test without variance interpolation
    #continue keeping track of worstMuErr
    gg.batchInterpolate(queries,mu,ntest)   
    for i in range(ntest):
      err = mu[i]-mushld[i]
      if mushld[i] !=  0.0:
        err = err/mushld[i]
      if err < 0.0:
        err = -1.0 * err
      if err > worstMuErr:
        worstMuErr = err

    self.assertTrue(worstMuErr < tol)
    self.assertTrue(worstvarerr < tol)
    
    print "\nThe errors for batch interpolation\n"
    print "worst mu error ",worstMuErr
    print "worst sig2 error ",worstvarerr

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
    
    gg = gp.GaussianProcessD(dd,pp,data,fn)
    gg.setKrigingParameter(30.0)
    gg.setLambda(0.00002)
    
    f = open("tests/data/gp_self_solutions.sav","r")
    ff = f.readlines()
    f.close()
    variance = np.zeros((1),dtype = float)
    neighshld = np.zeros((kk),dtype = np.int32)
    neigh = np.zeros((kk),dtype = np.int32)
    
    hh = np.zeros((1),dtype = float)
    hh[0] = 20.0
    gg.setHyperParameters(hh)
    
    worstMuErr = -1.0
    worstSigErr = -1.0
    
    for i in range(pp):
      s = ff[i].split()
      mushld = float(s[0])
      sig2shld = float(s[1])
      for j in range(kk):
        neighshld[j] = np.int32(s[j + 2])
      
      mu = gg.selfInterpolate(i,variance,kk)
      gg.getNeighbors(neigh)
      
      for j in range(kk):
        self.assertEqual(neigh[j],neighshld[j])
      
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

def suite():
  utilsTests.init()
  suites = []
  suites  +=  unittest.makeSuite(GaussianProcessTestCase)
  return unittest.TestSuite(suites)

def run(shouldExit = False):
  utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
  run(True)  
