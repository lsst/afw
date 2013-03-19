import os
import unittest
import warnings
import sys
import numpy as np
import lsst.afw.math as gp
import lsst.utils.tests as utilsTests

class GaussianProcessTestCase(unittest.TestCase):
 
  def testExpCovar(self):
    pp=2000
    dd=10
    kk=15
    tol=1.0e-3

    data=np.zeros((pp,dd),dtype=float)
    fn=np.zeros((pp),dtype=float)
    test=np.zeros((dd),dtype=float)
    neigh=np.zeros((kk),dtype=np.int32)
    sigma=np.zeros((1),dtype=float)
    neighshld=np.zeros((kk),dtype=np.int32)
    hh=np.zeros((1),dtype=float);
    
    hh[0]=100.0

    f=open("tests/data/gp_exp_covar_data.sav")
    ff=f.readlines()
    f.close()
    fff=[]
    for i in range(len(ff)):
      fff.append(ff[i].split())

    for i in range(len(fff)):
     fn[i]=float(fff[i][10])
     for j in range(10):
       data[i][j]=float(fff[i][j])
  
    gg=gp.GaussianProcessD(dd,pp,data,fn)
    gg.setLambda(0.001)
    gg.setHyperParameters(hh);
    #print "build the gp"

    s=open("tests/data/gp_exp_covar_solutions.sav")
    ss=s.readlines()
    s.close()

    worstmuerr=1.0
    worstsigerr=1.0
    worstfbarerr=1.0
    for z in range(len(ss)):
      sss=ss[z].split()
      for i in range(dd):
        test[i]=float(sss[i])
    
      for i in range(kk):
        neighshld[i]=int(sss[dd+i])
    
      mushld=float(sss[dd+kk])
      sigshld=float(sss[dd+kk+1])
  
      mu=gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      for i in range(kk):
        self.assertEqual(neigh[i],neighshld[i])
	
  
      err=(mu-mushld)
      if mushld != 0.0:
        err=err/mushld
	
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstmuerr:
        worstmuerr=err
  
  
  
      err=(sigma[0]-sigshld)
      if sigshld != 0.0:
        err=err/sigshld
	
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstsigerr:
        worstsigerr=err
    
      fbar=0.0
      for i in range(kk):
        fbar=fbar+fn[neigh[i]]
      fbar=fbar/float(kk)
      nn=float(sss[dd+kk+2])
      err=(fbar-nn)
      if nn != 0.0:
        err=err/nn
      
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstfbarerr:
        worstfbarerr=err
    
    print "\nThe errors for squared exponent covariogram\n"
    print "worst mu error ",worstmuerr
    print "worst sig2 error ",worstsigerr
    print "worst fbar error ",worstfbarerr
    
    self.assertTrue(worstmuerr<tol)
    self.assertTrue(worstsigerr<tol)
    self.assertTrue(worstfbarerr<tol)
    i=gg.testKdTree();
    self.assertEqual(i,1)
    #now try with the Neural Network gp
    
    kk=50
    neigh=np.zeros((kk),dtype=np.int32)
    neighshld=np.zeros((kk),dtype=np.int32)
    hh=np.zeros((2),dtype=float)
    hh[0]=1.23
    hh[1]=0.452
    gg.setLambda(0.0045)
    
    gg.setCovariogramType(gg.neuralNetwork)
    gg.setHyperParameters(hh)
    s=open("tests/data/gp_nn_solutions.sav")
    ss=s.readlines()
    s.close()

    worstmuerr=1.0
    worstsigerr=1.0
    worstfbarerr=1.0
    for z in range(len(ss)):
      sss=ss[z].split()
      for i in range(dd):
        test[i]=float(sss[i])
    
      for i in range(kk):
        neighshld[i]=int(sss[dd+i])
    
      mushld=float(sss[dd+kk])
      sigshld=float(sss[dd+kk+1])
  
      mu=gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      for i in range(kk):
        self.assertEqual(neigh[i],neighshld[i])
	
  
      err=(mu-mushld)
      if mushld != 0.0:
        err=err/mushld
	
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstmuerr:
        worstmuerr=err
  
  
  
      err=(sigma[0]-sigshld)
      if sigshld != 0.0:
        err=err/sigshld
	
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstsigerr:
        worstsigerr=err
    
      fbar=0.0
      for i in range(kk):
        fbar=fbar+fn[neigh[i]]
      fbar=fbar/float(kk)
      nn=float(sss[dd+kk+2])
      err=(fbar-nn)
      if nn != 0.0:
        err=err/nn
	
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstfbarerr:
        worstfbarerr=err
    
    print "\nThe errors for neural net covariogram\n"
    print "worst mu error ",worstmuerr
    print "worst sig2 error ",worstsigerr
    print "worst fbar error ",worstfbarerr
    
    self.assertTrue(worstmuerr<tol)
    self.assertTrue(worstsigerr<tol)
    self.assertTrue(worstfbarerr<tol)
  
  def testMinMax(self):
    pp=2000
    dd=10
    kk=50
    tol=1.0e-4
    data=np.zeros((pp,dd),dtype=float)
    fn=np.zeros((pp),dtype=float)
    test=np.zeros((dd),dtype=float)
    neigh=np.zeros((kk),dtype=np.int32)
    sigma=np.zeros((1),dtype=float)
    neighshld=np.zeros((kk),dtype=np.int32)
    hh=np.zeros((2),dtype=float);
    mins=np.zeros((dd),dtype=float)
    maxs=np.zeros((dd),dtype=float)
   
    hh[0]=0.555
    hh[1]=0.112

    f=open("tests/data/gp_exp_covar_data.sav")
    ff=f.readlines()
    f.close()
    fff=[]
    for i in range(len(ff)):
      fff.append(ff[i].split())
    for i in range(len(fff)):
     fn[i]=float(fff[i][10])
     for j in range(10):
       data[i][j]=float(fff[i][j])
 
    for i in range(pp):
      for j in range(dd):
        if (i==0) or (data[i][j]<mins[j]):
	  mins[j]=data[i][j]
	if (i==0) or (data[i][j]>maxs[j]):
	  maxs[j]=data[i][j]
  
    mins[2]=0.0
    maxs[2]=10.0
    gg=gp.GaussianProcessD(dd,pp,data,mins,maxs,fn)
    gg.setLambda(0.0045)
    gg.setCovariogramType(gg.neuralNetwork)
    gg.setHyperParameters(hh);
    #print "build the gp"
   
    s=open("tests/data/gp_minmax_solutions.sav")
    ss=s.readlines()
    s.close()

    worstmuerr=1.0
    worstsigerr=1.0
    worstfbarerr=1.0
    for z in range(len(ss)):
      sss=ss[z].split()
      for i in range(dd):
        test[i]=float(sss[i])
    
      for i in range(kk):
        neighshld[i]=int(sss[dd+i])
    
      mushld=float(sss[dd+kk])
      sigshld=float(sss[dd+kk+1])
  
      mu=gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      for i in range(kk):
        self.assertEqual(neigh[i],neighshld[i])
	
  
      err=(mu-mushld)
      if mushld != 0.0:
        err=err/mushld
	
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstmuerr:
        worstmuerr=err
  
  
  
      err=(sigma[0]-sigshld)
      if sigshld != 0.0:
        err=err/sigshld

      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstsigerr:
        worstsigerr=err
    
      fbar=0.0
      for i in range(kk):
        fbar=fbar+fn[neigh[i]]
      fbar=fbar/float(kk)
      nn=float(sss[dd+kk+2])
      err=(fbar-nn)
      if nn != 0.0:
        err=err/nn
      
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstfbarerr:
        worstfbarerr=err
    print "\nThe errors for Gaussian process using min-max normalization\n"
    print "worst mu error ",worstmuerr
    print "worst sig2 error ",worstsigerr
    print "worstf bar error ",worstfbarerr
    
    self.assertTrue(worstmuerr<tol)
    self.assertTrue(worstsigerr<tol)
    self.assertTrue(worstfbarerr<tol)
    i=gg.testKdTree();
    self.assertEqual(i,1)
 
  def testAddition(self):
    pp=1000
    dd=10
    kk=15
    tol=1.0e-4

    data=np.zeros((pp,dd),dtype=float)
    fn=np.zeros((pp),dtype=float)
    test=np.zeros((dd),dtype=float)
    neigh=np.zeros((kk),dtype=np.int32)
    sigma=np.zeros((1),dtype=float)
    neighshld=np.zeros((kk),dtype=np.int32)
    hh=np.zeros((1),dtype=float);
    
    hh[0]=5.0

    f=open("tests/data/gp_additive_test_root.sav")
    ff=f.readlines()
    f.close()
    fff=[]
    for i in range(len(ff)):
      fff.append(ff[i].split())

    for i in range(len(fff)):
     fn[i]=float(fff[i][10])
     for j in range(10):
       data[i][j]=float(fff[i][j])
  
    gg=gp.GaussianProcessD(dd,pp,data,fn)
    gg.setLambda(0.002)
    gg.setHyperParameters(hh)
    #print "build the gp"
    
    a=open("tests/data/gp_additive_test_data.sav")
    aa=a.readlines()
    a.close()
    for z in range(len(aa)):
      aaa=aa[z].split()
      for i in range(dd):
	test[i]=float(aaa[i])
      mushld=float(aaa[dd])
      gg.addPoint(test,mushld)

    s=open("tests/data/gp_additive_test_solutions.sav")
    ss=s.readlines()
    s.close()

    worstmuerr=1.0
    worstsigerr=1.0
    worstfbarerr=1.0
    for z in range(len(ss)):
      sss=ss[z].split()
      for i in range(dd):
        test[i]=float(sss[i])
    
      for i in range(kk):
        neighshld[i]=int(sss[dd+i])
    
      mushld=float(sss[dd+kk])
      sigshld=float(sss[dd+kk+1])
  
      mu=gg.interpolate(test,sigma,kk)
      gg.getNeighbors(neigh)
      
      for i in range(kk):
        self.assertEqual(neigh[i],neighshld[i])
	
  
      err=(mu-mushld)
      if mushld!=0:
        err=err/mushld
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstmuerr:
        worstmuerr=err
  
  
  
      err=(sigma[0]-sigshld)
      if sigshld!=0:
        err=err/sigshld
      if err<0.0:
        err=-1.0*err
      if z==0 or err>worstsigerr:
        worstsigerr=err
    
      
    print "\nThe errors for the test of adding points to the Gaussian process\n"
    print "worst mu error ",worstmuerr
    print "worst sig2 error ",worstsigerr
    
    
    self.assertTrue(worstmuerr<tol)
    self.assertTrue(worstsigerr<tol)
  
    i=gg.testKdTree();
    self.assertEqual(i,1)
  
  def testKdTree(self):
    pp=100
    dd=5
    data=np.zeros((pp,dd),dtype=float)
    fn=np.zeros((pp),dtype=float)
    
    f=open("tests/data/kd_test_data.sav")
    ff=f.readlines()
    f.close()
    for i in range(len(ff)):
      fff=ff[i].split()
      for j in range(dd):
        data[i][j]=float(fff[j])
      fn[i]=float(fff[dd])
    
    gg=gp.GaussianProcessD(dd,pp,data,fn)
    i=gg.testKdTree()
    self.assertEqual(i,1)

  def testBatch(self):
    pp=100
    dd=10
    tol=1.0e-3
    
    data=np.zeros((pp,dd),dtype=float)
    fn=np.zeros((pp),dtype=float)    
    
    f=open("tests/data/gp_exp_covar_data.sav","r");
    ff=f.readlines()
    f.close()
    for i in range(100):
      s=ff[i].split()
      for j in range(dd):
        data[i][j]=float(s[j])
      fn[i]=float(s[dd])
    
    gg=gp.GaussianProcessD(dd,pp,data,fn)
    gg.setLambda(0.0032)
    hh=np.zeros((1),dtype=float)
    hh[0]=2.0
    gg.setHyperParameters(hh)
    
    f=open("tests/data/gp_batch_solutions.sav","r")
    ff=f.readlines()
    f.close()
    
    ntest=len(ff)
    mushld=np.zeros((ntest),dtype=float)
    varshld=np.zeros((ntest),dtype=float)
    mu=np.zeros((ntest),dtype=float)
    var=np.zeros((ntest),dtype=float)
    
    queries=np.zeros((ntest,dd),dtype=float)
    
    for i in range(ntest):
      s=ff[i].split()
      for j in range(dd):
        queries[i][j]=float(s[j])
      mushld[i]=float(s[dd])
      varshld[i]=float(s[dd+1])
    
    gg.batchInterpolate(queries,mu,var,ntest)
    
    worstmuerr=-1.0
    worstvarerr=-1.0
    for i in range(ntest):
      err=mu[i]-mushld[i]
      if mushld[i] != 0.0:
        err=err/mushld[i]
      if err<0.0:
        err=-1.0*err
      if err>worstmuerr:
        worstmuerr=err
      
      err=var[i]-varshld[i]
      if varshld[i] !=0.0:
        err=err/varshld[i]
      if err<0.0:
        err=-1.0*err
      if err>worstvarerr:
        worstvarerr=err

    gg.batchInterpolate(queries,mu,ntest)   
    for i in range(ntest):
      err=mu[i]-mushld[i]
      if mushld[i] != 0.0:
        err=err/mushld[i]
      if err<0.0:
        err=-1.0*err
      if err>worstmuerr:
        worstmuerr=err

    self.assertTrue(worstmuerr<tol)
    self.assertTrue(worstvarerr<tol)
    
    print "\nThe errors for batch interpolation\n"
    print "worst mu error ",worstmuerr
    print "worst sig2 error ",worstvarerr

  def testSelf(self):
    pp=2000
    dd=10
    tol=1.0e-3
    kk=20
    
    data=np.zeros((pp,dd),dtype=float)
    fn=np.zeros((pp),dtype=float)    
    
    f=open("tests/data/gp_exp_covar_data.sav","r");
    ff=f.readlines()
    f.close()
    for i in range(pp):
      s=ff[i].split()
      for j in range(dd):
        data[i][j]=float(s[j])
      fn[i]=float(s[dd])
    
    gg=gp.GaussianProcessD(dd,pp,data,fn)
    gg.setKrigingParameter(30.0)
    gg.setLambda(0.00002)
    
    f=open("tests/data/gp_self_solutions.sav","r")
    ff=f.readlines()
    f.close()
    variance=np.zeros((1),dtype=float)
    neighshld=np.zeros((kk),dtype=np.int32)
    neigh=np.zeros((kk),dtype=np.int32)
    
    hh=np.zeros((1),dtype=float)
    hh[0]=20.0
    gg.setHyperParameters(hh)
    
    worstmuerr=-1.0
    worstsigerr=-1.0
    
    for i in range(pp):
      s=ff[i].split()
      mushld=float(s[0])
      sig2shld=float(s[1])
      for j in range(kk):
        neighshld[j]=np.int32(s[j+2])
      
      mu=gg.selfInterpolate(i,variance,kk)
      gg.getNeighbors(neigh)
      
      for j in range(kk):
        self.assertEqual(neigh[j],neighshld[j])
      
      err=mu-mushld
      if mushld != 0.0:
        err=err/mushld
      if err<0.0:
        err=err*(-1.0)
      if i==0 or err>worstmuerr:
        worstmuerr=err
      
      err=variance[0]-sig2shld
      if sig2shld != 0.0:
        err=err/sig2shld
      if err<0.0:
        err=err*(-1.0)
      if i==0 or err>worstsigerr:
        worstsigerr=err
    
    print "\nThe errors for self interpolation\n"
    print "worst mu error ",worstmuerr
    print "worst sig2 error ",worstsigerr
    self.assertTrue(worstmuerr<tol)
    self.assertTrue(worstsigerr<tol)

def suite():
  utilsTests.init()
  suites = []
  suites += unittest.makeSuite(GaussianProcessTestCase)
  return unittest.TestSuite(suites)

def run(shouldExit=False):
  utilsTests.run(suite(),shouldExit)

if __name__=="__main__":
  run(True)  
