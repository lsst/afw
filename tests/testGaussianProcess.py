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
    tol=1.0e-4

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
  
    gg=gp.GaussianProcessDD(dd,pp,data,fn)
    gg.setLambda(0.001)
    gg.setHyperParameters(hh);
    #print "build the gp"

    s=open("tests/data/gp_exp_covar_solutions.sav")
    ss=s.readlines()
    s.close()

    worstmuerr=-1.0
    worstsigerr=-1.0
    worstfbarerr=-1.0
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
	
  
      err=(mu-mushld)/mushld
      if err<0.0:
        err=-1.0*err
      if err>worstmuerr:
        worstmuerr=err
  
  
  
      err=(sigma-sigshld)/sigshld
      if err<0.0:
        err=-1.0*err
      if err>worstsigerr:
        worstsigerr=err
    
      fbar=0.0
      for i in range(kk):
        fbar=fbar+fn[neigh[i]]
      fbar=fbar/float(kk)
      nn=float(sss[dd+kk+2])
      err=(fbar-nn)/nn
      if err<0.0:
        err=-1.0*err
      if err>worstfbarerr:
        worstfbarerr=err

    print "worstmuerr ",worstmuerr
    print "worstsigerr ",worstsigerr
    print "worstfbarerr ",worstfbarerr
    
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
  
    gg=gp.GaussianProcessDD(dd,pp,data,fn)
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

    worstmuerr=-1.0
    worstsigerr=-1.0
    worstfbarerr=-1.0
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
	
  
      err=(mu-mushld)/mushld
      if err<0.0:
        err=-1.0*err
      if err>worstmuerr:
        worstmuerr=err
  
  
  
      err=(sigma-sigshld)/sigshld
      if err<0.0:
        err=-1.0*err
      if err>worstsigerr:
        worstsigerr=err
    
      

    print "worstmuerr ",worstmuerr
    print "worstsigerr ",worstsigerr
    
    
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
    
    gg=gp.GaussianProcessDD(dd,pp,data,fn)
    i=gg.testKdTree()
    self.assertEqual(i,1)

    



def suite():
  utilsTests.init()
  suites = []
  suites += unittest.makeSuite(GaussianProcessTestCase)
  return unittest.TestSuite(suites)

def run(shouldExit=False):
  utilsTests.run(suite(),shouldExit)

if __name__=="__main__":
  run(True)  
