import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy

import lsst.fw.Core.fwLib as fw
import lsst.mwi.tests as tests
import lsst.mwi.utils as mwiu

try:
    type(verbose)
except NameError:
    verbose = 0
    mwiu.Trace_setVerbosity("fw.kernel", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# convenience functions
def arrayFromImage(im, dtype=float):
    """Return a numpy array representation of an image.
    
    A temporary hack until image offers a better way to do this 
    """
    arr = numpy.zeros([im.getCols(), im.getRows()], dtype=dtype)
    for row in range(im.getRows()):
        for col in range(im.getCols()):
            arr[col, row] = im.getPtr(col, row)
    return arr

class KernelTestCase(unittest.TestCase):
    """A test case for Kernels"""
    def testFixedKernel(self):
        """Test FixedKernel using a ramp function
        """
        kCols = 5
        kRows = 6
        
        inArr = numpy.arange(kCols * kRows, dtype=float)
        inArr.shape = [kCols, kRows]

        inImage = fw.ImageD(kCols, kRows)
        for row in range(inImage.getRows()):
            for col in range(inImage.getCols()):
                inImage.set(col, row, inArr[col, row])
        
        fixedKernel = fw.FixedKernelF(inImage);
        outImage = fixedKernel.computeNewImage(0.0, 0.0, False)[0]
        outArr = arrayFromImage(outImage)
        if not numpy.allclose(inArr, outArr):
            self.fail("%s = %s != %s (not normalized)" % \
                (k.__class__.__name__, inArr, outArr))
        
        normInArr = inArr / inArr.sum()
        normOutImage = fixedKernel.computeNewImage(0.0, 0.0, True)[0]
        normOutArr = arrayFromImage(normOutImage)
        if not numpy.allclose(normOutArr, normInArr):
            self.fail("%s = %s != %s (normalized)" % \
                (k.__class__.__name__, normInArr, normOutArr))

    def testGaussianKernel(self):
        """Test AnalyticKernel using a Gaussian function
        """
        kCols = 5
        kRows = 8

        f = fw.GaussianFunction2F(1.0, 1.0)
        fPtr =  fw.Function2PtrTypeF(f)
        f.this.disown() # Only the shared pointer now owns f
        k = fw.AnalyticKernelF(fPtr, kCols, kRows)
        fArr = numpy.zeros(shape=[k.getCols(), k.getRows()], dtype=float)
        for xsigma in (0.1, 1.0, 3.0):
            for ysigma in (0.1, 1.0, 3.0):
                f.setParameters((xsigma, ysigma))
                # compute array of function values and normalize
                for row in range(k.getRows()):
                    y = row - k.getCtrRow()
                    for col in range(k.getCols()):
                        x = col - k.getCtrCol()
                        fArr[col, row] = f(x, y)
                fArr /= fArr.sum()
                
                k.setKernelParameters((xsigma, ysigma))
                kImage = k.computeNewImage(0.0, 0.0, True)[0]
                kArr = arrayFromImage(kImage)
                if not numpy.allclose(fArr, kArr):
                    self.fail("%s = %s != %s for xsigma=%s, ysigma=%s" % \
                        (k.__class__.__name__, kArr, fArr, xsigma, ysigma))
        
    def testLinearCombinationKernel(self):
        """Test LinearCombinationKernel using a set of delta basis functions
        """
        kCols = 5
        kRows = 8
        
        # create list of kernels
        kPtrList = []
        ctrCol = (kCols - 1) // 2
        ctrRow = (kRows - 1) // 2
        for row in range(kRows):
            y = float(row - ctrRow)
            for col in range(kCols):
                x = float(col - ctrCol)
                f = fw.IntegerDeltaFunction2F(x, y)
                fPtr = fw.Function2PtrTypeF(f)
                f.this.disown() # only the shared pointer now owns f
                basisKernel = fw.AnalyticKernelF(fPtr, kCols, kRows)
                kPtr = fw.KernelPtrTypeF(basisKernel)
                basisKernel.this.disown() # only the shared pointer now owns basisKernel
                kPtrList.append(kPtr)
        
        kParams = numpy.zeros(len(kPtrList), dtype=float)
        
        # this doesn't work -- can't match a constructor --
        # so the python interface needs some work
        #k = fw.LinearCombinationKernelF(tuple(kPtrList), tuple(kParams))
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(KernelTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())
