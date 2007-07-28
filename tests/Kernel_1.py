import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.fw.Core.tests as tests
import lsst.fw.Core.fwLib as fw
import lsst.mwi.tests as mtests
import lsst.mwi.utils as mwiu

try:
    type(verbose)
except NameError:
    verbose = 0
    mwiu.Trace_setVerbosity("fw.kernel", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class FixedKernelTestCase(unittest.TestCase):
    """A test case for FixedKernel"""
    def setUp(self):
        sigmaX = 2.0;
        sigmaY = 2.5;
        kernelCols = 5;
        kernelRows = 7;

        g = fw.GaussianFunction2D(sigmaX, sigmaY)
        kfuncPtr =  fw.Function2PtrTypeD(g); g.this.disown() # Only the shared pointer owns g
        
        self.analyticKernel = fw.AnalyticKernelD(kfuncPtr, kernelCols, kernelRows)

    def tearDown(self):
        del self.analyticKernel

    def testAnalyticImage(self):
        """Get an image from an AnalyticKernel"""

        image = self.analyticKernel.getImage()
        image *= 47.3           	# denormalize by some arbitrary factor

        if False:
            print "Gaussian kernel with sigmaX=%.1f, sigmaY=%.1f" % \
                  self.analyticKernel.getKernelFunction().getParameters()
            
            fw.printKernelD(self.analyticKernel);

    def testAnalyticImage2(self):
        """Compute an image from an AnalyticKernel"""

        image = self.analyticKernel.getImage()
        self.analyticKernel.computeImage(image)
        image *= 47.3           	# denormalize by some arbitrary factor

        if False:
            print "Gaussian kernel with sigmaX=%.1f, sigmaY=%.1f" % \
                  self.analyticKernel.getKernelFunction().getParameters()
            
            fw.printKernelD(self.analyticKernel);

    def testFixedKernel(self):
        """Make a fixedKernel from an image"""

        image = fw.ImageD(5,7)
        image.setVal(0); image.setVal(image.getCols()//2, image.getRows()//2, 1.0)

        fixedKernel = fw.FixedKernelD(image);

        if False:
            print "Fixed kernel"
            
            fw.printKernelD(fixedKernel);

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    mtests.init()

    suites = []
    suites += unittest.makeSuite(FixedKernelTestCase)
    suites += unittest.makeSuite(mtests.MemoryTestCase)

    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())
