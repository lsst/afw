#include <iostream>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <lsst/afw/math.h>

using namespace std;
namespace afwMath = lsst::afw::math;

int main() {
    unsigned int kernelCols = 3;
    unsigned int kernelRows = 2;
    unsigned int nPixels = kernelCols * kernelRows;
    
    // create linear combination kernel as a set of delta function basis kernels
    lsst::afw::math::KernelList<> kernelVec;
    int colCtr = (kernelCols - 1) / 2;
    int rowCtr = (kernelRows - 1) / 2;
    unsigned int ind = 0;
    for (unsigned int row = 0; row < kernelRows; ++row) {
        for (unsigned int col = 0; col < kernelCols; ++col) {
            int const x = col - colCtr;
            int const y = row - rowCtr;
            cout << boost::format("Delta function kernel %3d: x=%.1f, y=%.1f\n") % ind % x % y;
            lsst::afw::math::IntegerDeltaFunction2<lsst::afw::math::Kernel::PixelT> deltaFunc(x, y);
            lsst::afw::math::Kernel::PtrT kernelPtr(
#if 0
		        new lsst::afw::math::AnalyticKernel(deltaFunc, kernelCols, kernelRows)
#else
                new lsst::afw::math::DeltaFunctionKernel(x, y, kernelCols, kernelRows)
#endif
            );
            kernelVec.push_back(kernelPtr);
            ++ind;
        }
    }
    cout << endl;
    std::vector<double> kernelParams(nPixels); // initial kernel parameters
    lsst::afw::math::LinearCombinationKernel deltaFunctionKernelSet(kernelVec, kernelParams);
    
    // set various kernel parameters and print the results
    for (unsigned int ind = 0; ind < nPixels; ++ind) {
        kernelParams[ind] = 1.0;
        deltaFunctionKernelSet.setKernelParameters(kernelParams);

        cout << "DeltaFunctionKernelSet with kernelParams=";
        for (unsigned int ii = 0; ii < nPixels; ++ii) {
            cout << kernelParams[ii] << " ";
        }
        cout << endl << endl;
        lsst::afw::math::printKernel(deltaFunctionKernelSet);
    }
}
