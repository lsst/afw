#include <stdexcept>
#include <sstream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <lsst/fw/FunctionLibrary.h>
#include <lsst/fw/Kernel.h>

using namespace std;

int main() {
    lsst::fw::Kernel<double>::Function2PtrType kfuncPtr(new lsst::fw::GaussianFunction2<double>(1.0, 2.0, 2.5));
    lsst::fw::AnalyticKernel<double> kernel(kfuncPtr, 5, 5);
    lsst::fw::Image<double> kImage(kernel.getImage());
    boost::shared_ptr<vw::ImageView<double> > kViewPtr(kImage.getIVwPtr());
    
    for (unsigned row = 0; row < kViewPtr->rows(); row++) {
        for (unsigned col = 0; col < kViewPtr->cols(); col++) {
            cout << "kImage(" << col << ", " << row << ") = " << (*kViewPtr)(col, row) << endl;
        }
    }
}
