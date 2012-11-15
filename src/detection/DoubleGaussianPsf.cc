// -*- LSST-C++ -*-
#include <cmath>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/DoubleGaussianPsf.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/image/ImageUtils.h"

namespace afwMath = lsst::afw::math;

namespace lsst {
namespace afw {
namespace detection {

DoubleGaussianPsf::DoubleGaussianPsf(int width, int height, double sigma1, double sigma2, double b) :
    KernelPsf(), _sigma1(sigma1), _sigma2(sigma2), _b(b)
{
    if (b == 0.0 && sigma2 == 0.0) {
        sigma2 = 1.0;                  // avoid 0/0 at centre of Psf
    }

    if (sigma1 <= 0 || sigma2 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("sigma may not be 0: %g, %g") % sigma1 % sigma2).str());
    }
    
    if (width > 0) {
        afwMath::DoubleGaussianFunction2<double> dg(sigma1, sigma2, b);
        setKernel(afwMath::Kernel::Ptr(new afwMath::AnalyticKernel(width, height, dg)));
    }
}

namespace {

// We need to make an instance here so as to register it
volatile bool isInstance =
    Psf::registerMe<DoubleGaussianPsf, boost::tuple<int, int, double, double, double> >("DoubleGaussian");

} // anonymous

}}} // namespace lsst::afw::detection

