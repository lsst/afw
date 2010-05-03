// -*- LSST-C++ -*-
/*!
 * Represent a Psf as a circularly symmetrical double Gaussian
 *
 * \file
 *
 * \ingroup afw
 */
#include <cmath>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/detail/dgPsf.h"
#include "lsst/afw/image/ImageUtils.h"

namespace lsst {
namespace afw {
namespace detection {

/************************************************************************************************************/
/**
 * Constructor for a dgPsf
 */
dgPsf::dgPsf(int width,                         ///< Number of columns in realisations of Psf
             int height,                        ///< Number of rows in realisations of Psf
             double sigma1,                     ///< Width of inner Gaussian
             double sigma2,                     ///< Width of outer Gaussian
             double b                   ///< Central amplitude of outer Gaussian (inner amplitude == 1)
            ) :
    KernelPsf(),
    _sigma1(sigma1), _sigma2(sigma2), _b(b) {
    if (b == 0.0 && sigma2 == 0.0) {
        _sigma2 = 1.0;                  // avoid 0/0 at centre of Psf
    }

    if (_sigma1 <= 0 || _sigma2 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("sigma may not be 0: %g, %g") % _sigma1 % _sigma2).str());
    }
    
    if (width > 0) {
        lsst::afw::math::DoubleGaussianFunction2<double> dg(_sigma1, _sigma2, _b);
        setKernel(lsst::afw::math::Kernel::Ptr(new lsst::afw::math::AnalyticKernel(width, height, dg)));
    }
}

namespace {
// Evaluate the Psf at (dx, dy) (relative to the centre), taking the central amplitude to be 1.0
double dgPsf::getValue(double const dx,            // Desired column (relative to centre of Psf)
                       double const dy             // Desired row (relative to centre of Psf)
                      ) const {
    double const r2 = dx*dx + dy*dy;
    double const psf1 = exp(-r2/(2*_sigma1*_sigma1));
    if (_b == 0.0) {
        return psf1;
    }
    
    double const psf2 = exp(-r2/(2*_sigma2*_sigma2));

    return (psf1 + _b*psf2)/(1 + _b);
}
}

/*
 * Return an Image of the the Psf at the point (x, y), setting the sum of all the Psf's pixels to 1.0
 *
 * The specified position is a floating point number, and the resulting image will
 * have a Psf with the correct fractional position, with the centre within pixel (width/2, height/2)
 * Specifically, fractional positions in [0, 0.5] will appear above/to the right of the center,
 * and fractional positions in (0.5, 1] will appear below/to the left (0.9999 is almost back at middle)
 */
lsst::afw::image::Image<Psf::Pixel>::Ptr dgPsf::getImage(double const x, ///< column posn in parent %image
                                                         double const y  ///< row posn in parent %image
                                                        ) const {
    Psf::Image::Ptr image(new Psf::Image(getKernel()->getWidth(), getKernel()->getHeight()));

    double const dx = lsst::afw::image::positionToIndex(x, true).second; // fractional part of position
    double const dy = lsst::afw::image::positionToIndex(y, true).second;

    int const xcen = static_cast<int>(getKernel()->getWidth()/2);
    int const ycen = static_cast<int>(getKernel()->getHeight()/2);

    double sum = 0;
    for (int iy = 0; iy != image->getHeight(); ++iy) {
        Psf::Image::x_iterator row = image->row_begin(iy);
        for (int ix = 0; ix != image->getWidth(); ++ix) {
            Psf::Pixel val = getValue(ix - dx - xcen, iy - dy - ycen);

            row[ix] = val;
            sum += val;
        }
    }

    *image /= sum;

    return image;                                                    
}

//
// We need to make an instance here so as to register it
//
// \cond
namespace {
    volatile bool isInstance =
        Psf::registerMe<dgPsf, boost::tuple<int, int, double, double, double> >("DoubleGaussian");
}

// \endcond
}}}
