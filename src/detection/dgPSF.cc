// -*- LSST-C++ -*-
/*!
 * Represent a PSF as a circularly symmetrical double Gaussian
 *
 * \file
 *
 * \ingroup algorithms
 */
#include <cmath>
#include "lsst/pex/exceptions.h"
#include "lsst/meas/algorithms/detail/dgPsf.h"
#include "lsst/afw/image/ImageUtils.h"

namespace lsst {
namespace meas {
namespace algorithms {

/************************************************************************************************************/
/**
 * Constructor for a dgPsf
 */
dgPsf::dgPsf(int width,                         ///< Number of columns in realisations of PSF
             int height,                        ///< Number of rows in realisations of PSF
             double sigma1,                     ///< Width of inner Gaussian
             double sigma2,                     ///< Width of outer Gaussian
             double b                   ///< Central amplitude of outer Gaussian (inner amplitude == 1)
            ) :
    PSF(width, height),
    _sigma1(sigma1), _sigma2(sigma2), _b(b) {
    if (b == 0.0 && sigma2 == 0.0) {
        _sigma2 = 1.0;                  // avoid 0/0 at centre of PSF
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

/// \brief Evaluate the PSF at (dx, dy) (relative to the centre), taking the central amplitude to be 1.0
double dgPsf::doGetValue(double const dx,            ///< Desired column (relative to centre of PSF)
                         double const dy,            ///< Desired row (relative to centre of PSF)
                         int,                        ///< Desired column position in image (think "CCD")
                         int                         ///< Desired row position in image (think "CCD")
                        ) const {
    double const r2 = dx*dx + dy*dy;
    double const psf1 = exp(-r2/(2*_sigma1*_sigma1));
    if (_b == 0.0) {
        return psf1;
    }
    
    double const psf2 = exp(-r2/(2*_sigma2*_sigma2));

    return (psf1 + _b*psf2)/(1 + _b);
}

/*
 * Return an Image of the the PSF at the point (x, y), setting the sum of all the PSF's pixels to 1.0
 *
 * The specified position is a floating point number, and the resulting image will
 * have a PSF with the correct fractional position, with the centre within pixel (width/2, height/2)
 * Specifically, fractional positions in [0, 0.5] will appear above/to the right of the center,
 * and fractional positions in (0.5, 1] will appear below/to the left (0.9999 is almost back at middle)
 */
lsst::afw::image::Image<PSF::Pixel>::Ptr dgPsf::getImage(double const x, ///< column posn in parent %image
                                                         double const y  ///< row posn in parent %image
                                                        ) const {
    PSF::Image::Ptr image(new PSF::Image(getWidth(), getHeight()));

    double const dx = lsst::afw::image::positionToIndex(x, true).second; // fractional part of position
    double const dy = lsst::afw::image::positionToIndex(y, true).second;

    int const xcen = static_cast<int>(getWidth()/2);
    int const ycen = static_cast<int>(getHeight()/2);

    double sum = 0;
    for (int iy = 0; iy != image->getHeight(); ++iy) {
        PSF::Image::x_iterator row = image->row_begin(iy);
        for (int ix = 0; ix != image->getWidth(); ++ix) {
            PSF::Pixel val = getValue(ix - dx - xcen, iy - dy - ycen);

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
        PSF::registerMe<dgPsf, boost::tuple<int, int, double, double, double> >("DoubleGaussian");
}

// \endcond
}}}
