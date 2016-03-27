// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file Background.cc
 * @ingroup afw
 * @brief Background estimation class code
 * @author Steve Bickerton
 * @date Jan 26, 2009
 */
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>
#include "lsst/utils/ieee.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace ex = pex::exceptions;

namespace afw {
namespace math {

/**
 * @brief Constructor for Background
 *
 * Estimate the statistical properties of the Image in a grid of cells;  we'll later call
 * getImage() to interpolate those values, creating an image the same size as the original
 *
 * @note The old and deprecated API specified the interpolation style as part of the BackgroundControl
 * object passed to this ctor.  This is still supported, but the work isn't done until the getImage()
 * method is called
 */
template<typename ImageT>
Background::Background(ImageT const& img,              ///< ImageT (or MaskedImage) whose properties we want
                       BackgroundControl const& bgCtrl ///< Control how the Background is estimated
                      ) :
    lsst::daf::base::Citizen(typeid(this)),
    _imgBBox(img.getBBox()),
    _bctrl(new BackgroundControl(bgCtrl)),
    _asUsedInterpStyle(Interpolate::UNKNOWN),
    _asUsedUndersampleStyle(THROW_EXCEPTION),
    _xcen(0),  _ycen(0), _xorig(0), _yorig(0), _xsize(0), _ysize(0)
{
    if (_imgBBox.isEmpty()) {
        throw LSST_EXCEPT(ex::InvalidParameterError, "Image contains no pixels");
    }

    // Check that an int's large enough to hold the number of pixels
    if (_imgBBox.getWidth()*static_cast<double>(_imgBBox.getHeight()) > std::numeric_limits<int>::max()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OverflowError,
                          str(boost::format("Image %dx%d has more pixels than fit in an int (%d)")
                              % _imgBBox.getWidth() % _imgBBox.getHeight() % std::numeric_limits<int>::max()));
    }

    _setCenOrigSize(_imgBBox.getWidth(), _imgBBox.getHeight(), bgCtrl.getNxSample(), bgCtrl.getNySample());
}

/************************************************************************************************************/
/**
 * Create a Background without any values in it
 *
 * \note This ctor is mostly used to create a Background given its sample values, and that (in turn)
 * is mostly used to implement persistence.
 */
Background::Background(geom::Box2I const imageBBox, ///< Bounding box for image to be created by getImage()
                       int const nx,                ///< Number of samples in x-direction
                       int const ny                 ///< Number of samples in y-direction
                      ) :
    lsst::daf::base::Citizen(typeid(this)),
    _imgBBox(imageBBox),
    _bctrl(new BackgroundControl(nx, ny)),
    _asUsedInterpStyle(Interpolate::UNKNOWN),
    _asUsedUndersampleStyle(THROW_EXCEPTION),
    _xcen(0),  _ycen(0), _xorig(0), _yorig(0), _xsize(0), _ysize(0)
{
    if (_imgBBox.isEmpty()) {
        throw LSST_EXCEPT(ex::InvalidParameterError, "Image contains no pixels");
    }

    // Check that an int's large enough to hold the number of pixels
    if (_imgBBox.getWidth()*static_cast<double>(_imgBBox.getHeight()) > std::numeric_limits<int>::max()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OverflowError,
                          str(boost::format("Image %dx%d has more pixels than fit in an int (%d)")
                              % _imgBBox.getWidth() % _imgBBox.getHeight() % std::numeric_limits<int>::max()));
    }

    _setCenOrigSize(_imgBBox.getWidth(), _imgBBox.getHeight(), nx, ny);
}

/************************************************************************************************************/
/**
 * Compute the centers, origins, and sizes of the patches used to compute image statistics
 * when estimating the Background
 */
void
Background::_setCenOrigSize(int const width, int const height,
                            int const nxSample, int const nySample)
{
    _xcen.resize( nxSample);  _ycen.resize(nySample);
    _xorig.resize(nxSample); _yorig.resize(nySample);
    _xsize.resize(nxSample), _ysize.resize(nySample);

    // Compute the centers and origins for the cells
    for (int iX = 0; iX < nxSample; ++iX) {
        const int endx = std::min(((iX+1)*width + nxSample/2)/nxSample, width);
        _xorig[iX] = (iX == 0) ? 0 : _xorig[iX-1] + _xsize[iX-1];
        _xsize[iX] = endx - _xorig[iX];
        _xcen [iX] = _xorig[iX] + (0.5 * _xsize[iX]) - 0.5;
    }

    for (int iY = 0; iY < nySample; ++iY) {
        const int endy = std::min(((iY+1)*height + nySample/2)/nySample, height);
        _yorig[iY] = (iY == 0) ? 0 : _yorig[iY-1] + _ysize[iY-1];
        _ysize[iY] = endy - _yorig[iY];
        _ycen [iY] = _yorig[iY] + (0.5 * _ysize[iY]) - 0.5;
    }
}

/************************************************************************************************************/
/**
 * @brief Conversion function to switch a string to an UndersampleStyle
 */
UndersampleStyle stringToUndersampleStyle(std::string const &style) {
    static std::map<std::string, UndersampleStyle> undersampleStrings;
    if (undersampleStrings.size() == 0) {
        undersampleStrings["THROW_EXCEPTION"]     = THROW_EXCEPTION;
        undersampleStrings["REDUCE_INTERP_ORDER"] = REDUCE_INTERP_ORDER;
        undersampleStrings["INCREASE_NXNYSAMPLE"] = INCREASE_NXNYSAMPLE;
    }

    if (undersampleStrings.find(style) == undersampleStrings.end()) {
        throw LSST_EXCEPT(ex::InvalidParameterError, "Understample style not defined: "+style);
    }
    return undersampleStrings[style];
}
/// \cond
/*
 * Explicit instantiations
 *
 */
#define INSTANTIATE_BACKGROUND(TYPE)                                    \
    template Background::Background(image::Image<TYPE> const& img,  \
                                        BackgroundControl const& bgCtrl); \
    template Background::Background(image::MaskedImage<TYPE> const& img, \
                                          BackgroundControl const& bgCtrl); \
    template PTR(image::Image<TYPE>) Background::getImage<TYPE>(Interpolate::Style const, \
                                                                    UndersampleStyle const) const;


INSTANTIATE_BACKGROUND(float)

/// \endcond
}}} // lsst::afw::math
