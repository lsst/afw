// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/**
 * @file BackgroundBase.cc
 * @ingroup afw
 * @brief BackgroundBase estimation class code
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
 * @brief Constructor for BackgroundBase
 *
 * Estimate the statistical properties of the Image in a grid of cells;  we'll later call
 * getImage() to interpolate those values, creating an image the same size as the original
 *
 * @note The old and deprecated API specified the interpolation style as part of the BackgroundControl
 * object passed to this ctor.  This is still supported, but the work isn't done until the getImage()
 * method is called
 */
template<typename ImageT>
BackgroundBase::BackgroundBase(ImageT const& img, ///< ImageT (or MaskedImage) whose properties we want
                             BackgroundControl const& bgCtrl ///< Control how the BackgroundBase is estimated
                            ) :
    _imgWidth(img.getWidth()), _imgHeight(img.getHeight()),
    _nxSample(bgCtrl.getNxSample()), _nySample(bgCtrl.getNySample()),
    _bctrl(bgCtrl),
    _asUsedInterpStyle(Interpolate::UNKNOWN),
    _xcen(_nxSample),  _ycen(_nySample),
    _xorig(_nxSample), _yorig(_nySample),
    _xsize(_nxSample), _ysize(_nySample)
{
    if (_imgWidth*_imgHeight == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Image contains no pixels");
    }

    // Check that an int's large enough to hold the number of pixels
    assert(_imgWidth*static_cast<double>(_imgHeight) < std::numeric_limits<int>::max());

    // Compute the centers and origins for the cells
    for (int iX = 0; iX < _nxSample; ++iX) {
        const int endx = std::min(((iX+1)*_imgWidth + _nxSample/2) / _nxSample, _imgWidth);
        _xorig[iX] = (iX == 0) ? 0 : _xorig[iX-1] + _xsize[iX-1];
        _xsize[iX] = endx - _xorig[iX];
        _xcen [iX] = _xorig[iX] + (0.5 * _xsize[iX]) - 0.5;
    }

    for (int iY = 0; iY < _nySample; ++iY) {
        const int endy = std::min(((iY+1)*_imgHeight + _nySample/2) / _nySample, _imgHeight);
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
        throw LSST_EXCEPT(ex::InvalidParameterException, "Understample style not defined: "+style);
    }
    return undersampleStrings[style];
}
/*
 * Explicit instantiations
 *
 * \cond
 */
#define INSTANTIATE_BACKGROUND(TYPE)                                    \
    template BackgroundBase::BackgroundBase(image::Image<TYPE> const& img,  \
                                        BackgroundControl const& bgCtrl); \
    template BackgroundBase::BackgroundBase(image::MaskedImage<TYPE> const& img, \
                                          BackgroundControl const& bgCtrl); \
    template PTR(image::Image<TYPE>) BackgroundBase::getImage<TYPE>(Interpolate::Style const, \
                                                                    UndersampleStyle const) const;


INSTANTIATE_BACKGROUND(double)
INSTANTIATE_BACKGROUND(float)
INSTANTIATE_BACKGROUND(int)

// \endcond
}}}

