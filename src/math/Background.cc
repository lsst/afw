// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2015 LSST Corporation.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

/*
 * Background estimation class code
 */
#include <iostream>
#include <limits>
#include <cmath>
#include <vector>
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace ex = pex::exceptions;

namespace afw {
namespace math {

template <typename ImageT>
Background::Background(ImageT const& img, BackgroundControl const& bgCtrl)
        : lsst::daf::base::Citizen(typeid(this)),
          _imgBBox(img.getBBox()),
          _bctrl(new BackgroundControl(bgCtrl)),
          _asUsedInterpStyle(Interpolate::UNKNOWN),
          _asUsedUndersampleStyle(THROW_EXCEPTION),
          _xcen(0),
          _ycen(0),
          _xorig(0),
          _yorig(0),
          _xsize(0),
          _ysize(0) {
    if (_imgBBox.isEmpty()) {
        throw LSST_EXCEPT(ex::InvalidParameterError, "Image contains no pixels");
    }

    // Check that an int's large enough to hold the number of pixels
    if (_imgBBox.getWidth() * static_cast<double>(_imgBBox.getHeight()) > std::numeric_limits<int>::max()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OverflowError,
                          str(boost::format("Image %dx%d has more pixels than fit in an int (%d)") %
                              _imgBBox.getWidth() % _imgBBox.getHeight() % std::numeric_limits<int>::max()));
    }

    _setCenOrigSize(_imgBBox.getWidth(), _imgBBox.getHeight(), bgCtrl.getNxSample(), bgCtrl.getNySample());
}

Background::Background(geom::Box2I const imageBBox, int const nx, int const ny)
        : lsst::daf::base::Citizen(typeid(this)),
          _imgBBox(imageBBox),
          _bctrl(new BackgroundControl(nx, ny)),
          _asUsedInterpStyle(Interpolate::UNKNOWN),
          _asUsedUndersampleStyle(THROW_EXCEPTION),
          _xcen(0),
          _ycen(0),
          _xorig(0),
          _yorig(0),
          _xsize(0),
          _ysize(0) {
    if (_imgBBox.isEmpty()) {
        throw LSST_EXCEPT(ex::InvalidParameterError, "Image contains no pixels");
    }

    // Check that an int's large enough to hold the number of pixels
    if (_imgBBox.getWidth() * static_cast<double>(_imgBBox.getHeight()) > std::numeric_limits<int>::max()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OverflowError,
                          str(boost::format("Image %dx%d has more pixels than fit in an int (%d)") %
                              _imgBBox.getWidth() % _imgBBox.getHeight() % std::numeric_limits<int>::max()));
    }

    _setCenOrigSize(_imgBBox.getWidth(), _imgBBox.getHeight(), nx, ny);
}

void Background::_setCenOrigSize(int const width, int const height, int const nxSample, int const nySample) {
    _xcen.resize(nxSample);
    _ycen.resize(nySample);
    _xorig.resize(nxSample);
    _yorig.resize(nySample);
    _xsize.resize(nxSample), _ysize.resize(nySample);

    // Compute the centers and origins for the cells
    for (int iX = 0; iX < nxSample; ++iX) {
        const int endx = std::min(((iX + 1) * width + nxSample / 2) / nxSample, width);
        _xorig[iX] = (iX == 0) ? 0 : _xorig[iX - 1] + _xsize[iX - 1];
        _xsize[iX] = endx - _xorig[iX];
        _xcen[iX] = _xorig[iX] + (0.5 * _xsize[iX]) - 0.5;
    }

    for (int iY = 0; iY < nySample; ++iY) {
        const int endy = std::min(((iY + 1) * height + nySample / 2) / nySample, height);
        _yorig[iY] = (iY == 0) ? 0 : _yorig[iY - 1] + _ysize[iY - 1];
        _ysize[iY] = endy - _yorig[iY];
        _ycen[iY] = _yorig[iY] + (0.5 * _ysize[iY]) - 0.5;
    }
}

UndersampleStyle stringToUndersampleStyle(std::string const& style) {
    static std::map<std::string, UndersampleStyle> undersampleStrings;
    if (undersampleStrings.size() == 0) {
        undersampleStrings["THROW_EXCEPTION"] = THROW_EXCEPTION;
        undersampleStrings["REDUCE_INTERP_ORDER"] = REDUCE_INTERP_ORDER;
        undersampleStrings["INCREASE_NXNYSAMPLE"] = INCREASE_NXNYSAMPLE;
    }

    if (undersampleStrings.find(style) == undersampleStrings.end()) {
        throw LSST_EXCEPT(ex::InvalidParameterError, "Understample style not defined: " + style);
    }
    return undersampleStrings[style];
}
/// @cond
/*
 * Explicit instantiations
 *
 */
#define INSTANTIATE_BACKGROUND(TYPE)                                                                       \
    template Background::Background(image::Image<TYPE> const& img, BackgroundControl const& bgCtrl);       \
    template Background::Background(image::MaskedImage<TYPE> const& img, BackgroundControl const& bgCtrl); \
    template std::shared_ptr<image::Image<TYPE>> Background::getImage<TYPE>(Interpolate::Style const,      \
                                                                            UndersampleStyle const) const;

INSTANTIATE_BACKGROUND(float)

/// @endcond
}  // namespace math
}  // namespace afw
}  // namespace lsst
