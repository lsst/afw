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

namespace {

    // Given two vectors x and y, with some nans in y we want vectors x' and y' that correspond to the data
    // without the nans basic idea is that 'x' is the values, and 'y' is the ref (where nan checking happens)
    //    cullNan(x, y, x', y')
    void cullNan(std::vector<double> const &values, std::vector<double> const &refs,
                 std::vector<double> &culledValues, std::vector<double> &culledRefs
                ) {
        culledValues.reserve(refs.size());
        culledRefs.reserve(refs.size());
        for (std::vector<double>::const_iterator pVal = values.begin(), pRef = refs.begin();
             pRef != refs.end(); ++pRef, ++pVal) {
            if (!lsst::utils::isnan(*pRef)) {
                culledRefs.push_back(*pRef);
                culledValues.push_back(*pVal);
            }
        }
    }
}

/**
 * @brief Constructor for BackgroundMI
 *
 * Estimate the statistical properties of the Image in a grid of cells;  we'll later call
 * getImage() to interpolate those values, creating an image the same size as the original
 *
 * @note The old and deprecated API specified the interpolation style as part of the BackgroundControl
 * object passed to this ctor.  This is still supported, but the work isn't done until the getImage()
 * method is called
 */
template<typename ImageT>
BackgroundMI::BackgroundMI(ImageT const& img, ///< ImageT (or MaskedImage) whose properties we want
                             BackgroundControl const& bgCtrl ///< Control how the BackgroundMI is estimated
                            ) :
    Background(img, bgCtrl), _statsImage(PTR(image::MaskedImage<float>)())
{
    // =============================================================
    // Loop over the cells in the image, computing statistical properties
    // of each cell in turn and using them to set _statsImage
    _statsImage.reset(new image::MaskedImage<float>(_nxSample, _nySample));

    image::MaskedImage<float>::Image &im = *_statsImage->getImage();
    image::MaskedImage<float>::Variance &var = *_statsImage->getVariance();

    for (int iX = 0; iX < _nxSample; ++iX) {
        for (int iY = 0; iY < _nySample; ++iY) {
            ImageT subimg = ImageT(img, geom::Box2I(geom::Point2I(_xorig[iX], _yorig[iY]),
                                                    geom::Extent2I(_xsize[iX], _ysize[iY])), image::LOCAL);
            
            std::pair<double, double> res = makeStatistics(subimg, _bctrl.getStatisticsProperty() | ERRORS,
                                                           *_bctrl.getStatisticsControl()).getResult();
            im(iX, iY) = res.first;
            var(iX, iY) = res.second;
        }
    }
}

void BackgroundMI::_set_gridcolumns(Interpolate::Style const interpStyle,
                                  int const iX, std::vector<int> const& ypix) const
{
    image::MaskedImage<float>::Image &im = *_statsImage->getImage();

    _gridcolumns[iX].resize(_imgHeight);

    if (interpStyle == Interpolate::CONSTANT) {
        // A constant only makes sense when n[xy]Sample are both 1, but this should still work for other grid
        // sizes too.
        for (int iY = 0; iY < _imgHeight; ++iY) {
            int const iGridY = (_nySample*iY)/_imgHeight;
            _gridcolumns[iX][iY] = im(iX, iGridY);
        }
    } else {
        // Set _grid as a transitional measure
        std::vector<double> _grid(_nySample);
        std::copy(im.col_begin(iX), im.col_end(iX), _grid.begin());

        // remove nan from the grid values before computing columns
        // if we do it here (ie. in set_gridcolumns), it should
        // take care of all future occurrences, so we don't need to do this elsewhere
        std::vector<double> ycenTmp, gridTmp;
        cullNan(_ycen, _grid, ycenTmp, gridTmp);

        try {
            Interpolate intobj(ycenTmp, gridTmp, interpStyle);
            
            for (int iY = 0; iY < _imgHeight; ++iY) {
                _gridcolumns[iX][iY] = intobj.interpolate(ypix[iY]);
            }
        } catch(ex::Exception &e) {
            LSST_EXCEPT_ADD(e, "setting _gridcolumns");
            throw e;
        }
    }
}

/**
 * @brief Add a scalar to the Background (equivalent to adding a constant to the original image)
 */
void BackgroundMI::operator+=(float const delta ///< Value to add
                                  )
{
    *_statsImage += delta;
}

/**
 * @brief Subtract a scalar from the Background (equivalent to subtracting a constant from the original image)
 */
void BackgroundMI::operator-=(float const delta ///< Value to subtract
                                  )
{
    *_statsImage -= delta;
}

/**
 * @brief Method to retrieve the background level at a pixel coord.
 *
 * @warning This can be a very costly function to get a single pixel
 *          If you want an image, use the getImage() method.
 *
 * @return an estimated background at x,y (double)
 */
double BackgroundMI::getPixel(Interpolate::Style const interpStyle, ///< How to interpolate
                            int const x, ///< x-pixel coordinate (column)
                            int const y ///< y-pixel coordinate (row)
                           ) const
{
    (void)getImage<double>(interpStyle);        // setup the splines

    // build an interpobj along the row y and get the x'th value
    std::vector<double> bg_x(_nxSample);
    for (int iX = 0; iX < _nxSample; iX++) {
        bg_x[iX] = _gridcolumns[iX][y];
    }

    if (interpStyle != Interpolate::CONSTANT) {
        try {
            Interpolate intobj(_xcen, bg_x, interpStyle);
            return static_cast<double>(intobj.interpolate(x));
        } catch(ex::Exception &e) {
            LSST_EXCEPT_ADD(e, "in getPixel()");
            throw e;
        }
    } else {
        int const iGridX = (_nxSample * x) / _imgWidth;
        return static_cast<double>(_gridcolumns[iGridX][y]);
    }
    
}

template<typename PixelT>
PTR(image::Image<PixelT>) BackgroundMI::doGetImage(
        Interpolate::Style const interpStyle_,   // Style of the interpolation
        UndersampleStyle const undersampleStyle // Behaviour if there are too few points
                                                ) const
{
    int const nxSample = _bctrl.getNxSample();
    int const nySample = _bctrl.getNySample();
    Interpolate::Style interpStyle = interpStyle_; // not const -- may be modified if REDUCE_INTERP_ORDER
    /*
     * Check if the requested nx,ny are sufficient for the requested interpolation style,
     * making suitable adjustments
     */
    bool const isXundersampled = (nxSample < lookupMinInterpPoints(interpStyle));
    bool const isYundersampled = (nySample < lookupMinInterpPoints(interpStyle));

    switch (undersampleStyle) {
      case THROW_EXCEPTION:
        if (isXundersampled && isYundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterException,
                              "nxSample and nySample have too few points for requested interpolation style.");
        } else if (isXundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterException,
                              "nxSample has too few points for requested interpolation style.");
        } else if (isYundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterException,
                              "nySample has too few points for requested interpolation style.");
        }
        break;
      case REDUCE_INTERP_ORDER:
        if (isXundersampled || isYundersampled) {
            Interpolate::Style const xStyle = lookupMaxInterpStyle(nxSample);
            Interpolate::Style const yStyle = lookupMaxInterpStyle(nySample);
            interpStyle = (nxSample < nySample) ? xStyle : yStyle;
            _asUsedInterpStyle = interpStyle;
        }
        break;
      case INCREASE_NXNYSAMPLE:
        if (isXundersampled || isYundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterException,
                              "The BackgroundControl UndersampleStyle INCREASE_NXNYSAMPLE is not supported.");
        }
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          str(boost::format("The selected BackgroundControl "
                                            "UndersampleStyle %d is not defined.") % undersampleStyle));
    }

    /*********************************************************************************************************/
    // Check that an int's large enough to hold the number of pixels
    assert(_imgWidth*static_cast<double>(_imgHeight) < std::numeric_limits<int>::max());

    // =============================================================
    // --> We'll store nxSample fully-interpolated columns to spline the rows over
    // make a vector containing the y pixel coords for the column
    // --> We'll store _nxSample fully-interpolated columns to spline the rows over
    // make a vector containing the y pixel coords for the column
    std::vector<int> ypix(_imgHeight);
    for (int iY = 0; iY < _imgHeight; ++iY) {
        ypix[iY] = iY;
    }

    _gridcolumns.resize(_imgWidth);
    for (int iX = 0; iX < _nxSample; ++iX) {
        _set_gridcolumns(interpStyle, iX, ypix);
    }

    // create a shared_ptr to put the background image in and return to caller
    PTR(image::Image<PixelT>) bg = PTR(image::Image<PixelT>) (
        new typename image::Image<PixelT>(
            geom::Extent2I(_imgWidth, _imgHeight)
        )
    );

    // need a vector of all x pixel coords to spline over
    std::vector<int> xpix(bg->getWidth());
    for (int iX = 0; iX < bg->getWidth(); ++iX) { xpix[iX] = iX; }
    
    // go through row by row
    // - spline on the gridcolumns that were pre-computed by the constructor
    // - copy the values to an ImageT to return to the caller.
    for (int iY = 0; iY < bg->getHeight(); ++iY) {

        // build an interp object for this row
        std::vector<double> bg_x(nxSample);
        for (int iX = 0; iX < nxSample; iX++) {
            bg_x[iX] = static_cast<double>(_gridcolumns[iX][iY]);
        }
        
        if (interpStyle != Interpolate::CONSTANT) {
            try {
                Interpolate intobj(_xcen, bg_x, interpStyle);
                // fill the image with interpolated objects.
                int iX = 0;
                for (typename image::Image<PixelT>::x_iterator ptr = bg->row_begin(iY),
                         end = ptr + bg->getWidth(); ptr != end; ++ptr, ++iX) {
                    *ptr = static_cast<PixelT>(intobj.interpolate(xpix[iX]));
                }
            } catch(ex::Exception &e) {
                LSST_EXCEPT_ADD(e, "Interpolating in x");
                throw e;
            }
        } else {
            // fill the image with interpolated objects.
            int iX = 0;
            for (typename image::Image<PixelT>::x_iterator ptr = bg->row_begin(iY),
                     end = ptr + bg->getWidth(); ptr != end; ++ptr, ++iX) {
                int const iGridX = (nxSample * iX) / _imgWidth;
                *ptr = static_cast<PixelT>(_gridcolumns[iGridX][iY]);
            }
        }

    }

    return bg;
}

/*
 * Explicit instantiations
 *
 * \cond
 */
#define INSTANTIATE_BACKGROUND(TYPE)                                    \
    template BackgroundMI::BackgroundMI(image::Image<TYPE> const& img, \
                                          BackgroundControl const& bgCtrl); \
    template BackgroundMI::BackgroundMI(image::MaskedImage<TYPE> const& img, \
                                          BackgroundControl const& bgCtrl); \
    PTR(image::Image<TYPE>)                                             \
    BackgroundMI::_getImage(                                              \
        Interpolate::Style const interpStyle,                    /* Style of the interpolation */ \
        UndersampleStyle const undersampleStyle,                 /* Behaviour if there are too few points */ \
        TYPE                                                     /* disambiguate */    \
                         ) const                                        \
    {                                                                   \
        return BackgroundMI::doGetImage<TYPE>(interpStyle, undersampleStyle); \
    }

INSTANTIATE_BACKGROUND(double)
INSTANTIATE_BACKGROUND(float)
INSTANTIATE_BACKGROUND(int)

// \endcond
}}}
