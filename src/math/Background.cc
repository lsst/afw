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
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace ex = lsst::pex::exceptions;


/**
 * @brief Constructor for Background
 *
 * Various things are pre-computed by the constructor to make the interpolation faster.
 *
 * @note This is hard-coded to use bicubic spline for interpolation, attempt to use linear interpolate
 *       will cause an assertion failure.
 * @todo Implement user-settable intepolation style
 */
template<typename ImageT>
math::Background::Background(ImageT const& img, ///< ImageT (or MaskedImage) whose properties we want
                             BackgroundControl const& bgCtrl ///< Control how the Background is estimated
                            ) :
    _imgWidth(img.getWidth()), _imgHeight(img.getHeight()),
    _bctrl(bgCtrl) { 

    //assert(_bctrl.ictrl.getInterpStyle() == math::NATURAL_SPLINE); // hard-coded for the time-being

    _n = _imgWidth*_imgHeight;
    
    if (_n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Image contains no pixels");
    }

    _checkSampling();

    _nxSample = _bctrl.getNxSample();
    _nySample = _bctrl.getNySample();
    _xcen.resize(_nxSample);
    _ycen.resize(_nySample);
    _xorig.resize(_nxSample);
    _yorig.resize(_nySample);
    _grid.resize(_nxSample);
    _gridcolumns.resize(_nxSample);


    // Check that an int's large enough to hold the number of pixels
    assert(_imgWidth*static_cast<double>(_imgHeight) < std::numeric_limits<int>::max());

    // =============================================================
    // Bulk of the code here
    // --> go to each sub image and get the stats
    // --> We'll store _nxSample fully-interpolated columns to spline the rows over

    // Compute the centers and origins for the sub-images
    _subimgWidth = _imgWidth / _nxSample;
    _subimgHeight = _imgHeight / _nySample;
    for (int iX = 0; iX < _nxSample; ++iX) {
        _xcen[iX] = std::floor((iX + 0.5)*_subimgWidth);
        _xorig[iX] = iX * _subimgWidth;
    }
    for (int iY = 0; iY < _nySample; ++iY) {
        _ycen[iY] = std::floor((iY + 0.5)*_subimgHeight);
        _yorig[iY] = iY * _subimgHeight;
    }

    // make a vector containing the y pixel coords for the column
    vector<int> ypix(_imgHeight);
    for (int iY = 0; iY < _imgHeight; ++iY) { ypix[iY] = iY; }


    // go to each sub-image and get it's stats.
    // -- do columns in the inner-loop and spline them as they complete
    for (int iX = 0; iX < _nxSample; ++iX) {
        
        _grid[iX].resize(_nySample);
        for (int iY = 0; iY < _nySample; ++iY) {
            
            ImageT subimg =
                ImageT(img, image::BBox(image::Point2I(_xorig[iX], _yorig[iY]),
                                        _subimgWidth, _subimgHeight));
            
            math::Statistics stats =
                math::makeStatistics(subimg, _bctrl.getStatisticsProperty(),
                                     *(_bctrl.getStatisticsControl()));
            
            _grid[iX][iY] = stats.getValue(_bctrl.getStatisticsProperty());
        }

        _gridcolumns[iX].resize(_imgHeight);

        // there isn't actually any way to interpolate as a constant ... do that manually here
        if (_bctrl.getInterpStyle() != Interpolate::CONSTANT) {
            // this is the real interpolation
            typename math::Interpolate intobj(_ycen, _grid[iX], _bctrl.getInterpStyle());
            for (int iY = 0; iY < _imgHeight; ++iY) {
                _gridcolumns[iX][iY] = intobj.interpolate(ypix[iY]);
            }
        } else {
            // this is the constant interpolation
            // it should only be used sanely when nx,nySample are both 1,
            //  but this should still work for other grid sizes.
            for (int iY = 0; iY < _imgHeight; ++iY) {
                int const iGridY = (iY/_subimgHeight < _nySample) ? iY/_subimgHeight : _nySample;
                _gridcolumns[iX][iY] = _grid[iX][iGridY];
            }
        }

    }

}


/**
 * @brief Method to retrieve the background level at a pixel coord.
 *
 * @param x x-pixel coordinate (column)
 * @param y y-pixel coordinate (row)
 *
 * @warning This can be a very costly function to get a single pixel
 *          If you want an image, use the getImage() method.
 *
 * @return an estimated background at x,y (double)
 */
double math::Background::getPixel(int const x, int const y) const {

    // build an interpobj along the row y and get the x'th value
    vector<double> bg_x(_nxSample);
    for (int iX = 0; iX < _nxSample; iX++) {
        bg_x[iX] = _gridcolumns[iX][y];
    }

    if (_bctrl.getInterpStyle() != Interpolate::CONSTANT) {
        math::Interpolate intobj(_xcen, bg_x, _bctrl.getInterpStyle());
        return static_cast<double>(intobj.interpolate(x));
    } else {
        int const iGridX = (x/_subimgWidth < _nxSample) ? x/_subimgHeight : _nxSample;
        return static_cast<double>(_gridcolumns[iGridX][y]);
    }
    
}


/**
 * @brief Method to compute the background for entire image and return a background image
 *
 * @return A boost shared-pointer to an image containing the estimated background
 */
template<typename PixelT>
typename image::Image<PixelT>::Ptr math::Background::getImage() const {

    // create a shared_ptr to put the background image in and return to caller
    typename image::Image<PixelT>::Ptr bg =
        typename image::Image<PixelT>::Ptr(new typename image::Image<PixelT>(_imgWidth, _imgHeight));

    // need a vector of all x pixel coords to spline over
    vector<int> xpix(bg->getWidth());
    for (int iX = 0; iX < bg->getWidth(); ++iX) { xpix[iX] = iX; }
    
    // go through row by row
    // - spline on the gridcolumns that were pre-computed by the constructor
    // - copy the values to an ImageT to return to the caller.
    for (int iY = 0; iY < bg->getHeight(); ++iY) {

        // build an interp object for this row
        vector<double> bg_x(_nxSample);
        for (int iX = 0; iX < _nxSample; iX++) {
            bg_x[iX] = static_cast<double>(_gridcolumns[iX][iY]);
        }
        
        if (_bctrl.getInterpStyle() != Interpolate::CONSTANT) {
            math::Interpolate intobj(_xcen, bg_x, _bctrl.getInterpStyle());
            // fill the image with interpolated objects.
            int iX = 0;
            for (typename image::Image<PixelT>::x_iterator ptr = bg->row_begin(iY),
                     end = ptr + bg->getWidth(); ptr != end; ++ptr, ++iX) {
                *ptr = static_cast<PixelT>(intobj.interpolate(xpix[iX]));
            }
        } else {
            // fill the image with interpolated objects.
            int iX = 0;
            for (typename image::Image<PixelT>::x_iterator ptr = bg->row_begin(iY),
                     end = ptr + bg->getWidth(); ptr != end; ++ptr, ++iX) {
                int const iGridX = (iX/_subimgWidth < _nxSample) ? iX/_subimgWidth : _nxSample - 1;
                *ptr = static_cast<PixelT>(_gridcolumns[iGridX][iY]);
            }
        }

    }
    
    return bg;
}

/************************************************************************************************************/
/**
 * @brief Conversion function to switch a string to an UndersampleStyle
 */
math::UndersampleStyle math::stringToUndersampleStyle(std::string const style) {
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

/**
 * @brief Method to see if the requested nx,ny are sufficient for the requested interpolation style.
 *
 */
void math::Background::_checkSampling() {

    bool isXundersampled = (_bctrl.getNxSample() < lookupMinInterpPoints(_bctrl.getInterpStyle()));
    bool isYundersampled = (_bctrl.getNySample() < lookupMinInterpPoints(_bctrl.getInterpStyle()));

    if (_bctrl.getUndersampleStyle() == THROW_EXCEPTION) {
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
        
    } else if (_bctrl.getUndersampleStyle() == REDUCE_INTERP_ORDER) {
        if (isXundersampled || isYundersampled) {
            math::Interpolate::Style const xStyle = lookupMaxInterpStyle(_bctrl.getNxSample());
            math::Interpolate::Style const yStyle = lookupMaxInterpStyle(_bctrl.getNySample());
            math::Interpolate::Style const style = (_bctrl.getNxSample() < _bctrl.getNySample()) ?
                xStyle : yStyle;
            _bctrl.setInterpStyle(style);
        }
        
    } else if (_bctrl.getUndersampleStyle() == INCREASE_NXNYSAMPLE) {
        if (isXundersampled) {
            _bctrl.setNxSample(lookupMinInterpPoints(_bctrl.getInterpStyle()));
        }
        if (isYundersampled) {
            _bctrl.setNySample(lookupMinInterpPoints(_bctrl.getInterpStyle()));
        }
        
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "The selected BackgroundControl UndersampleStyle is not defined.");
    }
    
}

/*
 * Explicit instantiations
 *
 * \cond
 */
#define INSTANTIATE_BACKGROUND(TYPE)                                    \
    template math::Background::Background(image::Image<TYPE> const& img, \
                                          math::BackgroundControl const& bgCtrl); \
    template math::Background::Background(image::MaskedImage<TYPE> const& img, \
                                          math::BackgroundControl const& bgCtrl); \
    template image::Image<TYPE>::Ptr math::Background::getImage<TYPE>() const;

INSTANTIATE_BACKGROUND(double)
INSTANTIATE_BACKGROUND(float)
INSTANTIATE_BACKGROUND(int)

// \endcond
