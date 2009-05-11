// -*- LSST-C++ -*-
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

    assert(_bctrl.ictrl.getStyle() == math::NATURAL_SPLINE); // hard-coded for the time-being


    _n = _imgWidth*_imgHeight;
    
    if (_n == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Image contains no pixels");
    }

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
    for (int i_x = 0; i_x < _nxSample; ++i_x) {
        _xcen[i_x] = static_cast<int>((i_x + 0.5) * _subimgWidth);
        _xorig[i_x] = i_x * _subimgWidth;
    }
    for (int i_y = 0; i_y < _nySample; ++i_y) {
        _ycen[i_y] = static_cast<int>((i_y + 0.5) * _subimgHeight);
        _yorig[i_y] = i_y * _subimgHeight;
    }

    // make a vector containing the y pixel coords for the column
    vector<int> ypix(_imgHeight);
    for (int i_y = 0; i_y < _imgHeight; ++i_y) { ypix[i_y] = i_y; }


    // go to each sub-image and get it's stats.
    // -- do columns in the inner-loop and spline them as they complete
    for (int i_x = 0; i_x < _nxSample; ++i_x) {
        
        _grid[i_x].resize(_nySample);
        for (int i_y = 0; i_y < _nySample; ++i_y) {
            
            ImageT subimg = ImageT(img,
                                   image::BBox(image::PointI(_xorig[i_x], _yorig[i_y]), _subimgWidth, _subimgHeight));
            
            math::Statistics stats = math::makeStatistics(subimg, math::MEAN | math::MEANCLIP | math::MEDIAN |
                                                                   math::IQRANGE | math::STDEVCLIP, _bctrl.sctrl);
            
            _grid[i_x][i_y] = stats.getValue(math::MEANCLIP);
        }
        
        typename math::SplineInterpolate<int,double> intobj(_ycen, _grid[i_x]);
        _gridcolumns[i_x].resize(_imgHeight);
        for (int i_y = 0; i_y < _imgHeight; ++i_y) {
            _gridcolumns[i_x][i_y] = intobj.interpolate(ypix[i_y]);
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
    for(int i = 0; i < _nxSample; i++) { bg_x[i] = _gridcolumns[i][y];  }
    math::SplineInterpolate<int,double> intobj(_xcen, bg_x);
    return static_cast<double>(intobj.interpolate(x));
    
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
    for (int i_x = 0; i_x < bg->getWidth(); ++i_x) { xpix[i_x] = i_x; }
    
    // go through row by row
    // - spline on the gridcolumns that were pre-computed by the constructor
    // - copy the values to an ImageT to return to the caller.
    for (int i_y = 0; i_y < bg->getHeight(); ++i_y) {

        // build an interp object for this row
        vector<PixelT> bg_x(_nxSample);
        for(int i_x = 0; i_x < _nxSample; i_x++) { bg_x[i_x] = static_cast<PixelT>(_gridcolumns[i_x][i_y]); }
        math::SplineInterpolate<int,PixelT> intobj(_xcen, bg_x);

        // fill the image with interpolated objects.
        int i_x = 0;
        for (typename image::Image<PixelT>::x_iterator ptr = bg->row_begin(i_y), end = ptr + bg->getWidth();
             ptr != end; ++ptr, ++i_x) {
            *ptr = static_cast<PixelT>(intobj.interpolate(xpix[i_x]));
        }
    }
    
    return bg;
}



/**
 * @brief Explicit instantiations
 *
 */
#define INSTANTIATE_BACKGROUND(TYPE) \
    template math::Background::Background(image::Image<TYPE> const& img, math::BackgroundControl const& bgCtrl=BackgroundControl()); \
    template image::Image<TYPE>::Ptr math::Background::getImage<TYPE>() const;

INSTANTIATE_BACKGROUND(double);
INSTANTIATE_BACKGROUND(float);
INSTANTIATE_BACKGROUND(int);

