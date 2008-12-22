/**
 * \file
 * \brief Support statistical operations on images
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

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}

/**
 * Constructor for Background
 *
 * Various things are pre-computed by the constructor to make the interpolation faster.
 */
template<typename ImageT>
math::Background<ImageT>::Background(ImageT const& img, ///< ImageT (or MaskedImage) whose properties we want
                                     BackgroundControl const& bgCtrl) : _img(img), _bctrl(bgCtrl) { 

    _n = _img.getWidth()*_img.getHeight();
    
    if (_n == 0) {
        throw lsst::pex::exceptions::InvalidParameter("Image contains no pixels");
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
    assert(img.getWidth()*static_cast<double>(img.getHeight()) < std::numeric_limits<int>::max());

    // =============================================================
    // Bulk of the code here
    // --> go to each sub image and get the stats
    // --> We'll store _nxSample fully-interpolated columns to spline the rows over

    // Compute the centers and origins for the sub-images
    _subimgWidth = img.getWidth() / _nxSample;
    _subimgHeight = img.getHeight() / _nySample;
    for (int i_x = 0; i_x < _nxSample; ++i_x) {
        _xcen[i_x] = static_cast<int>((i_x + 0.5) * _subimgWidth);
        _xorig[i_x] = i_x * _subimgWidth;
    }
    for (int i_y = 0; i_y < _nySample; ++i_y) {
        _ycen[i_y] = static_cast<int>((i_y + 0.5) * _subimgHeight);
        _yorig[i_y] = i_y * _subimgHeight;
    }

    // make a vector containing the y pixel coords for the column
    vector<int> ypix(img.getHeight());
    for (int i_y = 0; i_y < img.getHeight(); ++i_y) { ypix[i_y] = i_y; }


    // go to each sub-image and get it's stats.
    // -- do columns in the inner-loop and spline them as they complete
    for (int i_x = 0; i_x < _nxSample; ++i_x) {
        
        _grid[i_x].resize(_nySample);
        for (int i_y = 0; i_y < _nySample; ++i_y) {
            
            ImageT subimg = ImageT(img,
                                   image::BBox(image::PointI(_xorig[i_x], _yorig[i_y]), _subimgWidth, _subimgHeight));
            
            math::Statistics<ImageT> stats = math::make_Statistics(subimg, math::MEAN | math::MEANCLIP | math::MEDIAN |
                                                                   math::IQRANGE | math::STDEVCLIP, _bctrl.sctrl);
            
            _grid[i_x][i_y] = static_cast<typename ImageT::Pixel>(stats.getValue(math::MEANCLIP));
        }
        
        typename math::LinearInterpolate<int,typename ImageT::Pixel> intobj(_ycen, _grid[i_x]);
        _gridcolumns[i_x].resize(img.getHeight());
        for (int i_y = 0; i_y < img.getHeight(); ++i_y) {
            _gridcolumns[i_x][i_y] = intobj.interpolate(ypix[i_y]);
        }

    }

}

/**
 * \brief Method to retrieve the background level at a pixel coord.
 */
template<typename ImageT>
typename ImageT::Pixel math::Background<ImageT>::getPixel(int const x, int const y) const {

    // build an interpobj along the row and get the x'th value
    vector<typename ImageT::Pixel> bg_x(_nxSample);
    for(int i = 0; i < _nxSample; i++) { bg_x[i] = _gridcolumns[i][y];  }
    math::LinearInterpolate<int,typename ImageT::Pixel> intobj(_xcen, bg_x);
    return intobj.interpolate(x);
    
}


/**
 * \brief Method to compute the background for entire image and return a background image
 */
template<typename ImageT>
typename ImageT::Ptr math::Background<ImageT>::getFrame() const {
    
    typename ImageT::Ptr bg = typename ImageT::Ptr(new ImageT(_img, true));  // deep copy

    // need a vector of all x pixel coords to spline over
    vector<int> xpix(bg->getWidth());
    for (int i_x = 0; i_x < bg->getWidth(); ++i_x) { xpix[i_x] = i_x; }
    
    // go through row by row
    // - spline on the gridcolumns that were pre-computed by the constructor
    // - copy the values to an ImageT to return to the caller.
    for (int i_y = 0; i_y < bg->getHeight(); ++i_y) {

        // build an interp object for this row
        vector<typename ImageT::Pixel> bg_x(_nxSample);
        for(int i_x = 0; i_x < _nxSample; i_x++) { bg_x[i_x] = _gridcolumns[i_x][i_y]; }
        math::LinearInterpolate<int,typename ImageT::Pixel> intobj(_xcen, bg_x);

        int i_x = 0;
        for (typename ImageT::x_iterator ptr = bg->row_begin(i_y), end = ptr + bg->getWidth();
             ptr != end; ++ptr, ++i_x) {
            *ptr = intobj.interpolate(xpix[i_x]);
        }
    }
    
    return bg;
}



/************************************************************************************************************/
//
// Explicit instantiations
//
template class math::Background<image::Image<double> >;
template class math::Background<image::Image<float> >;
template class math::Background<image::Image<int> >;
