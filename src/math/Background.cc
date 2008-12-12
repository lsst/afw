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
namespace interpolate = lsst::afw::math::interpolate;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

namespace {
    double const NaN = std::numeric_limits<double>::quiet_NaN();
}

/**
 * Constructor for Background
 *
 * Most of the actual work is done in this constructor; the results
 * are retrieved using \c get_value etc.
 */
template<typename ImageT>
math::Background<ImageT>::Background(ImageT const& img, BackgroundControl const& bgCtrl) : _img(img) { ///< ImageT (or MaskedImage) whose properties we want

    _n = img.getWidth()*img.getHeight();
    
    if (_n == 0) {
        throw lsst::pex::exceptions::InvalidParameter("Image contains no pixels");
    }

    _nxSample = bgCtrl.getNxSample();
    _nySample = bgCtrl.getNySample();
    cout << _nxSample << " " << _nySample << endl;
    _xcen.resize(_nxSample);
    _ycen.resize(_nySample);
    _xorig.resize(_nxSample);
    _yorig.resize(_nySample);
    //_grid(_nxSample, _nySample);
    _grid.resize(_nxSample);
    
    // Check that an int's large enough to hold the number of pixels
    assert(img.getWidth()*static_cast<double>(img.getHeight()) < std::numeric_limits<int>::max());


    // transfer the statistical control info to the sCtrl object.
    math::StatisticsControl sctrl;
    sctrl.setNumIter(bgCtrl.getNumIter());
    sctrl.setNumSigmaClip(bgCtrl.getNumSigmaClip());

    // go to each sub image and get the stats
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

    vector<int> ypix(img.getHeight());
    for (int i_y = 0; i_y < img.getHeight(); ++i_y) { ypix[i_y] = i_y; }
    
    for (int i_x = 0; i_x < _nySample; ++i_x) {
        _grid[i_x].resize(_nySample);

        for (int i_y = 0; i_y < _nySample; ++i_y) {
            
            ImageT subimg = ImageT(img,
                                   image::BBox(image::PointI(_xorig[i_x], _yorig[i_y]), _subimgWidth, _subimgHeight));
            
            math::Statistics<ImageT> stats = math::make_Statistics(subimg, math::MEAN | math::MEANCLIP | math::MEDIAN |
                                                                   math::IQRANGE | math::STDEVCLIP, sctrl);
            
            _grid[i_x][i_y] = stats.getValue(math::MEANCLIP);
            //cout << i_x << " " << i_y << " " << _grid[i_x][i_y] << endl;
            
        }
        
        typename interpolate::NaturalSpline<int,typename ImageT::Pixel> intobj = interpolate::init_NaturalSpline(_ycen, _grid[i_x]);
        _gridcolumns.push_back( intobj.interp(ypix) );

    }

}


template<typename ImageT>
typename ImageT::Pixel math::Background<ImageT>::getPixel(int const x, int const y) const {

    // build an interpobj along the row and get the x'th value
    vector<typename ImageT::Pixel> bg_x(_nxSample);
    for(int i = 0; i < _nxSample; i++) { bg_x[i] = _gridcolumns[i][y];  }
    interpolate::NaturalSpline<int,typename ImageT::Pixel> intobj = interpolate::init_NaturalSpline(_xcen, bg_x);
    typename ImageT::Pixel interp = intobj.interp(x);
    
    return interp;
}

template<typename ImageT>
ImageT math::Background<ImageT>::getFrame() const {
    
    typename ImageT::Ptr bg = typename ImageT::Ptr(new ImageT(_img, true));  // deep copy

    vector<int> xpix(bg->getWidth());
    for (int i_x = 0; i_x < bg->getWidth(); ++i_x) { xpix[i_x] = i_x; }

    for (int i_y = 0; i_y < bg->getHeight(); ++i_y) {

        // build an interp object for this row
        vector<typename ImageT::Pixel> bg_x(_nxSample);
        for(int i_x = 0; i_x < _nxSample; i_x++) { bg_x[i_x] = _gridcolumns[i_x][i_y]; }
        interpolate::NaturalSpline<int,typename ImageT::Pixel> intobj = interpolate::init_NaturalSpline(_xcen, bg_x);
        vector<typename ImageT::Pixel> interp = intobj.interp(xpix);
        
        int i_x = 0;
        for (typename ImageT::x_iterator ptr = bg->row_begin(i_y), end = ptr + bg->getWidth(); ptr != end; ++ptr) {
            *ptr = interp[i_x];
            ++i_x;
        }
    }
    
    return *bg;
}



/************************************************************************************************************/
//
// Explicit instantiations
//
template class math::Background<image::Image<double> >;
template class math::Background<image::Image<float> >;
//template class math::Background<image::Image<int> >;
