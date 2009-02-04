// -*- LSST-C++ -*-
#if !defined(LSST_AFW_MATH_BACKGROUND_H)
#define LSST_AFW_MATH_BACKGROUND_H
/**
 * @file Background.h
 * @brief Use bi-cubic interpolation to estimate image background
 * @ingroup afw
 * @author Steve Bickerton
 */

#include "boost/shared_ptr.hpp"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/Interpolate.h"


namespace lsst { namespace afw { namespace math {

/**
 * @class BackgroundControl
 * @brief Pass parameters to a Background object
 */
class BackgroundControl {
public:
    BackgroundControl(Style const style=math::NATURAL_SPLINE,  ///< Style of the interpolation (not yet implemented)
                      int const nxSample=10,                   ///< Num. grid samples in x
                      int const nySample=10)                   ///< Num. grid samples in y
        : _nxSample(nxSample), _nySample(nySample) {
        assert(nxSample > 0);
        assert(nySample > 0);
        sctrl = StatisticsControl();
        ictrl = math::InterpControl(style);
    }
    ~BackgroundControl() {}
    void setNxSample (int nxSample) { assert(nxSample > 0); _nxSample = nxSample; }
    void setNySample (int nySample) { assert(nySample > 0); _nySample = nySample; }
    int getNxSample() const { return _nxSample; }
    int getNySample() const { return _nySample; }
    StatisticsControl sctrl;
    math::InterpControl ictrl;
private:
    int _nxSample;                      // number of grid squares to divide image into to sample in x
    int _nySample;                      // number of grid squares to divide image into to sample in y
};
    
/**
 * @class Background
 * @brief A class to evaluate %image background levels
 *
 * Break an image up into nx*ny sub-images and use 3-sigma clipped means to
 * estimate the background levels in each square.  Then use a bicubic spline or
 * bilinear interpolation (not currently implemented) algorithm to estimate background
 * at a given pixel coordinate.
 * Methods are available return background at a point (inefficiently), or an entire background image.
 * BackgroundControl contains public StatisticsControl and InterpolateControl members to allow
 * user control of how the backgrounds are computed.
 * @code
       math::BackgroundControl bctrl(math::NATURAL_SPLINE);
       bctrl.setNxSample(7);            // number of sub-image squares in x-dimension
       bctrl.setNySample(7);            // number of sub-image squares in y-dimention
       bctrl.sctrl.getNumSigmaClip(5.0); // use 5-sigma clipping for the sub-image means
       math::Background backobj = math::make_Background(img, bctrl);
       double somepoint = backobj.getPixel(i_x,i_y); // get the background at a pixel at i_x,i_y
       ImageT back = backobj.getFrame();             // get a whole background image
 * @endcode
 *
 */
class Background {
public:
    
    template<typename ImageT>
    explicit Background(ImageT const& img, ///< Image (or MaskedImage) whose background we want
                        BackgroundControl const& bgCtrl=BackgroundControl()); ///< Parameters to control Statistics and Interpolation
    
    ~Background() {}
    
    double getPixel(int const x, int const y) const;

    template<typename PixelT>
    typename lsst::afw::image::Image<PixelT>::Ptr getImage() const;
    
private:
    int _n;                             // number of pixels in the image
    double _meanclip;                   // n-sigma clipped mean
    int _imgWidth;                      // img.getWidth()
    int _imgHeight;                     // img.getHeight()
    int _nxSample;                      // number of sub-image squares in x-dimension
    int _nySample;                      // number of sub-image squares in y-dimension
    int _subimgWidth;                   // width in pixels of a subimage
    int _subimgHeight;                  // height in pixels of a subimage
    std::vector<int> _xcen;             // x center pix coords of sub images
    std::vector<int> _ycen;             // y center ...
    std::vector<int> _xorig;            // x origin pix coords of sub images
    std::vector<int> _yorig;            // y origin ...
    std::vector<std::vector<double> > _grid; // 3-sig clipped means for the grid of sub images.

    std::vector<std::vector<double> > _gridcolumns; // interpolated columns for the bicubic spline
    BackgroundControl _bctrl;           // control info set by user.
};

/**
 * @brief A convenience function that uses function overloading to make the correct type of Background
 *
 * cf. std::make_pair()
 */
template<typename ImageT>
Background make_Background(ImageT const& img, BackgroundControl const& bgCtrl=BackgroundControl()) {
    return Background(img, bgCtrl);
};
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
