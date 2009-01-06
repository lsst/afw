#if !defined(LSST_AFW_MATH_BACKGROUND_H)
#define LSST_AFW_MATH_BACKGROUND_H
/**
 * \file
 * \brief ImageT Background
 */

#include "boost/shared_ptr.hpp"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/Interpolate.h"


namespace lsst { namespace afw { namespace math {

/// \brief Pass parameters to a Background object
class BackgroundControl {
public:
    BackgroundControl(Style const style=math::NATURAL_SPLINE, int const nxSample=10, int const nySample=10)
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
 * A class to evaluate %image background levels
 *
 * Break an image up into nx*ny sub-images and use 3-sigma clipped means to
 * estimate the background levels in each square.  Then use a bicubic spline or
 * bilinear interpolation algorithm to estimate background at a given pixel coordinate.
 * Methods are available return background at a point, or an entire background image.
 * \code
       math::BackgroundControl bctrl(math::NATURAL_SPLINE);
       bctrl.setNxSample(7);            // number of sub-image squares in x-dimension
       bctrl.setNySample(7);            // number of sub-image squares in y-dimention
       math::Background<ImageT> backobj = math::make_Background(img, bctrl);
       double somepoint = backobj.getPixel(i_x,i_y); // get the background at a pixel at i_x,i_y
       ImageT back = backobj.getFrame();             // get a whole background image
 * \endcode
 *
 * (Note that we used a helper function, \c make_Background, rather that the constructor directly so that
 * the compiler could deduce the types -- cf. \c std::make_pair)
 */
template<typename ImageT>
class Background {
public:
    
    explicit Background(ImageT const& img, BackgroundControl const& bgCtrl=BackgroundControl());
    ~Background() {}
    typename ImageT::Pixel getPixel(int const x, int const y) const;
    typename lsst::afw::image::Image<typename ImageT::Pixel>::Ptr getImage() const;
    //~Background() { delete _grid; };
    
private:
    int _n;                             // number of pixels in the image
    double _meanclip;                   // n-sigma clipped mean
    int _imgWidth;                      // img.getWidth()
    int _imgHeight;                     // img.getHeight()
    //ImageT _img;                        // the image
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

/// A convenience function that uses function overloading to make the correct type of Background
/// cf. std::make_pair()
template<typename ImageT>
Background<ImageT> make_Background(ImageT const& img, BackgroundControl const& bgCtrl=BackgroundControl()) {
    return Background<ImageT>(img, bgCtrl);
};
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
