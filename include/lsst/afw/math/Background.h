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


namespace lsst {
namespace afw {
namespace math {


enum UndersampleStyle {
    THROW_EXCEPTION,
    REDUCE_INTERP_ORDER,
    INCREASE_NXNYSAMPLE,
};


/**
 * @brief Conversion function to switch a string to an UndersampleStyle
 *
 */
UndersampleStyle stringToUndersampleStyle(std::string const style) {
    std::map<std::string, UndersampleStyle> undersampleStrings;
    undersampleStrings["THROW_EXCEPTION"]     = THROW_EXCEPTION;
    undersampleStrings["REDUCE_INTERP_ORDER"] = REDUCE_INTERP_ORDER;
    undersampleStrings["INCREASE_NXNYSAMPLE"] = INCREASE_NXNYSAMPLE;
    return undersampleStrings[style];
}

    
/**
 * @class BackgroundControl
 * @brief Pass parameters to a Background object
 */
class BackgroundControl {
public:
    
    BackgroundControl(
        Interpolate::Style const style = Interpolate::AKIMA_SPLINE, ///< Style of the interpolation
        int const nxSample = 10,        ///< Num. grid samples in x
        int const nySample = 10,        ///< Num. grid samples in y
        UndersampleStyle const undersampleStyle = THROW_EXCEPTION ///< Behaviour if there are too few points
                     )
        : _style(style), _nxSample(nxSample), _nySample(nySample),
          _undersampleStyle(undersampleStyle) {
        assert(nxSample > 0);
        assert(nySample > 0);
        sctrl = StatisticsControl();
    }
    
    // overload constructor to handle strings for both interp and undersample styles.
    BackgroundControl(std::string const style,
                      int const nxSample = 10, 
                      int const nySample = 10, 
                      std::string const undersampleStyle = "THROW_EXCEPTION" )
        : _style(math::stringToInterpStyle(style)), _nxSample(nxSample), _nySample(nySample),
          _undersampleStyle(math::stringToUndersampleStyle(undersampleStyle)) {
        assert(nxSample > 0);
        assert(nySample > 0);
        sctrl = StatisticsControl();
    }
    

    virtual ~BackgroundControl() {}
    void setNxSample (int nxSample) { assert(nxSample > 0); _nxSample = nxSample; }
    void setNySample (int nySample) { assert(nySample > 0); _nySample = nySample; }

    void setInterpStyle (Interpolate::Style const style) { _style = style; }
    // overload to take a string
    void setInterpStyle (std::string const style) { _style = math::stringToInterpStyle(style); }
    
    void setUndersampleStyle (UndersampleStyle const undersampleStyle) {
        _undersampleStyle = undersampleStyle;
    }
    // overload to take a string
    void setUndersampleStyle (std::string const undersampleStyle) {
        _undersampleStyle = math::stringToUndersampleStyle(undersampleStyle);
    }
    
    int getNxSample() const { return _nxSample; }
    int getNySample() const { return _nySample; }
    Interpolate::Style getInterpStyle() const { return _style; }
    UndersampleStyle getUndersampleStyle() const { return _undersampleStyle; }
    StatisticsControl sctrl;
private:
    Interpolate::Style _style;                       // style of interpolation to use
    int _nxSample;                      // number of grid squares to divide image into to sample in x
    int _nySample;                      // number of grid squares to divide image into to sample in y
    UndersampleStyle _undersampleStyle; // what to do when nx,ny are too small for the requested interp style
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
       math::BackgroundControl bctrl(math::Interpolate::NATURAL_SPLINE);
       bctrl.setNxSample(7);            // number of sub-image squares in x-dimension
       bctrl.setNySample(7);            // number of sub-image squares in y-dimention
       bctrl.sctrl.getNumSigmaClip(5.0); // use 5-sigma clipping for the sub-image means
       math::Background backobj = math::makeBackground(img, bctrl);
       double somepoint = backobj.getPixel(i_x,i_y); // get the background at a pixel at i_x,i_y
       ImageT back = backobj.getImage();             // get a whole background image
 * @endcode
 *
 */
class Background {
public:
    
    template<typename ImageT>
    explicit Background(ImageT const& img, ///< Image (or MaskedImage) whose background we want
                        BackgroundControl const& bgCtrl = BackgroundControl()); ///< Control Parameters
    
    virtual ~Background() {}
    
    double getPixel(int const x, int const y) const;

    template<typename PixelT>
    typename lsst::afw::image::Image<PixelT>::Ptr getImage() const;
    
    BackgroundControl getBackgroundControl() const { return _bctrl; }
    
private:
    int _n;                             // number of pixels in the image
    double _meanclip;                   // n-sigma clipped mean
    int _imgWidth;                      // img.getWidth()
    int _imgHeight;                     // img.getHeight()
    int _nxSample;                      // number of sub-image squares in x-dimension
    int _nySample;                      // number of sub-image squares in y-dimension
    int _subimgWidth;                   // width in pixels of a subimage
    int _subimgHeight;                  // height in pixels of a subimage
    std::vector<double> _xcen;             // x center pix coords of sub images
    std::vector<double> _ycen;          // y center ...
    std::vector<int> _xorig;            // x origin pix coords of sub images
    std::vector<int> _yorig;            // y origin ...
    std::vector<std::vector<double> > _grid; // 3-sig clipped means for the grid of sub images.

    std::vector<std::vector<double> > _gridcolumns; // interpolated columns for the bicubic spline
    BackgroundControl _bctrl;           // control info set by user.

    void _checkSampling();
};

/**
 * @brief A convenience function that uses function overloading to make the correct type of Background
 *
 * cf. std::make_pair()
 */
template<typename ImageT>
Background makeBackground(ImageT const& img, BackgroundControl const& bgCtrl = BackgroundControl()) {
    return Background(img, bgCtrl);
};
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
