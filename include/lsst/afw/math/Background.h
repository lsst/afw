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
 
#if !defined(LSST_AFW_MATH_BACKGROUND_H)
#define LSST_AFW_MATH_BACKGROUND_H
/**
 * @file Background.h
 * @brief Use bi-cubic interpolation to estimate image background
 * @ingroup afw
 * @author Steve Bickerton
 */

#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/Interpolate.h"


namespace lsst {
namespace afw {
namespace math {

//
// Remember to update stringToUndersampleStyle if you change this.
// If this happens often, we can play CPP games to put the definition in exactly one place, although swig
// may not be happy (so we could think m4 thoughts instead)
//
enum UndersampleStyle {
    THROW_EXCEPTION,
    REDUCE_INTERP_ORDER,
    INCREASE_NXNYSAMPLE
};
UndersampleStyle stringToUndersampleStyle(std::string const style);
    
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
        UndersampleStyle const undersampleStyle = THROW_EXCEPTION, ///< Behaviour if there are too few points
        StatisticsControl const sctrl = StatisticsControl(), ///< Configuration for Stats to be computed
        Property const prop = MEANCLIP ///< statistical property to use for grid points
                     )
        : _style(style),
          _nxSample(nxSample), _nySample(nySample),
          _undersampleStyle(undersampleStyle),
          _sctrl(new StatisticsControl(sctrl)),
          _prop(prop) {
        assert(nxSample > 0);
        assert(nySample > 0);
    }
    
    // overload constructor to handle strings for both interp and undersample styles.
    BackgroundControl(
        std::string const style, ///< Style of the interpolation
        int const nxSample = 10, ///< num. grid samples in x
        int const nySample = 10, ///< num. grid samples in y
        std::string const undersampleStyle = "THROW_EXCEPTION", ///< behaviour if there are too few points
        StatisticsControl const sctrl = StatisticsControl(), ///< configuration for stats to be computed
        std::string const prop = "MEANCLIP" ///< statistical property to use for grid points
                     )
        : _style(math::stringToInterpStyle(style)),
          _nxSample(nxSample), _nySample(nySample),
          _undersampleStyle(math::stringToUndersampleStyle(undersampleStyle)),
          _sctrl(new StatisticsControl(sctrl)),
          _prop(stringToStatisticsProperty(prop)) {
        assert(nxSample > 0);
        assert(nySample > 0);
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
    StatisticsControl::Ptr getStatisticsControl() { return _sctrl; }

    Property getStatisticsProperty() { return _prop; }
    void setStatisticsProperty(Property prop) { _prop = prop; }
    void setStatisticsProperty(std::string prop) { _prop = stringToStatisticsProperty(prop); }
    
private:
    Interpolate::Style _style;                       // style of interpolation to use
    int _nxSample;                      // number of grid squares to divide image into to sample in x
    int _nySample;                      // number of grid squares to divide image into to sample in y
    UndersampleStyle _undersampleStyle; // what to do when nx,ny are too small for the requested interp style
    StatisticsControl::Ptr _sctrl;           // statistics control object
    Property _prop;                          // statistics Property
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
       bctrl.sctrl.setNumSigmaClip(5.0); // use 5-sigma clipping for the sub-image means
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
    
    void operator+=(float const delta);
    void operator-=(float const delta);

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
    std::vector<double> _xcen;             // x center pix coords of sub images
    std::vector<double> _ycen;          // y center ...
    std::vector<int> _xorig;            // x origin pix coords of sub images
    std::vector<int> _yorig;            // y origin ...
    std::vector<int> _xsize;            // x size of sub images
    std::vector<int> _ysize;            // y size ...
    std::vector<std::vector<double> > _grid; // 3-sig clipped means for the grid of sub images.

    std::vector<std::vector<double> > _gridcolumns; // interpolated columns for the bicubic spline
    BackgroundControl _bctrl;           // control info set by user.

    void _checkSampling();
    void _set_gridcolums(int iX, std::vector<int> const& ypix);
};

/**
 * @brief A convenience function that uses function overloading to make the correct type of Background
 *
 * cf. std::make_pair()
 */
template<typename ImageT>
Background makeBackground(ImageT const& img, BackgroundControl const& bgCtrl = BackgroundControl()) {
    return Background(img, bgCtrl);
}
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
