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
 * @brief Estimate image backgrounds
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
UndersampleStyle stringToUndersampleStyle(std::string const &style);
    
/**
 * @class BackgroundControl
 * @brief Pass parameters to a Background object
 */
class BackgroundControl {
public:
    BackgroundControl(
        int const nxSample,                                  ///< Num. grid samples in x
        int const nySample,                                  ///< Num. grid samples in y
        StatisticsControl const sctrl = StatisticsControl(), ///< Configuration for Stats to be computed
        Property const prop = MEANCLIP ///< statistical property to use for grid points
                     )
        : _style(Interpolate::AKIMA_SPLINE),
          _nxSample(nxSample), _nySample(nySample),
          _undersampleStyle(THROW_EXCEPTION),
          _sctrl(new StatisticsControl(sctrl)),
          _prop(prop) {
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              str(boost::format("You must specify at least one point, not %dx%d")
                                  % nxSample % nySample)
                             );
        }
    }
    
    /**
     * Overload constructor to handle string for statistical operator
     */
    BackgroundControl(
        int const nxSample,             ///< num. grid samples in x
        int const nySample,             ///< num. grid samples in y
        StatisticsControl const &sctrl, ///< configuration for stats to be computed
        std::string const &prop         ///< statistical property to use for grid points
                     )
        : _style(Interpolate::AKIMA_SPLINE),
          _nxSample(nxSample), _nySample(nySample),
          _undersampleStyle(THROW_EXCEPTION),
          _sctrl(new StatisticsControl(sctrl)),
          _prop(stringToStatisticsProperty(prop)) {
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              str(boost::format("You must specify at least one point, not %dx%d")
                                  % nxSample % nySample)
                             );
        }
    }
    // And now the two old APIs (preserved for backward compatibility)
    /**
     * \deprecated New code should specify the interpolation style in getImage, not the BackgroundControl ctor
     */
    BackgroundControl(
        Interpolate::Style const style, ///< Style of the interpolation
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
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              str(boost::format("You must specify at least one point, not %dx%d")
                                  % nxSample % nySample)
                             );
        }
    }
    
    /**
     * Overload constructor to handle strings for both interp and undersample styles.
     *
     * \deprecated New code should specify the interpolation style in getImage, not the BackgroundControl ctor
     */
    BackgroundControl(
        std::string const &style, ///< Style of the interpolation
        int const nxSample = 10, ///< num. grid samples in x
        int const nySample = 10, ///< num. grid samples in y
        std::string const &undersampleStyle = "THROW_EXCEPTION", ///< behaviour if there are too few points
        StatisticsControl const sctrl = StatisticsControl(), ///< configuration for stats to be computed
        std::string const &prop = "MEANCLIP" ///< statistical property to use for grid points
                     )
        : _style(math::stringToInterpStyle(style)),
          _nxSample(nxSample), _nySample(nySample),
          _undersampleStyle(math::stringToUndersampleStyle(undersampleStyle)),
          _sctrl(new StatisticsControl(sctrl)),
          _prop(stringToStatisticsProperty(prop)) {
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              str(boost::format("You must specify at least one point, not %dx%d")
                                  % nxSample % nySample)
                             );
        }
    }

    virtual ~BackgroundControl() {}
    void setNxSample (int nxSample) {
        if (nxSample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              str(boost::format("nxSample must be position, not %d") % nxSample));
        }
        _nxSample = nxSample;
    }
    void setNySample (int nySample) {
        if (nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                              str(boost::format("nySample must be position, not %d") % nySample));
        }
        _nySample = nySample;
    }

    void setInterpStyle (Interpolate::Style const style) { _style = style; }
    // overload to take a string
    void setInterpStyle (std::string const &style) { _style = math::stringToInterpStyle(style); }
    
    void setUndersampleStyle (UndersampleStyle const undersampleStyle) {
        _undersampleStyle = undersampleStyle;
    }
    // overload to take a string
    void setUndersampleStyle (std::string const &undersampleStyle) {
        _undersampleStyle = math::stringToUndersampleStyle(undersampleStyle);
    }
    
    int getNxSample() const { return _nxSample; }
    int getNySample() const { return _nySample; }
    Interpolate::Style getInterpStyle() const {
        if (_style < 0 || _style >= Interpolate::NUM_STYLES) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              str(boost::format("Style %d is invalid") % _style));

        }
        return _style;
    }
    UndersampleStyle getUndersampleStyle() const { return _undersampleStyle; }
    PTR(StatisticsControl) getStatisticsControl() { return _sctrl; }

    Property getStatisticsProperty() { return _prop; }
    void setStatisticsProperty(Property prop) { _prop = prop; }
    void setStatisticsProperty(std::string prop) { _prop = stringToStatisticsProperty(prop); }
    
private:
    Interpolate::Style _style;                       // style of interpolation to use
    int _nxSample;                      // number of grid squares to divide image into to sample in x
    int _nySample;                      // number of grid squares to divide image into to sample in y
    UndersampleStyle _undersampleStyle; // what to do when nx,ny are too small for the requested interp style
    PTR(StatisticsControl) _sctrl;           // statistics control object
    Property _prop;                          // statistics Property
};
    
/**
 * @class Background
 * @brief A class to evaluate %image background levels
 *
 * Break an image up into nx*ny sub-images and use a statistical to estimate the background levels in each
 * square.  Then use a user-specified or algorithm to estimate background at a given pixel coordinate.
 *
 * Methods are available to return the background at a point (inefficiently), or an entire background image.
 * BackgroundControl contains a public StatisticsControl member to allow user control of how the backgrounds are computed.
 * @code
       math::BackgroundControl bctrl(7, 7);  // number of sub-image squares in {x,y}-dimensions
       bctrl.sctrl.setNumSigmaClip(5.0);     // use 5-sigma clipping for the sub-image means
       math::Background backobj = math::makeBackground(img, bctrl);
       // get a whole background image
       Image<PixelT> back = backobj.getImage<PixelT>(math::Interpolate::NATURAL_SPLINE);
 * @endcode
 *
 * \deprecated
 * there is also
 * \code
 // get the background at a pixel at i_x,i_y
 double someValue = backobj.getPixel(math::Interpolate::LINEAR, i_x, i_y);
 * \endcode
 */
class Background {
public:
    template<typename ImageT>
    explicit Background(ImageT const& img, ///< Image (or MaskedImage) whose background we want
                        BackgroundControl const& bgCtrl); ///< Control Parameters
    
    ~Background() { }
    
    void operator+=(float const delta);
    void operator-=(float const delta);

    double getPixel(Interpolate::Style const style, int const x, int const y) const;
    /**
     * Return the background value at a point
     *
     * \note This is very inefficient -- only use it for debugging, if then.
     *
     * \deprecated New code should specify the interpolation style in getPixel, not the ctor
     */
    double getPixel(int const x, int const y) const {
        return getPixel(_bctrl.getInterpStyle(), x, y);
    }

    /**
     * \deprecated New code should specify the interpolation style in getImage, not the ctor
     */
    template<typename PixelT>
    PTR(lsst::afw::image::Image<PixelT>) getImage() const {
        return getImage<PixelT>(_bctrl.getInterpStyle(), _bctrl.getUndersampleStyle());
    }
    /**
     * \brief Method to compute the background for entire image and return a background image
     *
     * \return A boost shared-pointer to an image containing the estimated background
     */
    template<typename PixelT>
    PTR(lsst::afw::image::Image<PixelT>) getImage(
        Interpolate::Style const interpStyle,                           ///< Style of the interpolation
        UndersampleStyle const undersampleStyle=THROW_EXCEPTION   ///< Behaviour if there are too few points
                                                          ) const;
    template<typename PixelT>
    PTR(lsst::afw::image::Image<PixelT>) getImage(
        std::string const &interpStyle, ///< Style of the interpolation
        std::string const &undersampleStyle="THROW_EXCEPTION"   ///< Behaviour if there are too few points
                                                 ) const {
        return getImage<PixelT>(math::stringToInterpStyle(interpStyle),
                                stringToUndersampleStyle(undersampleStyle));
    }
    
    BackgroundControl getBackgroundControl() const { return _bctrl; }
    /**
     * Return the Interpolate::Style that we actually used
     */
    Interpolate::Style getAsUsedInterpStyle() const {
        return _asUsedInterpStyle;
    }
private:
    int _imgWidth;                      // img.getWidth()
    int _imgHeight;                     // img.getHeight()
    int _nxSample;                      // number of sub-image squares in x-dimension
    int _nySample;                      // number of sub-image squares in y-dimension
    BackgroundControl _bctrl;           // control info set by user.
    mutable Interpolate::Style _asUsedInterpStyle; // the style we actually used

    std::vector<double> _xcen;          // x center pix coords of sub images
    std::vector<double> _ycen;          // y center ...
    std::vector<int> _xorig;            // x origin pix coords of sub images
    std::vector<int> _yorig;            // y origin ...
    std::vector<int> _xsize;            // x size of sub images
    std::vector<int> _ysize;            // y size ...

    mutable std::vector<std::vector<double> > _grid; // 3-sig clipped means for the grid of sub images.
    mutable std::vector<std::vector<double> > _gridcolumns; // interpolated columns for the bicubic spline

    void _set_gridcolumns(Interpolate::Style const interpStyle,
                          int const iX, std::vector<int> const& ypix) const;
};

/**
 * @brief A convenience function that uses function overloading to make the correct type of Background
 *
 * cf. std::make_pair()
 */
template<typename ImageT>
Background makeBackground(ImageT const& img, BackgroundControl const& bgCtrl) {
    return Background(img, bgCtrl);
}
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
