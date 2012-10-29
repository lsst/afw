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
        Property const prop = MEANCLIP                       ///< statistical property to use for grid points
                     )
        : _nxSample(nxSample), _nySample(nySample),
          _sctrl(new StatisticsControl(sctrl)),
          _prop(prop) {
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
    
    int getNxSample() const { return _nxSample; }
    int getNySample() const { return _nySample; }
    PTR(StatisticsControl) getStatisticsControl() { return _sctrl; }

    Property getStatisticsProperty() { return _prop; }
    void setStatisticsProperty(Property prop) { _prop = prop; }
    void setStatisticsProperty(std::string prop) { setStatisticsProperty(stringToStatisticsProperty(prop)); }
    
private:
    int _nxSample;                      // number of grid squares to divide image into to sample in x
    int _nySample;                      // number of grid squares to divide image into to sample in y
    PTR(StatisticsControl) _sctrl;           // statistics control object
    Property _prop;                          // statistics Property
};
    
/**
 * @class Background
 * @brief A virtual base class to evaluate %image background levels
 */
class Background {
protected:
    template<typename ImageT>
    explicit Background(ImageT const& img, ///< Image (or MaskedImage) whose background we want
                            BackgroundControl const& bgCtrl); ///< Control Parameters
    
    virtual ~Background() { }
public:
    virtual void operator+=(float const delta) = 0;
    virtual void operator-=(float const delta) = 0;
    /**
     * \brief Method to interpolate and return the background for entire image
     *
     * \return A boost shared-pointer to an image containing the estimated background
     */
    template<typename PixelT>
    PTR(lsst::afw::image::Image<PixelT>) getImage(
        Interpolate::Style const interpStyle,                           ///< Style of the interpolation
        UndersampleStyle const undersampleStyle=THROW_EXCEPTION   ///< Behaviour if there are too few points
                                                 ) const {
        PixelT disambiguate = 0;
        return _getImage(interpStyle, undersampleStyle, disambiguate);
    }
    /**
     * \brief Method to interpolate and return the background for entire image
     *
     * \return A boost shared-pointer to an image containing the estimated background
     */
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
protected:
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
    /*
     * We want getImage to be present in the base class, but a templated virtual function
     * is impossible.  So we'll solve the dilemma with a hack: explicitly defined
     * virtual functions for the image types we need
     */
#if !defined(SWIG)
#define makeBackground_getImage(T) \
    virtual PTR(lsst::afw::image::Image<T>) _getImage( \
        Interpolate::Style const interpStyle,                           /* Style of the interpolation */\
        UndersampleStyle const undersampleStyle=THROW_EXCEPTION,   /* Behaviour if there are too few points */\
        T = 0                                                      /* disambiguate */ \
                                                   ) const
    makeBackground_getImage(double) = 0;
    makeBackground_getImage(float) = 0;
    makeBackground_getImage(int) = 0;
#endif
private:
    Background(Background const&);
    Background& operator=(Background const&);    
};
    
/**
 * @class BackgroundMI
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
       PTR(math::Background) backobj = math::makeBackground(img, bctrl);
       // get a whole background image
       Image<PixelT> back = backobj->getImage<PixelT>(math::Interpolate::NATURAL_SPLINE);
 * @endcode
 *
 * There is also
 * \code
 // get the background at a pixel at i_x,i_y
 double someValue = backobj.getPixel(math::Interpolate::LINEAR, i_x, i_y);
 * \endcode
 * \note This method may be inefficient, but is necessary for e.g. asking for the background level
 * at the position of an object
 */
class BackgroundMI : public Background {
public:
    template<typename ImageT>
    explicit BackgroundMI(ImageT const& img, ///< Image (or MaskedImage) whose background we want
                        BackgroundControl const& bgCtrl); ///< Control Parameters
    
    virtual void operator+=(float const delta);
    virtual void operator-=(float const delta);

    double getPixel(Interpolate::Style const style, int const x, int const y) const;
    /**
     * Return the image of statistical quantities extracted from the image
     */
    CONST_PTR(lsst::afw::image::MaskedImage<float>) getStatsImage() const {
        return _statsImage;
    }

private:
    PTR(lsst::afw::image::MaskedImage<float>) _statsImage;  // statistical properties for the grid of subimages
    mutable std::vector<std::vector<double> > _gridcolumns; // interpolated columns for the bicubic spline

    void _set_gridcolumns(Interpolate::Style const interpStyle,
                          int const iX, std::vector<int> const& ypix) const;
#if !defined(SWIG) && defined(makeBackground_getImage)
    makeBackground_getImage(double);
    makeBackground_getImage(float);
    makeBackground_getImage(int);
#undef makeBackground_getImage
#endif
    // Here's the worker function for _getImage (non-virtual; it's templated in BackgroundMI, not Background)
    template<typename PixelT>
    PTR(image::Image<PixelT>) doGetImage(
    Interpolate::Style const interpStyle_,       ///< Style of the interpolation
        UndersampleStyle const undersampleStyle ///< Behaviour if there are too few points
        ) const;
};

/**
 * @brief A convenience function that uses function overloading to make the correct type of Background
 *
 * cf. std::make_pair()
 */
template<typename ImageT>
PTR(Background) makeBackground(ImageT const& img, BackgroundControl const& bgCtrl) {
    return PTR(Background)(new BackgroundMI(img, bgCtrl));
}
    
}}}

#endif  //   LSST_AFW_MATH_BACKGROUND_H
