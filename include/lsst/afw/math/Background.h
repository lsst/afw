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
#include <boost/preprocessor/seq.hpp>
#include "boost/shared_ptr.hpp"
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/Interpolate.h"

namespace lsst {
namespace afw {
namespace math {
class ApproximateControl;
template<typename T> class Approximate;

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
    CONST_PTR(StatisticsControl) getStatisticsControl() const { return _sctrl; }

    Property getStatisticsProperty() const { return _prop; }
    void setStatisticsProperty(Property prop) { _prop = prop; }
    void setStatisticsProperty(std::string prop) { _prop = stringToStatisticsProperty(prop); }
    
private:
    Interpolate::Style _style;          // style of interpolation to use
    int _nxSample;                      // number of grid squares to divide image into to sample in x
    int _nySample;                      // number of grid squares to divide image into to sample in y
    UndersampleStyle _undersampleStyle; // what to do when nx,ny are too small for the requested interp style
    PTR(StatisticsControl) _sctrl;           // statistics control object
    Property _prop;                          // statistics Property
};
    
/**
 * @class Background
 * @brief A virtual base class to evaluate %image background levels
 */
class Background : public daf::base::Citizen {
protected:
    template<typename ImageT>
    explicit Background(ImageT const& img, BackgroundControl const& bgCtrl);

    explicit Background(geom::Box2I const imageBBox, int const nx, int const ny);
    /// dtor
    virtual ~Background() { }
public:
    typedef float InternalPixelT;    ///< type used for any internal images, and returned by getApproximate

    /// Add a constant level to a background
    virtual void operator+=(float const delta) = 0;
    /// Subtract a constant level from a background
    virtual void operator-=(float const delta) = 0;
    /**
     * \brief Method to interpolate and return the background for entire image
     *
     * \return A boost shared-pointer to an image containing the estimated background
     */
    template<typename PixelT>
    PTR(lsst::afw::image::Image<PixelT>) getImage(
        Interpolate::Style const interpStyle,                   ///< Style of the interpolation
        UndersampleStyle const undersampleStyle=THROW_EXCEPTION ///< Behaviour if there are too few points
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

    /**
     * \brief Method to interpolate and return the background for entire image
     * \deprecated New code should specify the interpolation style in getImage, not the ctor
     */
    template<typename PixelT>
    PTR(lsst::afw::image::Image<PixelT>) getImage() const {
        return getImage<PixelT>(_bctrl.getInterpStyle(), _bctrl.getUndersampleStyle());
    }
    /**
     * Return the Interpolate::Style that we actually used in the last call to getImage()
     *
     * N.b. Interpolate can fallback to a lower order if there aren't enough samples
     */
    Interpolate::Style getAsUsedInterpStyle() const {
        return _asUsedInterpStyle;
    }
    /**
     * Return the UndersampleStyle that we actually used in the last call to getImage()
     */
    UndersampleStyle getAsUsedUndersampleStyle() const {
        return _asUsedUndersampleStyle;
    }
    /**
     * \brief Method to return an approximation to the background
     */
    PTR(math::Approximate<InternalPixelT>) getApproximate(
        ApproximateControl const& actrl,                        ///< Approximation style
        UndersampleStyle const undersampleStyle=THROW_EXCEPTION ///< Behaviour if there are too few points
                                                 ) const {
        InternalPixelT disambiguate = 0;
        return _getApproximate(actrl, undersampleStyle, disambiguate);
    }
    /**
     * Return the input image's (PARENT) bounding box
     */
    geom::Box2I getImageBBox() const { return _imgBBox; }

protected:
    geom::Box2I _imgBBox;                             ///< size and origin of input image
    BackgroundControl _bctrl;                         ///< control info set by user.
    mutable Interpolate::Style _asUsedInterpStyle;    ///< the style we actually used
    mutable UndersampleStyle _asUsedUndersampleStyle; ///< the undersampleStyle we actually used

    std::vector<double> _xcen;          ///< x center pix coords of sub images
    std::vector<double> _ycen;          ///< y center ...
    std::vector<int> _xorig;            ///< x origin pix coords of sub images
    std::vector<int> _yorig;            ///< y origin ...
    std::vector<int> _xsize;            ///< x size of sub images
    std::vector<int> _ysize;            ///< y size ...
    /*
     * We want getImage to be present in the base class, but a templated virtual function
     * is impossible.  So we'll solve the dilemma with a hack: explicitly defined
     * virtual functions for the image types we need
     */
#if !defined(SWIG)
// We'll evaluate LSST_makeBackground_get{Approximation,Image} for each type in
// LSST_makeBackground_get{Approximation,Image}_types,
// setting v to the second arg (i.e. "= 0" for the first invocation).  The first agument, m, is ignores

// Desired types
#define LSST_makeBackground_getImage_types            (double)(float)(int)
#define LSST_makeBackground_getApproximate_types      (Background::InternalPixelT)
#define LSST_makeBackground_getImage(m, v, T)                \
    virtual PTR(lsst::afw::image::Image<T>) _getImage( \
        Interpolate::Style const interpStyle,                     /* Style of the interpolation */ \
        UndersampleStyle const undersampleStyle=THROW_EXCEPTION,  /* Behaviour if there are too few points */\
        T = 0                                                     /* disambiguate */ \
                                                     ) const v;

#define LSST_makeBackground_getApproximate(m, v, T)           \
    virtual PTR(Approximate<T>) _getApproximate( \
        ApproximateControl const& actrl,                          /* Approximation style */ \
        UndersampleStyle const undersampleStyle=THROW_EXCEPTION,  /* Behaviour if there are too few points */\
        T = 0                                                     /* disambiguate */ \
                                                      ) const v;

    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getImage, = 0, LSST_makeBackground_getImage_types)
    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getApproximate, = 0, LSST_makeBackground_getApproximate_types)
#endif
private:
    Background(Background const&);
    Background& operator=(Background const&);    
    void _setCenOrigSize(int const width, int const height, int const nxSample, int const nySample);
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
 * \deprecated
 * there is also
 * \code
 // get the background at a pixel at i_x,i_y
 double someValue = backobj.getPixel(math::Interpolate::LINEAR, i_x, i_y);
 * \endcode
 */
class BackgroundMI : public Background {
public:
    template<typename ImageT>
    explicit BackgroundMI(ImageT const& img,
                        BackgroundControl const& bgCtrl);
    explicit BackgroundMI(geom::Box2I const imageDimensions,
                          image::MaskedImage<InternalPixelT> const& statsImage);
    
    virtual void operator+=(float const delta);
    virtual void operator-=(float const delta);

    double getPixel(Interpolate::Style const style, int const x, int const y) const;
    /**
     * \brief Return the background value at a point
     *
     * \warning This is very inefficient -- only use it for debugging, if then.
     *
     * \deprecated New code should specify the interpolation style in getPixel, not the ctor
     */
    double getPixel(int const x, int const y) const {
        return getPixel(_bctrl.getInterpStyle(), x, y);
    }
    /**
     * \brief Return the image of statistical quantities extracted from the image
     */
    lsst::afw::image::MaskedImage<InternalPixelT> getStatsImage() const {
        return _statsImage;
    }

private:
    lsst::afw::image::MaskedImage<InternalPixelT> _statsImage; // statistical properties for the grid of subimages
    mutable std::vector<std::vector<double> > _gridColumns; // interpolated columns for the bicubic spline

    void _setGridColumns(Interpolate::Style const interpStyle,
                         UndersampleStyle const undersampleStyle,
                         int const iX, std::vector<int> const& ypix) const;

#if !defined(SWIG) && defined(LSST_makeBackground_getImage)
    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getImage, , LSST_makeBackground_getImage_types)
    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getApproximate, , LSST_makeBackground_getApproximate_types)
#if 0                                   // keep for use in Background instantiations
#undef LSST_makeBackground_getImage_types
#undef LSST_makeBackground_getApproximate_types
#endif
#undef LSST_makeBackground_getImage
#undef LSST_makeBackground_getApproximate
#endif
    // Here's the worker function for _getImage (non-virtual; it's templated in BackgroundMI, not Background)
    template<typename PixelT>
    PTR(image::Image<PixelT>) doGetImage(Interpolate::Style const interpStyle_,
                                         UndersampleStyle const undersampleStyle) const;
    // and for _getApproximate
    template<typename PixelT>
    PTR(Approximate<PixelT>) doGetApproximate(ApproximateControl const& actrl,
                                              UndersampleStyle const undersampleStyle) const;
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
