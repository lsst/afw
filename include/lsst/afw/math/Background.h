// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2015 AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#if !defined(LSST_AFW_MATH_BACKGROUND_H)
#define LSST_AFW_MATH_BACKGROUND_H
/*
 * Estimate image backgrounds
 */
#include <boost/preprocessor/seq.hpp>
#include <memory>
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Approximate.h"

namespace lsst {
namespace afw {
namespace math {

//
// Remember to update stringToUndersampleStyle if you change this.
// If this happens often, we can play CPP games to put the definition in exactly one place, although swig
// may not be happy (so we could think m4 thoughts instead)
//
enum UndersampleStyle { THROW_EXCEPTION, REDUCE_INTERP_ORDER, INCREASE_NXNYSAMPLE };
/**
 * Conversion function to switch a string to an UndersampleStyle
 */
UndersampleStyle stringToUndersampleStyle(std::string const& style);

/**
 * Pass parameters to a Background object
 */
class BackgroundControl {
public:
    /**
     * @param nxSample Num. grid samples in x
     * @param nySample Num. grid samples in y
     * @param sctrl Configuration for Stats to be computed
     * @param prop statistical property to use for grid points
     * @param actrl configuration for approx to be computed
     */
    BackgroundControl(int const nxSample, int const nySample,
                      StatisticsControl const sctrl = StatisticsControl(), Property const prop = MEANCLIP,
                      ApproximateControl const actrl = ApproximateControl(ApproximateControl::UNKNOWN, 1))
            : _style(Interpolate::AKIMA_SPLINE),
              _nxSample(nxSample),
              _nySample(nySample),
              _undersampleStyle(THROW_EXCEPTION),
              _sctrl(new StatisticsControl(sctrl)),
              _prop(prop),
              _actrl(new ApproximateControl(actrl)) {
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              str(boost::format("You must specify at least one point, not %dx%d") % nxSample %
                                  nySample));
        }
    }

    /**
     * Overload constructor to handle string for statistical operator
     *
     * @param nxSample num. grid samples in x
     * @param nySample num. grid samples in y
     * @param sctrl configuration for stats to be computed
     * @param prop statistical property to use for grid points
     * @param actrl configuration for approx to be computed
     */
    BackgroundControl(int const nxSample, int const nySample, StatisticsControl const& sctrl,
                      std::string const& prop,
                      ApproximateControl const actrl = ApproximateControl(ApproximateControl::UNKNOWN, 1))
            : _style(Interpolate::AKIMA_SPLINE),
              _nxSample(nxSample),
              _nySample(nySample),
              _undersampleStyle(THROW_EXCEPTION),
              _sctrl(new StatisticsControl(sctrl)),
              _prop(stringToStatisticsProperty(prop)),
              _actrl(new ApproximateControl(actrl)) {
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              str(boost::format("You must specify at least one point, not %dx%d") % nxSample %
                                  nySample));
        }
    }
    // And now the two old APIs (preserved for backward compatibility)
    /**
     * @deprecated New code should specify the interpolation style in getImage, not the BackgroundControl ctor
     *
     * @param style Style of the interpolation
     * @param nxSample Num. grid samples in x
     * @param nySample Num. grid samples in y
     * @param undersampleStyle Behaviour if there are too few points
     * @param sctrl Configuration for Stats to be computed
     * @param prop statistical property to use for grid points
     * @param actrl configuration for approx to be computed
     */
    BackgroundControl(Interpolate::Style const style, int const nxSample = 10, int const nySample = 10,
                      UndersampleStyle const undersampleStyle = THROW_EXCEPTION,
                      StatisticsControl const sctrl = StatisticsControl(), Property const prop = MEANCLIP,
                      ApproximateControl const actrl = ApproximateControl(ApproximateControl::UNKNOWN, 1)

                              )
            : _style(style),
              _nxSample(nxSample),
              _nySample(nySample),
              _undersampleStyle(undersampleStyle),
              _sctrl(new StatisticsControl(sctrl)),
              _prop(prop),
              _actrl(new ApproximateControl(actrl)) {
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              str(boost::format("You must specify at least one point, not %dx%d") % nxSample %
                                  nySample));
        }
    }

    /**
     * Overload constructor to handle strings for both interp and undersample styles.
     *
     * @deprecated New code should specify the interpolation style in getImage, not the BackgroundControl ctor
     *
     * @param style Style of the interpolation
     * @param nxSample num. grid samples in x
     * @param nySample num. grid samples in y
     * @param undersampleStyle behaviour if there are too few points
     * @param sctrl configuration for stats to be computed
     * @param prop statistical property to use for grid points
     * @param actrl configuration for approx to be computed
     */
    BackgroundControl(std::string const& style, int const nxSample = 10, int const nySample = 10,
                      std::string const& undersampleStyle = "THROW_EXCEPTION",
                      StatisticsControl const sctrl = StatisticsControl(),
                      std::string const& prop = "MEANCLIP",
                      ApproximateControl const actrl = ApproximateControl(ApproximateControl::UNKNOWN, 1))
            : _style(math::stringToInterpStyle(style)),
              _nxSample(nxSample),
              _nySample(nySample),
              _undersampleStyle(math::stringToUndersampleStyle(undersampleStyle)),
              _sctrl(new StatisticsControl(sctrl)),
              _prop(stringToStatisticsProperty(prop)),
              _actrl(new ApproximateControl(actrl)) {
        if (nxSample <= 0 || nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              str(boost::format("You must specify at least one point, not %dx%d") % nxSample %
                                  nySample));
        }
    }

    virtual ~BackgroundControl() {}
    void setNxSample(int nxSample) {
        if (nxSample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              str(boost::format("nxSample must be position, not %d") % nxSample));
        }
        _nxSample = nxSample;
    }
    void setNySample(int nySample) {
        if (nySample <= 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              str(boost::format("nySample must be position, not %d") % nySample));
        }
        _nySample = nySample;
    }

    void setInterpStyle(Interpolate::Style const style) { _style = style; }
    // overload to take a string
    void setInterpStyle(std::string const& style) { _style = math::stringToInterpStyle(style); }

    void setUndersampleStyle(UndersampleStyle const undersampleStyle) {
        _undersampleStyle = undersampleStyle;
    }
    // overload to take a string
    void setUndersampleStyle(std::string const& undersampleStyle) {
        _undersampleStyle = math::stringToUndersampleStyle(undersampleStyle);
    }

    int getNxSample() const { return _nxSample; }
    int getNySample() const { return _nySample; }
    Interpolate::Style getInterpStyle() const {
        if (_style < 0 || _style >= Interpolate::NUM_STYLES) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              str(boost::format("Style %d is invalid") % _style));
        }
        return _style;
    }
    UndersampleStyle getUndersampleStyle() const { return _undersampleStyle; }
    std::shared_ptr<StatisticsControl> getStatisticsControl() { return _sctrl; }
    std::shared_ptr<StatisticsControl const> getStatisticsControl() const { return _sctrl; }

    Property getStatisticsProperty() const { return _prop; }
    void setStatisticsProperty(Property prop) { _prop = prop; }
    void setStatisticsProperty(std::string prop) { _prop = stringToStatisticsProperty(prop); }

    void setApproximateControl(std::shared_ptr<ApproximateControl> actrl) { _actrl = actrl; }
    std::shared_ptr<ApproximateControl> getApproximateControl() { return _actrl; }
    std::shared_ptr<ApproximateControl const> getApproximateControl() const { return _actrl; }

private:
    Interpolate::Style _style;           // style of interpolation to use
    int _nxSample;                       // number of grid squares to divide image into to sample in x
    int _nySample;                       // number of grid squares to divide image into to sample in y
    UndersampleStyle _undersampleStyle;  // what to do when nx,ny are too small for the requested interp style
    std::shared_ptr<StatisticsControl> _sctrl;   // statistics control object
    Property _prop;                              // statistics Property
    std::shared_ptr<ApproximateControl> _actrl;  // approximate control object
};

/**
 * A virtual base class to evaluate %image background levels
 */
class Background : public daf::base::Citizen {
protected:
    /**
     * Constructor for Background
     *
     * Estimate the statistical properties of the Image in a grid of cells;  we'll later call
     * getImage() to interpolate those values, creating an image the same size as the original
     *
     * @note The old and deprecated API specified the interpolation style as part of the BackgroundControl
     * object passed to this ctor.  This is still supported, but the work isn't done until the getImage()
     * method is called
     *
     * @param img ImageT (or MaskedImage) whose properties we want
     * @param bgCtrl Control how the Background is estimated
     */
    template <typename ImageT>
    explicit Background(ImageT const& img, BackgroundControl const& bgCtrl);

    /**
     * Create a Background without any values in it
     *
     * @param imageBBox Bounding box for image to be created by getImage()
     * @param nx Number of samples in x-direction
     * @param ny Number of samples in y-direction
     *
     * @note This ctor is mostly used to create a Background given its sample values, and that (in turn)
     * is mostly used to implement persistence.
     */
    explicit Background(geom::Box2I const imageBBox, int const nx, int const ny);
    /// dtor
    virtual ~Background() {}

public:
    typedef float InternalPixelT;  ///< type used for any internal images, and returned by getApproximate

    /// Add a constant level to a background
    virtual Background& operator+=(float const delta) = 0;
    /// Subtract a constant level from a background
    virtual Background& operator-=(float const delta) = 0;
    /**
     * Method to interpolate and return the background for entire image
     *
     * @param interpStyle Style of the interpolation
     * @param undersampleStyle Behaviour if there are too few points
     * @returns A boost shared-pointer to an image containing the estimated background
     */
    template <typename PixelT>
    std::shared_ptr<lsst::afw::image::Image<PixelT>> getImage(
            Interpolate::Style const interpStyle,
            UndersampleStyle const undersampleStyle = THROW_EXCEPTION) const {
        return getImage<PixelT>(_imgBBox, interpStyle, undersampleStyle);
    }
    /**
     * Method to interpolate and return the background for entire image
     *
     * @param interpStyle Style of the interpolation
     * @param undersampleStyle Behaviour if there are too few points
     * @returns A boost shared-pointer to an image containing the estimated background
     */
    template <typename PixelT>
    std::shared_ptr<lsst::afw::image::Image<PixelT>> getImage(
            std::string const& interpStyle, std::string const& undersampleStyle = "THROW_EXCEPTION") const {
        return getImage<PixelT>(math::stringToInterpStyle(interpStyle),
                                stringToUndersampleStyle(undersampleStyle));
    }
    /**
     * @param bbox Bounding box for sub-image
     * @param interpStyle Style of the interpolation
     * @param undersampleStyle Behaviour if there are too few points
     */
    template <typename PixelT>
    std::shared_ptr<lsst::afw::image::Image<PixelT>> getImage(
            lsst::afw::geom::Box2I const& bbox, Interpolate::Style const interpStyle,
            UndersampleStyle const undersampleStyle = THROW_EXCEPTION) const {
        return _getImage(bbox, interpStyle, undersampleStyle, static_cast<PixelT>(0));
    }
    /**
     * @param bbox Bounding box for sub-image
     * @param interpStyle Style of the interpolation
     * @param undersampleStyle Behaviour if there are too few points
     */
    template <typename PixelT>
    std::shared_ptr<lsst::afw::image::Image<PixelT>> getImage(
            lsst::afw::geom::Box2I const& bbox, std::string const& interpStyle,
            std::string const& undersampleStyle = "THROW_EXCEPTION") const {
        return _getImage(bbox, math::stringToInterpStyle(interpStyle),
                         stringToUndersampleStyle(undersampleStyle), static_cast<PixelT>(0));
    }

    /**
     * Method to interpolate and return the background for entire image
     * @deprecated New code should specify the interpolation style in getImage, not the ctor
     */
    template <typename PixelT>
    std::shared_ptr<lsst::afw::image::Image<PixelT>> getImage() const {
        return getImage<PixelT>(_bctrl->getInterpStyle(), _bctrl->getUndersampleStyle());
    }
    /**
     * Return the Interpolate::Style that we actually used in the last call to getImage()
     *
     * N.b. Interpolate can fallback to a lower order if there aren't enough samples
     */
    Interpolate::Style getAsUsedInterpStyle() const { return _asUsedInterpStyle; }
    /**
     * Return the UndersampleStyle that we actually used in the last call to getImage()
     */
    UndersampleStyle getAsUsedUndersampleStyle() const { return _asUsedUndersampleStyle; }
    /**
     * Method to return an approximation to the background
     *
     * @param actrl Approximation style
     * @param undersampleStyle Behaviour if there are too few points
     */
    std::shared_ptr<math::Approximate<InternalPixelT>> getApproximate(
            ApproximateControl const& actrl,
            UndersampleStyle const undersampleStyle = THROW_EXCEPTION) const {
        InternalPixelT disambiguate = 0;
        return _getApproximate(actrl, undersampleStyle, disambiguate);
    }
    /**
     * Return the input image's (PARENT) bounding box
     */
    geom::Box2I getImageBBox() const { return _imgBBox; }

    std::shared_ptr<BackgroundControl> getBackgroundControl() { return _bctrl; }
    std::shared_ptr<BackgroundControl const> getBackgroundControl() const { return _bctrl; }

protected:
    geom::Box2I _imgBBox;                              ///< size and origin of input image
    std::shared_ptr<BackgroundControl> _bctrl;         ///< control info set by user.
    mutable Interpolate::Style _asUsedInterpStyle;     ///< the style we actually used
    mutable UndersampleStyle _asUsedUndersampleStyle;  ///< the undersampleStyle we actually used

    std::vector<double> _xcen;  ///< x center pix coords of sub images
    std::vector<double> _ycen;  ///< y center ...
    std::vector<int> _xorig;    ///< x origin pix coords of sub images
    std::vector<int> _yorig;    ///< y origin ...
    std::vector<int> _xsize;    ///< x size of sub images
    std::vector<int> _ysize;    ///< y size ...
                                /*
                                 * We want getImage to be present in the base class, but a templated virtual function
                                 * is impossible.  So we'll solve the dilemma with a hack: explicitly defined
                                 * virtual functions for the image types we need
                                 */
// We'll evaluate LSST_makeBackground_get{Approximation,Image} for each type in
// LSST_makeBackground_get{Approximation,Image}_types,
// setting v to the second arg (i.e. "= 0" for the first invocation).  The first agument, m, is ignores

// Desired types
#define LSST_makeBackground_getImage_types (Background::InternalPixelT)
#define LSST_makeBackground_getApproximate_types (Background::InternalPixelT)
#define LSST_makeBackground_getImage(m, v, T)                                      \
    virtual std::shared_ptr<lsst::afw::image::Image<T>> _getImage(                 \
            lsst::afw::geom::Box2I const& bbox,                                    \
            Interpolate::Style const interpStyle, /* Style of the interpolation */ \
            UndersampleStyle const undersampleStyle =                              \
                    THROW_EXCEPTION, /* Behaviour if there are too few points */   \
            T = 0                    /* disambiguate */                            \
            ) const v;

#define LSST_makeBackground_getApproximate(m, v, T)                              \
    virtual std::shared_ptr<Approximate<T>> _getApproximate(                     \
            ApproximateControl const& actrl, /* Approximation style */           \
            UndersampleStyle const undersampleStyle =                            \
                    THROW_EXCEPTION, /* Behaviour if there are too few points */ \
            T = 0                    /* disambiguate */                          \
            ) const v;

    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getImage, = 0, LSST_makeBackground_getImage_types)
    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getApproximate, = 0, LSST_makeBackground_getApproximate_types)
private:
    Background(Background const&);
    Background& operator=(Background const&);
    /**
     * Compute the centers, origins, and sizes of the patches used to compute image statistics
     * when estimating the Background
     */
    void _setCenOrigSize(int const width, int const height, int const nxSample, int const nySample);
};

/**
 * A class to evaluate %image background levels
 *
 * Break an image up into nx*ny sub-images and use a statistical to estimate the background levels in each
 * square.  Then use a user-specified or algorithm to estimate background at a given pixel coordinate.
 *
 * Methods are available to return the background at a point (inefficiently), or an entire background image.
 * BackgroundControl contains a public StatisticsControl member to allow user control of how the backgrounds
 * are computed.
 *
 *     math::BackgroundControl bctrl(7, 7);  // number of sub-image squares in {x,y}-dimensions
 *     bctrl.sctrl.setNumSigmaClip(5.0);     // use 5-sigma clipping for the sub-image means
 *     std::shared_ptr<math::Background> backobj = math::makeBackground(img, bctrl);
 *     // get a whole background image
 *     Image<PixelT> back = backobj->getImage<PixelT>(math::Interpolate::NATURAL_SPLINE);
 *
 * @deprecated
 * there is also
 *
 *     // get the background at a pixel at i_x,i_y
 *     double someValue = backobj.getPixel(math::Interpolate::LINEAR, i_x, i_y);
 */
class BackgroundMI : public Background {
public:
    template <typename ImageT>
    /**
     * Constructor for BackgroundMI
     *
     * Estimate the statistical properties of the Image in a grid of cells;  we'll later call
     * getImage() to interpolate those values, creating an image the same size as the original
     *
     * @param img ImageT (or MaskedImage) whose properties we want
     * @param bgCtrl Control how the BackgroundMI is estimated
     *
     * @note If there are heavily masked or Nan regions in the image we may not be able to estimate
     * all the cells in the "statsImage".  Interpolation will still work, but if you want to prevent
     * the code wildly extrapolating, it may be better to set the values directly; e.g.
     *
     *     defaultValue = 10
     *     statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
     *     sim = statsImage.getImage().getArray()
     *     sim[np.isnan(sim)] = defaultValue # replace NaN by defaultValue
     *     bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
     *
     * There is a ticket (#2825) to allow getImage to specify a default value to use when interpolation fails
     *
     * @deprecated The old and deprecated API specified the interpolation style as part of the
     * BackgroundControl
     * object passed to this ctor.  This is still supported, but the work isn't done until the getImage()
     * method is called
     */
    explicit BackgroundMI(ImageT const& img, BackgroundControl const& bgCtrl);
    /**
     * Recreate a BackgroundMI from the statsImage and the original Image's BBox
     *
     * @param imageDimensions unbinned Image's BBox
     * @param statsImage Internal stats image
     */
    explicit BackgroundMI(geom::Box2I const imageDimensions,
                          image::MaskedImage<InternalPixelT> const& statsImage);

    /**
     * Add a scalar to the Background (equivalent to adding a constant to the original image)
     *
     * @param delta Value to add
     */
    virtual BackgroundMI& operator+=(float const delta);
    /**
     * Subtract a scalar from the Background (equivalent to subtracting a constant from the original image)
     *
     * @param delta Value to subtract
     */
    virtual BackgroundMI& operator-=(float const delta);

    /**
     * Method to retrieve the background level at a pixel coord.
     *
     * @param style How to interpolate
     * @param x x-pixel coordinate (column)
     * @param y y-pixel coordinate (row)
     * @returns an estimated background at x,y (double)
     *
     * @deprecated Don't call this image (not even in test code).
     * This can be a very costly function to get a single pixel. If you want an image, use the getImage()
     * method.
     */
    double getPixel(Interpolate::Style const style, int const x, int const y) const;
    /**
     * Return the background value at a point
     *
     * @warning This is very inefficient -- only use it for debugging, if then.
     *
     * @deprecated New code should specify the interpolation style in getPixel, not the ctor
     */
    double getPixel(int const x, int const y) const { return getPixel(_bctrl->getInterpStyle(), x, y); }
    /**
     * Return the image of statistical quantities extracted from the image
     */
    lsst::afw::image::MaskedImage<InternalPixelT> getStatsImage() const { return _statsImage; }

private:
    lsst::afw::image::MaskedImage<InternalPixelT>
            _statsImage;  // statistical properties for the grid of subimages
    mutable std::vector<std::vector<double>> _gridColumns;  // interpolated columns for the bicubic spline

    void _setGridColumns(Interpolate::Style const interpStyle, UndersampleStyle const undersampleStyle,
                         int const iX, std::vector<int> const& ypix) const;

#if defined(LSST_makeBackground_getImage)
    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getImage, , LSST_makeBackground_getImage_types)
    BOOST_PP_SEQ_FOR_EACH(LSST_makeBackground_getApproximate, , LSST_makeBackground_getApproximate_types)
#if 0  // keep for use in Background instantiations
#undef LSST_makeBackground_getImage_types
#undef LSST_makeBackground_getApproximate_types
#endif
#undef LSST_makeBackground_getImage
#undef LSST_makeBackground_getApproximate
#endif
    // Here's the worker function for _getImage (non-virtual; it's templated in BackgroundMI, not Background)
    /**
     * Worker routine for getImage
     */
    template <typename PixelT>
    std::shared_ptr<image::Image<PixelT>> doGetImage(geom::Box2I const& bbox,
                                                     Interpolate::Style const interpStyle_,
                                                     UndersampleStyle const undersampleStyle) const;
    // and for _getApproximate
    template <typename PixelT>
    std::shared_ptr<Approximate<PixelT>> doGetApproximate(ApproximateControl const& actrl,
                                                          UndersampleStyle const undersampleStyle) const;
};
/**
 * A convenience function that uses function overloading to make the correct type of Background
 *
 * cf. std::make_pair()
 */
template <typename ImageT>
std::shared_ptr<Background> makeBackground(ImageT const& img, BackgroundControl const& bgCtrl) {
    return std::shared_ptr<Background>(new BackgroundMI(img, bgCtrl));
}
}
}
}  // lsst::afw::math

#endif  //   LSST_AFW_MATH_BACKGROUND_H
