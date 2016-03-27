// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

/**
 * @file Background.cc
 * @ingroup afw
 * @brief Background estimation class code
 * @author Steve Bickerton
 * @date Jan 26, 2009
 */
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>
#include "lsst/utils/ieee.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Approximate.h"
#include "lsst/afw/math/Background.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace ex = pex::exceptions;

namespace afw {
namespace math {

namespace {

    // Given two vectors x and y, with some nans in y we want vectors x' and y' that correspond to the data
    // without the nans basic idea is that 'x' is the values, and 'y' is the ref (where nan checking happens)
    //    cullNan(x, y, x', y')
    void cullNan(std::vector<double> const &values, std::vector<double> const &refs,
                 std::vector<double> &culledValues, std::vector<double> &culledRefs,
                 double const defaultValue=std::numeric_limits<double>::quiet_NaN()
                ) {
        if (culledValues.capacity() == 0) {
            culledValues.reserve(refs.size());
        } else {
            culledValues.clear();
        }
        if (culledRefs.capacity() == 0) {
            culledRefs.reserve(refs.size());
        } else {
            culledRefs.clear();
        }

        bool const haveDefault = !lsst::utils::isnan(defaultValue);

        for (std::vector<double>::const_iterator pVal = values.begin(), pRef = refs.begin();
             pRef != refs.end(); ++pRef, ++pVal) {
            if (!lsst::utils::isnan(*pRef)) {
                culledValues.push_back(*pVal);
                culledRefs.push_back(*pRef);
            } else if(haveDefault) {
                culledValues.push_back(*pVal);
                culledRefs.push_back(defaultValue);
            } else {
                ;                       // drop a NaN
            }
        }
    }
}

/**
 * @brief Constructor for BackgroundMI
 *
 * Estimate the statistical properties of the Image in a grid of cells;  we'll later call
 * getImage() to interpolate those values, creating an image the same size as the original
 *
 * \note If there are heavily masked or Nan regions in the image we may not be able to estimate
 * all the cells in the "statsImage".  Interpolation will still work, but if you want to prevent
 * the code wildly extrapolating, it may be better to set the values directly; e.g.
 * \code
 * defaultValue = 10
 * statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
 * sim = statsImage.getImage().getArray()
 * sim[np.isnan(sim)] = defaultValue # replace NaN by defaultValue
 * bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
 * \endcode
 * There is a ticket (#2825) to allow getImage to specify a default value to use when interpolation fails
 *
 * \deprecated The old and deprecated API specified the interpolation style as part of the BackgroundControl
 * object passed to this ctor.  This is still supported, but the work isn't done until the getImage()
 * method is called
 */
template<typename ImageT>
BackgroundMI::BackgroundMI(ImageT const& img, ///< ImageT (or MaskedImage) whose properties we want
                             BackgroundControl const& bgCtrl ///< Control how the BackgroundMI is estimated
                            ) :
    Background(img, bgCtrl), _statsImage(image::MaskedImage<InternalPixelT>())
{
    // =============================================================
    // Loop over the cells in the image, computing statistical properties
    // of each cell in turn and using them to set _statsImage
    int const nxSample = bgCtrl.getNxSample();
    int const nySample = bgCtrl.getNySample();
    _statsImage = image::MaskedImage<InternalPixelT>(nxSample, nySample);

    image::MaskedImage<InternalPixelT>::Image &im = *_statsImage.getImage();
    image::MaskedImage<InternalPixelT>::Variance &var = *_statsImage.getVariance();

    for (int iX = 0; iX < nxSample; ++iX) {
        for (int iY = 0; iY < nySample; ++iY) {
            ImageT subimg = ImageT(img, geom::Box2I(geom::Point2I(_xorig[iX], _yorig[iY]),
                                                    geom::Extent2I(_xsize[iX], _ysize[iY])), image::LOCAL);
            
            std::pair<double, double> res = makeStatistics(subimg, bgCtrl.getStatisticsProperty() | ERRORS,
                                                           *bgCtrl.getStatisticsControl()).getResult();
            im(iX, iY) = res.first;
            var(iX, iY) = res.second;
        }
    }
}
/**
 * Recreate a BackgroundMI from the statsImage and the original Image's BBox
 */
BackgroundMI::BackgroundMI(geom::Box2I const imageBBox,                         ///< unbinned Image's BBox
                           image::MaskedImage<InternalPixelT> const& statsImage ///< Internal stats image
                          ) :
    Background(imageBBox, statsImage.getWidth(), statsImage.getHeight()),
    _statsImage(statsImage)
{
}

void BackgroundMI::_setGridColumns(Interpolate::Style const interpStyle,
                                   UndersampleStyle const undersampleStyle,
                                   int const iX, std::vector<int> const& ypix) const
{
    image::MaskedImage<InternalPixelT>::Image &im = *_statsImage.getImage();

    int const height = _imgBBox.getHeight();
    _gridColumns[iX].resize(height);

    // Set _grid as a transitional measure
    std::vector<double> _grid(_statsImage.getHeight());
    std::copy(im.col_begin(iX), im.col_end(iX), _grid.begin());
    
    // remove nan from the grid values before computing columns
    // if we do it here (ie. in _setGridColumns), it should
    // take care of all future occurrences, so we don't need to do this elsewhere
    std::vector<double> ycenTmp, gridTmp;
    cullNan(_ycen, _grid, ycenTmp, gridTmp);
    
    PTR(Interpolate) intobj;
    try {
        intobj = makeInterpolate(ycenTmp, gridTmp, interpStyle);
    } catch(pex::exceptions::OutOfRangeError &e) {
        switch (undersampleStyle) {
          case THROW_EXCEPTION:
            LSST_EXCEPT_ADD(e, "setting _gridcolumns");
            throw;
          case REDUCE_INTERP_ORDER:
            {
                if (gridTmp.empty()) {
                    // Set the column to NaN.  We'll deal with this properly when interpolating in x
                    ycenTmp.push_back(0);
                    gridTmp.push_back(std::numeric_limits<double>::quiet_NaN());

                    intobj = makeInterpolate(ycenTmp, gridTmp, Interpolate::CONSTANT);
                    break;
                } else {
                    return _setGridColumns(lookupMaxInterpStyle(gridTmp.size()), undersampleStyle, iX, ypix);
                }
            }
          case INCREASE_NXNYSAMPLE:
            LSST_EXCEPT_ADD(e, "The BackgroundControl UndersampleStyle INCREASE_NXNYSAMPLE is not supported.");
            throw;
          default:
            LSST_EXCEPT_ADD(e, str(boost::format("The selected BackgroundControl "
                                                 "UndersampleStyle %d is not defined.") % undersampleStyle));
            throw;
        }
    } catch(ex::Exception &e) {
        LSST_EXCEPT_ADD(e, "setting _gridcolumns");
        throw;
    }
        
    for (int iY = 0; iY < height; ++iY) {
        _gridColumns[iX][iY] = intobj->interpolate(ypix[iY]);
    }
}

/**
 * @brief Add a scalar to the Background (equivalent to adding a constant to the original image)
 */
void BackgroundMI::operator+=(float const delta ///< Value to add
                                  )
{
    _statsImage += delta;
}

/**
 * @brief Subtract a scalar from the Background (equivalent to subtracting a constant from the original image)
 */
void BackgroundMI::operator-=(float const delta ///< Value to subtract
                                  )
{
    _statsImage -= delta;
}

/**
 * @brief Method to retrieve the background level at a pixel coord.
 *
 * @return an estimated background at x,y (double)
 *
 * \deprecated Don't call this image (not even in test code).
 * This can be a very costly function to get a single pixel. If you want an image, use the getImage() method.
 */
double BackgroundMI::getPixel(Interpolate::Style const interpStyle, ///< How to interpolate
                            int const x, ///< x-pixel coordinate (column)
                            int const y ///< y-pixel coordinate (row)
                           ) const
{
    (void)getImage<InternalPixelT>(interpStyle);        // setup the interpolation

    // build an interpobj along the row y and get the x'th value
    int const nxSample = _statsImage.getWidth();
    std::vector<double> bg_x(nxSample);
    for (int iX = 0; iX < nxSample; iX++) {
        bg_x[iX] = _gridColumns[iX][y];
    }
    std::vector<double> xcenTmp, bgTmp;
    cullNan(_xcen, bg_x, xcenTmp, bgTmp);

    try {
        PTR(Interpolate) intobj = makeInterpolate(xcenTmp, bgTmp, interpStyle);
        return static_cast<double>(intobj->interpolate(x));
    } catch(ex::Exception &e) {
        LSST_EXCEPT_ADD(e, "in getPixel()");
        throw;
    }
}
/*
 * Worker routine for getImage
 */
template<typename PixelT>
PTR(image::Image<PixelT>) BackgroundMI::doGetImage(
    geom::Box2I const& bbox,
        Interpolate::Style const interpStyle_,   // Style of the interpolation
        UndersampleStyle const undersampleStyle // Behaviour if there are too few points
                                                ) const
{
    if (!_imgBBox.contains(bbox)) {
        throw LSST_EXCEPT(ex::LengthError,
                          str(boost::format("BBox (%d:%d,%d:%d) out of range (%d:%d,%d:%d)") %
                              bbox.getMinX() % bbox.getMaxX() % bbox.getMinY() % bbox.getMaxY() %
                              _imgBBox.getMinX() % _imgBBox.getMaxX() %
                              _imgBBox.getMinY() % _imgBBox.getMaxY()));
    }
    int const nxSample = _statsImage.getWidth();
    int const nySample = _statsImage.getHeight();
    Interpolate::Style interpStyle = interpStyle_; // not const -- may be modified if REDUCE_INTERP_ORDER

    /*
     * Save the as-used interpStyle and undersampleStyle.
     *
     * N.b. The undersampleStyle may actually be overridden for some columns of the statsImage if they
     * have too few good values.  This doesn't prevent you reproducing the results of getImage() by
     * calling getImage(getInterpStyle(), getUndersampleStyle())
     */
    _asUsedInterpStyle = interpStyle;
    _asUsedUndersampleStyle = undersampleStyle;
    /*
     * Check if the requested nx,ny are sufficient for the requested interpolation style,
     * making suitable adjustments
     */
    bool const isXundersampled = (nxSample < lookupMinInterpPoints(interpStyle));
    bool const isYundersampled = (nySample < lookupMinInterpPoints(interpStyle));

    switch (undersampleStyle) {
      case THROW_EXCEPTION:
        if (isXundersampled && isYundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "nxSample and nySample have too few points for requested interpolation style.");
        } else if (isXundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "nxSample has too few points for requested interpolation style.");
        } else if (isYundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "nySample has too few points for requested interpolation style.");
        }
        break;
      case REDUCE_INTERP_ORDER:
        if (isXundersampled || isYundersampled) {
            Interpolate::Style const xStyle = lookupMaxInterpStyle(nxSample);
            Interpolate::Style const yStyle = lookupMaxInterpStyle(nySample);
            interpStyle = (nxSample < nySample) ? xStyle : yStyle;
            _asUsedInterpStyle = interpStyle;
        }
        break;
      case INCREASE_NXNYSAMPLE:
        if (isXundersampled || isYundersampled) {
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              "The BackgroundControl UndersampleStyle INCREASE_NXNYSAMPLE is not supported.");
        }
        break;
      default:
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          str(boost::format("The selected BackgroundControl "
                                            "UndersampleStyle %d is not defined.") % undersampleStyle));
    }

    // if we're approximating, don't bother with the rest of the interp-related work.  Return from here.
    if (_bctrl->getApproximateControl()->getStyle() != ApproximateControl::UNKNOWN) {
        return doGetApproximate<PixelT>(*_bctrl->getApproximateControl(), _asUsedUndersampleStyle)->getImage();
    }
    
    // =============================================================
    // --> We'll store nxSample fully-interpolated columns to interpolate the rows over
    // make a vector containing the y pixel coords for the column
    int const width = _imgBBox.getWidth();
    int const height = _imgBBox.getHeight();
    auto const bboxOff = bbox.getMin() - _imgBBox.getMin();

    std::vector<int> ypix(height);
    for (int iY = 0; iY < height; ++iY) {
        ypix[iY] = iY;
    }

    _gridColumns.resize(width);
    for (int iX = 0; iX < nxSample; ++iX) {
        _setGridColumns(interpStyle, undersampleStyle, iX, ypix);
    }

    // create a shared_ptr to put the background image in and return to caller
    // start with xy0 = 0 and set final xy0 later
    PTR(image::Image<PixelT>) bg =
        PTR(image::Image<PixelT>)(new image::Image<PixelT>(bbox.getDimensions()));

    // go through row by row
    // - interpolate on the gridcolumns that were pre-computed by the constructor
    // - copy the values to an ImageT to return to the caller.
    std::vector<double> xcenTmp, bgTmp;

    // N.b. There's no API to set defaultValue to other than NaN (due to issues with persistence
    // that I don't feel like fixing;  #2825).  If we want to address this, this is the place
    // to start, but note that NaN is treated specially -- it means, "Interpolate" so to allow
    // us to put a NaN into the outputs some changes will be needed
    double defaultValue = std::numeric_limits<double>::quiet_NaN();

    for (int y = 0, iY = bboxOff.getY(); y < bbox.getHeight(); ++y, ++iY) {
        // build an interp object for this row
        std::vector<double> bg_x(nxSample);
        for (int iX = 0; iX < nxSample; iX++) {
            bg_x[iX] = static_cast<double>(_gridColumns[iX][iY]);
        }
        cullNan(_xcen, bg_x, xcenTmp, bgTmp, defaultValue);

        PTR(Interpolate) intobj;
        try {
            intobj = makeInterpolate(xcenTmp, bgTmp, interpStyle);
        } catch(pex::exceptions::OutOfRangeError &e) {
            switch (undersampleStyle) {
              case THROW_EXCEPTION:
                LSST_EXCEPT_ADD(e, str(boost::format("Interpolating in y (iY = %d)") % iY));
                throw;
              case REDUCE_INTERP_ORDER:
                {
                    if (bgTmp.empty()) {
                        xcenTmp.push_back(0);
                        bgTmp.push_back(defaultValue);
                        
                        intobj = makeInterpolate(xcenTmp, bgTmp, Interpolate::CONSTANT);
                        break;
                    } else {
                        intobj = makeInterpolate(xcenTmp, bgTmp, lookupMaxInterpStyle(bgTmp.size()));
                    }
                }
                break;
              case INCREASE_NXNYSAMPLE:
                LSST_EXCEPT_ADD(e, "The BackgroundControl UndersampleStyle INCREASE_NXNYSAMPLE is not supported.");
                throw;
              default:
                LSST_EXCEPT_ADD(e, str(boost::format("The selected BackgroundControl "
                                                     "UndersampleStyle %d is not defined.") % undersampleStyle));
                throw;
            }
        } catch(ex::Exception &e) {
            LSST_EXCEPT_ADD(e, str(boost::format("Interpolating in y (iY = %d)") % iY));
            throw;
        }

        // fill the image with interpolated values
        for (int iX = bboxOff.getX(), x = 0; x < bbox.getWidth(); ++iX, ++x) {
            (*bg)(x, y) = static_cast<PixelT>(intobj->interpolate(iX));
        }
    }
    bg->setXY0(bbox.getMin());

    return bg;
}

/************************************************************************************************************/

template<typename PixelT>
PTR(Approximate<PixelT>) BackgroundMI::doGetApproximate(
        ApproximateControl const& actrl,                          /* Approximation style */
        UndersampleStyle const undersampleStyle                   /* Behaviour if there are too few points */
                                    ) const
{
    auto const localBBox = afw::geom::Box2I(afw::geom::Point2I(0, 0), _imgBBox.getDimensions());
    return makeApproximate(_xcen, _ycen, _statsImage, localBBox, actrl);
}

/// \cond
/*
 * Create the versions we need of _get{Approximate,Image} and Explicit instantiations
 *
 */
#define CREATE_BACKGROUND(m, v, TYPE)                              \
    template BackgroundMI::BackgroundMI(image::Image<TYPE> const& img, \
                                          BackgroundControl const& bgCtrl); \
    template BackgroundMI::BackgroundMI(image::MaskedImage<TYPE> const& img, \
                                          BackgroundControl const& bgCtrl); \
    PTR(image::Image<TYPE>)                                     \
    BackgroundMI::_getImage(                                            \
        geom::Box2I const& bbox, \
        Interpolate::Style const interpStyle,                    /* Style of the interpolation */ \
        UndersampleStyle const undersampleStyle,                 /* Behaviour if there are too few points */ \
        TYPE                                                     /* disambiguate */    \
                         ) const                                        \
    {                                                                   \
        return BackgroundMI::doGetImage<TYPE>(bbox, interpStyle, undersampleStyle); \
    }

#define CREATE_getApproximate(m, v, TYPE)                               \
PTR(Approximate<TYPE>) BackgroundMI::_getApproximate(                   \
        ApproximateControl const& actrl,                         /* Approximation style */ \
        UndersampleStyle const undersampleStyle,                 /* Behaviour if there are too few points */ \
        TYPE                                                     /* disambiguate */ \
                                               ) const                  \
    {                                                                   \
        return BackgroundMI::doGetApproximate<TYPE>(actrl, undersampleStyle); \
    }

BOOST_PP_SEQ_FOR_EACH(CREATE_BACKGROUND, , LSST_makeBackground_getImage_types)
BOOST_PP_SEQ_FOR_EACH(CREATE_getApproximate, , LSST_makeBackground_getApproximate_types)

/// \endcond
}}} // lsst::afw::math
