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

/*
 * Background estimation class code
 */
#include <limits>
#include <vector>
#include <cmath>
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
void cullNan(std::vector<double> const& values, std::vector<double> const& refs,
             std::vector<double>& culledValues, std::vector<double>& culledRefs,
             double const defaultValue = std::numeric_limits<double>::quiet_NaN()) {
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

    bool const haveDefault = !std::isnan(defaultValue);

    for (std::vector<double>::const_iterator pVal = values.begin(), pRef = refs.begin(); pRef != refs.end();
         ++pRef, ++pVal) {
        if (!std::isnan(*pRef)) {
            culledValues.push_back(*pVal);
            culledRefs.push_back(*pRef);
        } else if (haveDefault) {
            culledValues.push_back(*pVal);
            culledRefs.push_back(defaultValue);
        } else {
            ;  // drop a NaN
        }
    }
}
}  // namespace

template <typename ImageT>
BackgroundMI::BackgroundMI(ImageT const& img, BackgroundControl const& bgCtrl)
        : Background(img, bgCtrl), _statsImage(image::MaskedImage<InternalPixelT>()) {
    // =============================================================
    // Loop over the cells in the image, computing statistical properties
    // of each cell in turn and using them to set _statsImage
    int const nxSample = bgCtrl.getNxSample();
    int const nySample = bgCtrl.getNySample();
    _statsImage = image::MaskedImage<InternalPixelT>(nxSample, nySample);

    image::MaskedImage<InternalPixelT>::Image& im = *_statsImage.getImage();
    image::MaskedImage<InternalPixelT>::Variance& var = *_statsImage.getVariance();

    for (int iX = 0; iX < nxSample; ++iX) {
        for (int iY = 0; iY < nySample; ++iY) {
            ImageT subimg = ImageT(img,
                                   lsst::geom::Box2I(lsst::geom::Point2I(_xorig[iX], _yorig[iY]),
                                                     lsst::geom::Extent2I(_xsize[iX], _ysize[iY])),
                                   image::LOCAL);

            std::pair<double, double> res = makeStatistics(subimg, bgCtrl.getStatisticsProperty() | ERRORS,
                                                           *bgCtrl.getStatisticsControl())
                                                    .getResult();
            im(iX, iY) = res.first;
            var(iX, iY) = res.second;
        }
    }
}
BackgroundMI::BackgroundMI(lsst::geom::Box2I const imageBBox,
                           image::MaskedImage<InternalPixelT> const& statsImage)
        : Background(imageBBox, statsImage.getWidth(), statsImage.getHeight()), _statsImage(statsImage) {}

void BackgroundMI::_setGridColumns(Interpolate::Style const interpStyle,
                                   UndersampleStyle const undersampleStyle, int const iX,
                                   std::vector<int> const& ypix) const {
    image::MaskedImage<InternalPixelT>::Image& im = *_statsImage.getImage();

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

    std::shared_ptr<Interpolate> intobj;
    try {
        intobj = makeInterpolate(ycenTmp, gridTmp, interpStyle);
    } catch (pex::exceptions::OutOfRangeError& e) {
        switch (undersampleStyle) {
            case THROW_EXCEPTION:
                LSST_EXCEPT_ADD(e, "setting _gridcolumns");
                throw;
            case REDUCE_INTERP_ORDER: {
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
                LSST_EXCEPT_ADD(
                        e, "The BackgroundControl UndersampleStyle INCREASE_NXNYSAMPLE is not supported.");
                throw;
            default:
                LSST_EXCEPT_ADD(e, str(boost::format("The selected BackgroundControl "
                                                     "UndersampleStyle %d is not defined.") %
                                       undersampleStyle));
                throw;
        }
    } catch (ex::Exception& e) {
        LSST_EXCEPT_ADD(e, "setting _gridcolumns");
        throw;
    }

    for (int iY = 0; iY < height; ++iY) {
        _gridColumns[iX][iY] = intobj->interpolate(ypix[iY]);
    }
}

BackgroundMI& BackgroundMI::operator+=(float const delta) {
    _statsImage += delta;
    return *this;
}

BackgroundMI& BackgroundMI::operator-=(float const delta) {
    _statsImage -= delta;
    return *this;
}

ndarray::Array<double, 1, 1> BackgroundMI::getBinCentersX() const {
    ndarray::Array<double, 1, 1> result = ndarray::allocate(_xcen.size());
    std::copy(_xcen.begin(), _xcen.end(), result.begin());
    result.deep() += _imgBBox.getMinX();
    return result;
}

ndarray::Array<double, 1, 1> BackgroundMI::getBinCentersY() const {
    ndarray::Array<double, 1, 1> result = ndarray::allocate(_ycen.size());
    std::copy(_ycen.begin(), _ycen.end(), result.begin());
    result.deep() += _imgBBox.getMinY();
    return result;
}

template <typename PixelT>
std::shared_ptr<image::Image<PixelT>> BackgroundMI::doGetImage(
        lsst::geom::Box2I const& bbox,
        Interpolate::Style const interpStyle_,   // Style of the interpolation
        UndersampleStyle const undersampleStyle  // Behaviour if there are too few points
        ) const {
    if (!_imgBBox.contains(bbox)) {
        throw LSST_EXCEPT(
                ex::LengthError,
                str(boost::format("BBox (%d:%d,%d:%d) out of range (%d:%d,%d:%d)") % bbox.getMinX() %
                    bbox.getMaxX() % bbox.getMinY() % bbox.getMaxY() % _imgBBox.getMinX() %
                    _imgBBox.getMaxX() % _imgBBox.getMinY() % _imgBBox.getMaxY()));
    }
    int const nxSample = _statsImage.getWidth();
    int const nySample = _statsImage.getHeight();
    Interpolate::Style interpStyle = interpStyle_;  // not const -- may be modified if REDUCE_INTERP_ORDER

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
                throw LSST_EXCEPT(
                        ex::InvalidParameterError,
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
                throw LSST_EXCEPT(
                        ex::InvalidParameterError,
                        "The BackgroundControl UndersampleStyle INCREASE_NXNYSAMPLE is not supported.");
            }
            break;
        default:
            throw LSST_EXCEPT(ex::InvalidParameterError,
                              str(boost::format("The selected BackgroundControl "
                                                "UndersampleStyle %d is not defined.") %
                                  undersampleStyle));
    }

    // if we're approximating, don't bother with the rest of the interp-related work.  Return from here.
    if (_bctrl->getApproximateControl()->getStyle() != ApproximateControl::UNKNOWN) {
        return doGetApproximate<PixelT>(*_bctrl->getApproximateControl(), _asUsedUndersampleStyle)
                ->getImage();
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
    std::shared_ptr<image::Image<PixelT>> bg =
            std::shared_ptr<image::Image<PixelT>>(new image::Image<PixelT>(bbox.getDimensions()));

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

        std::shared_ptr<Interpolate> intobj;
        try {
            intobj = makeInterpolate(xcenTmp, bgTmp, interpStyle);
        } catch (pex::exceptions::OutOfRangeError& e) {
            switch (undersampleStyle) {
                case THROW_EXCEPTION:
                    LSST_EXCEPT_ADD(e, str(boost::format("Interpolating in y (iY = %d)") % iY));
                    throw;
                case REDUCE_INTERP_ORDER: {
                    if (bgTmp.empty()) {
                        xcenTmp.push_back(0);
                        bgTmp.push_back(defaultValue);

                        intobj = makeInterpolate(xcenTmp, bgTmp, Interpolate::CONSTANT);
                        break;
                    } else {
                        intobj = makeInterpolate(xcenTmp, bgTmp, lookupMaxInterpStyle(bgTmp.size()));
                    }
                } break;
                case INCREASE_NXNYSAMPLE:
                    LSST_EXCEPT_ADD(
                            e,
                            "The BackgroundControl UndersampleStyle INCREASE_NXNYSAMPLE is not supported.");
                    throw;
                default:
                    LSST_EXCEPT_ADD(e, str(boost::format("The selected BackgroundControl "
                                                         "UndersampleStyle %d is not defined.") %
                                           undersampleStyle));
                    throw;
            }
        } catch (ex::Exception& e) {
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

template <typename PixelT>
std::shared_ptr<Approximate<PixelT>> BackgroundMI::doGetApproximate(
        ApproximateControl const& actrl,        /* Approximation style */
        UndersampleStyle const undersampleStyle /* Behaviour if there are too few points */
        ) const {
    auto const localBBox = lsst::geom::Box2I(lsst::geom::Point2I(0, 0), _imgBBox.getDimensions());
    return makeApproximate(_xcen, _ycen, _statsImage, localBBox, actrl);
}

/// @cond
/*
 * Create the versions we need of _get{Approximate,Image} and Explicit instantiations
 *
 */
#define CREATE_BACKGROUND(m, v, TYPE)                                                                    \
    template BackgroundMI::BackgroundMI(image::Image<TYPE> const& img, BackgroundControl const& bgCtrl); \
    template BackgroundMI::BackgroundMI(image::MaskedImage<TYPE> const& img,                             \
                                        BackgroundControl const& bgCtrl);                                \
    std::shared_ptr<image::Image<TYPE>> BackgroundMI::_getImage(                                         \
            lsst::geom::Box2I const& bbox,                                                               \
            Interpolate::Style const interpStyle,    /* Style of the interpolation */                    \
            UndersampleStyle const undersampleStyle, /* Behaviour if there are too few points */         \
            TYPE                                     /* disambiguate */                                  \
            ) const {                                                                                    \
        return BackgroundMI::doGetImage<TYPE>(bbox, interpStyle, undersampleStyle);                      \
    }

#define CREATE_getApproximate(m, v, TYPE)                                                        \
    std::shared_ptr<Approximate<TYPE>> BackgroundMI::_getApproximate(                            \
            ApproximateControl const& actrl,         /* Approximation style */                   \
            UndersampleStyle const undersampleStyle, /* Behaviour if there are too few points */ \
            TYPE                                     /* disambiguate */                          \
            ) const {                                                                            \
        return BackgroundMI::doGetApproximate<TYPE>(actrl, undersampleStyle);                    \
    }

BOOST_PP_SEQ_FOR_EACH(CREATE_BACKGROUND, , LSST_makeBackground_getImage_types)
BOOST_PP_SEQ_FOR_EACH(CREATE_getApproximate, , LSST_makeBackground_getApproximate_types)

/// @endcond
}  // namespace math
}  // namespace afw
}  // namespace lsst
