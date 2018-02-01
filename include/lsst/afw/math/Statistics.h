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

#if !defined(LSST_AFW_MATH_STATISTICS_H)
#define LSST_AFW_MATH_STATISTICS_H
/**
 * Compute Image Statistics
 *
 * @note The Statistics class itself can only handle lsst::afw::image::MaskedImage() types.
 *       The philosophy has been to handle other types by making them look like
 *       lsst::afw::image::MaskedImage() and reusing that code.
 *       Users should have no need to instantiate a Statistics object directly,
 *       but should use the overloaded makeStatistics() factory functions.
 */

#include <algorithm>
#include <cassert>
#include <limits>
#include "boost/iterator/iterator_adaptor.hpp"
#include "boost/tuple/tuple.hpp"
#include <memory>
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/MaskedVector.h"

namespace lsst {
namespace afw {
namespace image {
template <typename>
class Image;
template <typename, typename, typename>
class MaskedImage;
}
namespace math {
template <typename>
class MaskedVector;  // forward declaration

typedef lsst::afw::image::VariancePixel WeightPixel;  // Type used for weights

/**
 * control what is calculated
 */
enum Property {
    NOTHING = 0x0,         ///< We don't want anything
    ERRORS = 0x1,          ///< Include errors of requested quantities
    NPOINT = 0x2,          ///< number of sample points
    MEAN = 0x4,            ///< estimate sample mean
    STDEV = 0x8,           ///< estimate sample standard deviation
    VARIANCE = 0x10,       ///< estimate sample variance
    MEDIAN = 0x20,         ///< estimate sample median
    IQRANGE = 0x40,        ///< estimate sample inter-quartile range
    MEANCLIP = 0x80,       ///< estimate sample N-sigma clipped mean (N set in StatisticsControl, default=3)
    STDEVCLIP = 0x100,     ///< estimate sample N-sigma clipped stdev (N set in StatisticsControl, default=3)
    VARIANCECLIP = 0x200,  ///< estimate sample N-sigma clipped variance
                           ///<  (N set in StatisticsControl, default=3)
    MIN = 0x400,           ///< estimate sample minimum
    MAX = 0x800,           ///< estimate sample maximum
    SUM = 0x1000,          ///< find sum of pixels in the image
    MEANSQUARE = 0x2000,   ///< find mean value of square of pixel values
    ORMASK = 0x4000,       ///< get the or-mask of all pixels used.
    NCLIPPED = 0x8000,     ///< number of clipped points
    NMASKED = 0x10000      ///< number of masked points
};
/// Conversion function to switch a string to a Property (see Statistics.h)
Property stringToStatisticsProperty(std::string const property);

/**
 * Pass parameters to a Statistics object
 *
 * A class to pass parameters which control how the stats are calculated.
 *
 */
class StatisticsControl {
public:
    enum WeightsBoolean { WEIGHTS_FALSE = 0, WEIGHTS_TRUE = 1, WEIGHTS_NONE };  // initial state is NONE

    StatisticsControl(double numSigmaClip = 3.0,  ///< number of standard deviations to clip at
                      int numIter = 3,            ///< Number of iterations
                      lsst::afw::image::MaskPixel andMask =
                              0x0,  ///< and-Mask: defines which mask bits cause a value to be ignored
                      bool isNanSafe = true,  ///< flag NaNs & Infs
                      WeightsBoolean useWeights =
                              WEIGHTS_NONE  ///< use weighted statistics (via a vector or an inverse variance)
                      )
            : _numSigmaClip(numSigmaClip),
              _numIter(numIter),
              _andMask(andMask),
              _noGoodPixelsMask(0x0),
              _isNanSafe(isNanSafe),
              _useWeights(useWeights),
              _calcErrorFromInputVariance(false),
              _maskPropagationThresholds() {
        try {
            _noGoodPixelsMask = lsst::afw::image::Mask<>::getPlaneBitMask("NO_DATA");
        } catch (lsst::pex::exceptions::InvalidParameterError) {
            ;  // Mask has no NO_DATA plane defined
        }

        assert(_numSigmaClip > 0);
        assert(_numIter > 0);
    }

    //@{
    /**  When pixels with the given bit are rejected, we count what fraction the rejected
     *   pixels would have contributed (including the weights, if any) if those pixels had
     *   not been rejected, and set that bit in the return value of Statistics::getOrMask()
     *   if it exceeds the given threshold.
     */
    double getMaskPropagationThreshold(int bit) const;
    void setMaskPropagationThreshold(int bit, double threshold);
    //@}

    double getNumSigmaClip() const { return _numSigmaClip; }
    int getNumIter() const { return _numIter; }
    int getAndMask() const { return _andMask; }
    int getNoGoodPixelsMask() const { return _noGoodPixelsMask; }
    bool getNanSafe() const { return _isNanSafe; }
    bool getWeighted() const { return _useWeights == WEIGHTS_TRUE ? true : false; }
    bool getWeightedIsSet() const { return _useWeights != WEIGHTS_NONE ? true : false; }
    bool getCalcErrorFromInputVariance() const { return _calcErrorFromInputVariance; }

    void setNumSigmaClip(double numSigmaClip) {
        assert(numSigmaClip > 0);
        _numSigmaClip = numSigmaClip;
    }
    void setNumIter(int numIter) {
        assert(numIter > 0);
        _numIter = numIter;
    }
    void setAndMask(int andMask) { _andMask = andMask; }
    void setNoGoodPixelsMask(int noGoodPixelsMask) { _noGoodPixelsMask = noGoodPixelsMask; }
    void setNanSafe(bool isNanSafe) { _isNanSafe = isNanSafe; }
    void setWeighted(bool useWeights) { _useWeights = useWeights ? WEIGHTS_TRUE : WEIGHTS_FALSE; }
    void setCalcErrorFromInputVariance(bool calcErrorFromInputVariance) {
        _calcErrorFromInputVariance = calcErrorFromInputVariance;
    }

private:
    friend class Statistics;

    double _numSigmaClip;              // Number of standard deviations to clip at
    int _numIter;                      // Number of iterations
    int _andMask;                      // and-Mask to specify which mask planes to ignore
    int _noGoodPixelsMask;             // mask to set if no values are acceptable
    bool _isNanSafe;                   // Check for NaNs & Infs before running (slower)
    WeightsBoolean _useWeights;        // Calculate weighted statistics (enum because of 3-valued logic)
    bool _calcErrorFromInputVariance;  // Calculate errors from the input variances, if available
    std::vector<double> _maskPropagationThresholds;  // Thresholds for when to propagate mask bits,
                                                     // treated like a dict (unset bits are set to 1.0)
};

/**
 * @ingroup afw
 *
 * A class to evaluate %image statistics
 *
 * The basic strategy is to construct a Statistics object from an Image and
 * a statement of what we want to know.  The desired results can then be
 * returned using Statistics methods.  A StatisticsControl object is used to
 * pass parameters.  The statistics currently implemented are listed in the
 * enum Properties in Statistics.h.
 *
 *
 *      // sets NumSigclip (3.0), and NumIter (3) for clipping
 *      lsst::afw::math::StatisticsControl sctrl(3.0, 3);
 *
 *      sctrl.setNumSigmaClip(4.0);            // reset number of standard deviations for N-sigma clipping
 *      sctrl.setNumIter(5);                   // reset number of iterations for N-sigma clipping
 *      sctrl.setAndMask(0x1);                 // ignore pixels with these mask bits set
 *      sctrl.setNanSafe(true);                // check for NaNs & Infs, a bit slower (default=true)
 *
 *      lsst::afw::math::Statistics statobj =
 *          lsst::afw::math::makeStatistics(*img, afwMath::NPOINT |
 *                                                afwMath::MEAN | afwMath::MEANCLIP, sctrl);
 *      double const n = statobj.getValue(lsst::afw::math::NPOINT);
 *      std::pair<double, double> const mean =
 *                                       statobj.getResult(lsst::afw::math::MEAN); // Returns (value, error)
 *      double const meanError = statobj.getError(lsst::afw::math::MEAN);                // just the error
 *
 *
 * @note Factory function: We used a helper function, `makeStatistics`, rather that the constructor
 *       directly so that the compiler could deduce the types -- cf. `std::make_pair()`
 *
 * @note Inputs: The class Statistics is templated, and makeStatistics() can take either:
 *       (1) an image, (2) a maskedImage, or (3) a std::vector<>
 *       Overloaded makeStatistics() functions then wrap what they were passed in Image/Mask-like classes
 *       and call the Statistics constructor.
 * @note Clipping: The clipping is done iteratively with numSigmaClip and numIter specified in
 *       the StatisticsControl object.  The first clip (ie. the first iteration) is performed at:
 *       median +/- numSigmaClip*IQ_TO_STDEV*IQR, where IQ_TO_STDEV=~0.74 is the conversion factor
 *       between the IQR and sigma for a Gaussian distribution.  All subsequent iterations perform
 *       clips at mean +/- numSigmaClip*stdev.
 *
 */
class Statistics {
public:
    /// The type used to report (value, error) for desired statistics
    typedef std::pair<double, double> Value;

    /**
     * Constructor for Statistics object
     *
     * @param img Image whose properties we want
     * @param msk Mask to control which pixels are included
     * @param var Variances corresponding to values in Image
     * @param flags Describe what we want to calculate
     * @param sctrl Control how things are calculated
     *
     * @note Most of the actual work is done in this constructor; the results
     * are retrieved using `getValue` etc.
     */
    template <typename ImageT, typename MaskT, typename VarianceT>
    explicit Statistics(ImageT const &img, MaskT const &msk, VarianceT const &var, int const flags,
                        StatisticsControl const &sctrl = StatisticsControl());

    /**
     * @param img Image whose properties we want
     * @param msk Mask to control which pixels are included
     * @param var Variances corresponding to values in Image
     * @param weights Weights to use corresponding to values in Image
     * @param flags Describe what we want to calculate
     * @param sctrl Control how things are calculated
     */
    template <typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
    explicit Statistics(ImageT const &img, MaskT const &msk, VarianceT const &var, WeightT const &weights,
                        int const flags, StatisticsControl const &sctrl = StatisticsControl());

    Statistics(Statistics const &) = default;
    Statistics(Statistics &&) = default;
    Statistics & operator=(Statistics const &) = default;
    Statistics & operator=(Statistics &&) = default;
    ~Statistics() = default;

    /** Return the value and error in the specified statistic (e.g. MEAN)
     *
     * @param prop the afw::math::Property to retrieve. If NOTHING (default) and you only asked for
     *             one property (and maybe its error) in the constructor, that property is returned
     *
     * @note Only quantities requested in the constructor may be retrieved; in particular
     * errors may not be available if you didn't specify ERROR in the constructor
     *
     * @see getValue and getError
     *
     * @todo uncertainties on MEANCLIP,STDEVCLIP are sketchy.  _n != _nClip
     *
     */
    Value getResult(Property const prop = NOTHING) const;

    /** Return the error in the desired property (if specified in the constructor)
     *
     * @param prop the afw::math::Property to retrieve. If NOTHING (default) and you only
     *             asked for one property in the constructor, that property's error is returned
     *
     * @note You may have needed to specify ERROR to the ctor
     */
    double getError(Property const prop = NOTHING) const;
    /** Return the value of the desired property (if specified in the constructor)
     *
     * @param prop the afw::math::Property to retrieve. If NOTHING (default) and you only
     *             asked for one property in the constructor, that property is returned
     */
    double getValue(Property const prop = NOTHING) const;
    lsst::afw::image::MaskPixel getOrMask() const { return _allPixelOrMask; }

private:
    long _flags;  // The desired calculation

    int _n;                                       // number of pixels in the image
    Value _mean;                                  // the image's mean
    Value _variance;                              // the image's variance
    double _min;                                  // the image's minimum
    double _max;                                  // the image's maximum
    double _sum;                                  // the sum of all the image's pixels
    Value _meanclip;                              // the image's N-sigma clipped mean
    Value _varianceclip;                          // the image's N-sigma clipped variance
    Value _median;                                // the image's median
    int _nClipped;                                // number of pixels clipped
    int _nMasked;                                 // number of pixels masked
    double _iqrange;                              // the image's interquartile range
    lsst::afw::image::MaskPixel _allPixelOrMask;  //  the 'or' of all masked pixels

    StatisticsControl _sctrl;        // the control structure
    bool _weightsAreMultiplicative;  // Multiply by weights rather than dividing by them

    /**
     * @param img Image whose properties we want
     * @param msk Mask to control which pixels are included
     * @param var Variances corresponding to values in Image
     * @param weights Weights to use corresponding to values in Image
     * @param flags Describe what we want to calculate
     * @param sctrl Control how things are calculated
     */
    template <typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
    void doStatistics(ImageT const &img, MaskT const &msk, VarianceT const &var, WeightT const &weights,
                      int const flags, StatisticsControl const &sctrl);
};

/* ************************************  The factory functions ********************************* */
/**
 * @brief This iterator will never increment.  It is returned by row_begin() in the MaskImposter class
 *        (below) to allow phony mask pixels to be iterated over for non-mask images within Statistics.
 */
template <typename ValueT>
class infinite_iterator : public boost::iterator_adaptor<infinite_iterator<ValueT>, const ValueT *,
                                                         const ValueT, boost::forward_traversal_tag> {
public:
    infinite_iterator() : infinite_iterator::iterator_adaptor_(0) {}
    explicit infinite_iterator(const ValueT *p) : infinite_iterator::iterator_adaptor_(p) {}

private:
    friend class boost::iterator_core_access;
    void increment() { ; }  // never actually advance the iterator
};
/**
 * @brief A Mask wrapper to provide an infinite_iterator for Mask::row_begin().  This allows a fake
 *        Mask to be passed in to Statistics with a regular (non-masked) Image.
 */
template <typename ValueT>
class MaskImposter {
public:
    typedef infinite_iterator<ValueT> x_iterator;
    explicit MaskImposter(ValueT val = 0) { _val[0] = val; }
    x_iterator row_begin(int) const { return x_iterator(_val); }

private:
    ValueT _val[1];
};

/**
 * Handle a watered-down front-end to the constructor (no variance)
 * @relatesalso Statistics
 */
template <typename Pixel>
Statistics makeStatistics(lsst::afw::image::Image<Pixel> const &img,
                          lsst::afw::image::Mask<image::MaskPixel> const &msk, int const flags,
                          StatisticsControl const &sctrl = StatisticsControl()) {
    MaskImposter<WeightPixel> var;
    return Statistics(img, msk, var, flags, sctrl);
}

/**
 * Handle a straight front-end to the constructor
 * @relatesalso Statistics
 */
template <typename ImageT, typename MaskT, typename VarianceT>
Statistics makeStatistics(ImageT const &img, MaskT const &msk, VarianceT const &var, int const flags,
                          StatisticsControl const &sctrl = StatisticsControl()) {
    return Statistics(img, msk, var, flags, sctrl);
}

/**
 * Handle MaskedImages, just pass the getImage() and getMask() values right on through.
 * @relatesalso Statistics
 */
template <typename Pixel>
Statistics makeStatistics(lsst::afw::image::MaskedImage<Pixel> const &mimg, int const flags,
                          StatisticsControl const &sctrl = StatisticsControl()) {
    if (sctrl.getWeighted() || sctrl.getCalcErrorFromInputVariance()) {
        return Statistics(*mimg.getImage(), *mimg.getMask(), *mimg.getVariance(), flags, sctrl);
    } else {
        MaskImposter<WeightPixel> var;
        return Statistics(*mimg.getImage(), *mimg.getMask(), var, flags, sctrl);
    }
}

/**
 * Handle MaskedImages, just pass the getImage() and getMask() values right on through.
 * @relatesalso Statistics
 */
template <typename Pixel>
Statistics makeStatistics(lsst::afw::image::MaskedImage<Pixel> const &mimg,
                          lsst::afw::image::Image<WeightPixel> const &weights, int const flags,
                          StatisticsControl const &sctrl = StatisticsControl()) {
    if (sctrl.getWeighted() || sctrl.getCalcErrorFromInputVariance() ||
        (!sctrl.getWeightedIsSet() && (weights.getWidth() != 0 && weights.getHeight() != 0))) {
        return Statistics(*mimg.getImage(), *mimg.getMask(), *mimg.getVariance(), weights, flags, sctrl);
    } else {
        MaskImposter<WeightPixel> var;
        return Statistics(*mimg.getImage(), *mimg.getMask(), var, weights, flags, sctrl);
    }
}

/**
 * Specialization to handle Masks
 *
 * @param msk Image (or MaskedImage) whose properties we want
 * @param flags Describe what we want to calculate
 * @param sctrl Control how things are calculated
 *
 * @relatesalso Statistics
 */
Statistics makeStatistics(lsst::afw::image::Mask<lsst::afw::image::MaskPixel> const &msk, int const flags,
                          StatisticsControl const &sctrl = StatisticsControl());

/**
 * The makeStatistics() overload to handle regular (non-masked) Images
 * @relatesalso Statistics
 */
template <typename Pixel>
Statistics makeStatistics(
        lsst::afw::image::Image<Pixel> const &img,            ///< Image (or Image) whose properties we want
        int const flags,                                      ///< Describe what we want to calculate
        StatisticsControl const &sctrl = StatisticsControl()  ///< Control calculation
        ) {
    // make a phony mask that will be compiled out
    MaskImposter<lsst::afw::image::MaskPixel> const msk;
    MaskImposter<WeightPixel> const var;
    return Statistics(img, msk, var, flags, sctrl);
}

/**
 * @brief A vector wrapper to provide a vector with the necessary methods and typedefs to
 *        be processed by Statistics as though it were an Image.
 */
template <typename ValueT>
class ImageImposter {
public:
    // types we'll use in Statistics
    typedef typename std::vector<ValueT>::const_iterator x_iterator;
    typedef typename std::vector<ValueT>::const_iterator fast_iterator;
    typedef ValueT Pixel;

    // constructors for std::vector<>, and copy constructor
    // These are both shallow! ... no actual copying of values
    explicit ImageImposter(std::vector<ValueT> const &v) : _v(v) {}
    explicit ImageImposter(ImageImposter<ValueT> const &img) : _v(img._getVector()) {}

    // The methods we'll use in Statistics
    x_iterator row_begin(int) const { return _v.begin(); }
    x_iterator row_end(int) const { return _v.end(); }
    int getWidth() const { return _v.size(); }
    int getHeight() const { return 1; }
    afw::geom::Extent2I getDimensions() const { return afw::geom::Extent2I(getWidth(), getHeight()); }

    bool empty() const { return _v.empty(); }

private:
    std::vector<ValueT> const &_v;                                // a private reference to the data
    std::vector<ValueT> const &_getVector() const { return _v; }  // get the ref for the copyCon
};

/**
 * The makeStatistics() overload to handle std::vector<>
 * @relatesalso Statistics
 */
template <typename EntryT>
Statistics makeStatistics(std::vector<EntryT> const &v,  ///< Image (or MaskedImage) whose properties we want
                          int const flags,               ///< Describe what we want to calculate
                          StatisticsControl const &sctrl = StatisticsControl()  ///< Control calculation
                          ) {
    ImageImposter<EntryT> img(v);                   // wrap the vector in a fake image
    MaskImposter<lsst::afw::image::MaskPixel> msk;  // instantiate a fake mask that will be compiled out.
    MaskImposter<WeightPixel> var;
    return Statistics(img, msk, var, flags, sctrl);
}

/**
 * The makeStatistics() overload to handle std::vector<>
 * @relatesalso Statistics
 */
template <typename EntryT>
Statistics makeStatistics(std::vector<EntryT> const &v,  ///< Image (or MaskedImage) whose properties we want
                          std::vector<WeightPixel> const &vweights,  ///< Weights
                          int const flags,                           ///< Describe what we want to calculate
                          StatisticsControl const &sctrl = StatisticsControl()  ///< Control calculation
                          ) {
    ImageImposter<EntryT> img(v);                   // wrap the vector in a fake image
    MaskImposter<lsst::afw::image::MaskPixel> msk;  // instantiate a fake mask that will be compiled out.
    MaskImposter<WeightPixel> var;

    ImageImposter<WeightPixel> weights(vweights);

    return Statistics(img, msk, var, weights, flags, sctrl);
}

/**
 * The makeStatistics() overload to handle lsst::afw::math::MaskedVector<>
 * @relatesalso Statistics
 */
template <typename EntryT>
Statistics makeStatistics(lsst::afw::math::MaskedVector<EntryT> const &mv,  ///< MaskedVector
                          int const flags,  ///< Describe what we want to calculate
                          StatisticsControl const &sctrl = StatisticsControl()  ///< Control calculation
                          ) {
    if (sctrl.getWeighted() || sctrl.getCalcErrorFromInputVariance()) {
        return Statistics(*mv.getImage(), *mv.getMask(), *mv.getVariance(), flags, sctrl);
    } else {
        MaskImposter<WeightPixel> var;
        return Statistics(*mv.getImage(), *mv.getMask(), var, flags, sctrl);
    }
}

/**
 * The makeStatistics() overload to handle lsst::afw::math::MaskedVector<>
 * @relatesalso Statistics
 */
template <typename EntryT>
Statistics makeStatistics(lsst::afw::math::MaskedVector<EntryT> const &mv,  ///< MaskedVector
                          std::vector<WeightPixel> const &vweights,         ///< weights
                          int const flags,  ///< Describe what we want to calculate
                          StatisticsControl const &sctrl = StatisticsControl()  ///< Control calculation
                          ) {
    ImageImposter<WeightPixel> weights(vweights);

    if (sctrl.getWeighted() || sctrl.getCalcErrorFromInputVariance()) {
        return Statistics(*mv.getImage(), *mv.getMask(), *mv.getVariance(), weights, flags, sctrl);
    } else {
        MaskImposter<WeightPixel> var;
        return Statistics(*mv.getImage(), *mv.getMask(), var, weights, flags, sctrl);
    }
}
}
}
}

#endif
