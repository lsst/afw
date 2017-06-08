/*
 * LSST Data Management System
 * Copyright 2008-2017 LSST Corporation.
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

#include <vector>

#include "lsst/afw/math/NewStatistics.h"
#include "lsst/pex/exceptions/Exception.h"

namespace lsst {
namespace afw {
namespace math {

double const NaN = std::numeric_limits<double>::quiet_NaN();
double const MAX_DOUBLE = std::numeric_limits<double>::max();
double const IQ_TO_STDEV = 0.741301109252802;  // 1 sigma in units of iqrange (assume Gaussian)

// Return the variance of a variance, assuming a Gaussian
// There is apparently an attempt to correct for bias in the factor (n - 1)/n.  RHL
inline double varianceError(double const variance, int const n) {
    return 2 * (n - 1) * variance * variance / static_cast<double>(n * n);
}

Result::Result()
        : count(0),
          mean(NaN),
          biasedVariance(NaN),
          variance(NaN),
          median(NaN),
          iqrange(NaN),
          min(NaN),
          max(NaN),
          allPixelOrMask(0x0) {}

class UnusedParameter {
public:
    using difference_type = ptrdiff_t;

    UnusedParameter const &begin() const noexcept { return *this; }
    UnusedParameter const &end() const noexcept { return *this; }
    size_t const size() const noexcept { return 0; }

    int const operator*() const noexcept { return 0; }
    UnusedParameter const &operator++() const noexcept { return *this; }
    UnusedParameter const &operator+=(difference_type) const noexcept { return *this; }
    UnusedParameter const operator+(difference_type) const noexcept { return *this; }
};

template <typename T>
constexpr bool isUnusedParameter() {
    return std::is_same<typename std::decay<T>::type, UnusedParameter>::value;
}

class AlwaysTrue {
public:
    template <typename... T>
    bool operator()(T...) const {
        return true;
    };
};

/// @internal A boolean functor to test |val| < limit  (for templated conditionals)
class CheckRange {
public:
    CheckRange(double center, double limit) : _center(center), _limit(limit) {}

    template <typename T, typename... Ts>
    bool operator()(T val, Ts...) const {
        T tmp = fabs(val - _center);
        return (tmp <= _limit);
    }

private:
    double _center;
    double _limit;
};

class CheckMask {
public:
    explicit CheckMask(lsst::afw::image::MaskPixel mask = 0x0) : _mask(mask){};

    template <typename ImagePixelT, typename MaskPixelT, typename... Ts>
    bool operator()(ImagePixelT img, MaskPixelT msk, Ts...) const {
        return !(msk & _mask);
    }

private:
    const lsst::afw::image::MaskPixel _mask;
};

class CheckFinite {
public:
    template <typename T, typename... Ts>
    bool operator()(T val, Ts...) const {
        return std::isfinite(static_cast<float>(val));
    }
};

template <typename First, typename Second>
class CheckBoth {
public:
    CheckBoth(First const &fst, Second const &snd) : _fst(fst), _snd(snd){};
    CheckBoth(First &&fst, Second &&snd) : _fst(std::move(fst)), _snd(std::move(snd)){};

    CheckBoth(CheckBoth const &) = default;
    CheckBoth(CheckBoth &&) = default;
    CheckBoth &operator=(CheckBoth const &) = default;
    CheckBoth &operator=(CheckBoth &&) = default;

    template <typename... Args>
    bool operator()(Args &&... args) const {
        return _fst(std::forward<Args>(args)...) && _snd(std::forward<Args>(args)...);
    }

private:
    First _fst;
    Second _snd;
};

template <typename... T>
struct checker;

template <typename T>
struct checker<T> {
    using type = T;
};

template <typename T, typename... Ts>
struct checker<T, Ts...> {
    using type = CheckBoth<T, typename checker<Ts...>::type>;
};

template <typename T>
T makeCombinedChecker(T t) {
    return t;
}

template <typename T, typename... Ts>
typename checker<T, Ts...>::type makeCombinedChecker(T &&t, Ts &&... ts) {
    return {std::forward<T>(t), makeCombinedChecker(std::forward<Ts>(ts)...)};
}

/** percentile()
 *
 * @brief A wrapper using the nth_element() built-in to compute percentiles for an image
 *
 * @param img       an afw::Image
 * @param quartile  the desired percentile.
 *
 */
template <typename Pixel>
double percentile(std::vector<Pixel> &img, double const fraction) {
    assert(fraction >= 0.0 && fraction <= 1.0);

    int const n = img.size();

    if (n > 1) {
        double const idx = fraction * (n - 1);

        // interpolate linearly between the adjacent values
        // For efficiency:
        // - if we're asked for a fraction > 0.5,
        //    we'll do the second partial sort on shorter (upper) portion
        // - otherwise, the shorter portion will be the lower one, we'll partial-sort that.

        int const q1 = static_cast<int>(idx);
        int const q2 = q1 + 1;

        typename std::vector<Pixel>::iterator mid1 = img.begin() + q1;
        typename std::vector<Pixel>::iterator mid2 = img.begin() + q2;
        if (fraction > 0.5) {
            std::nth_element(img.begin(), mid1, img.end());
            std::nth_element(mid1, mid2, img.end());
        } else {
            std::nth_element(img.begin(), mid2, img.end());
            std::nth_element(img.begin(), mid1, mid2);
        }

        double val1 = static_cast<double>(*mid1);
        double val2 = static_cast<double>(*mid2);
        double w1 = (static_cast<double>(q2) - idx);
        double w2 = (idx - static_cast<double>(q1));
        return w1 * val1 + w2 * val2;

    } else if (n == 1) {
        return img[0];
    } else {
        return NaN;
    }
}

/** medianAndQuartiles()
 *
 * @brief A wrapper using the nth_element() built-in to compute median and Quartiles for an image
 *
 * @param img       an afw::Image
 * @param quartile  the desired percentile.
 *
 */
typedef std::tuple<double, double, double> MedianQuartileReturn;

template <typename Pixel>
MedianQuartileReturn medianAndQuartiles(std::vector<Pixel> &img) {
    int const n = img.size();

    if (n > 1) {
        double const idx50 = 0.50 * (n - 1);
        double const idx25 = 0.25 * (n - 1);
        double const idx75 = 0.75 * (n - 1);

        // For efficiency:
        // - partition at 50th, then partition the two half further to get 25th and 75th
        // - to get the adjacent points (for interpolation), partition between 25/50, 50/75, 75/end
        //   these should be much smaller partitions

        int const q50a = static_cast<int>(idx50);
        int const q50b = q50a + 1;
        int const q25a = static_cast<int>(idx25);
        int const q25b = q25a + 1;
        int const q75a = static_cast<int>(idx75);
        int const q75b = q75a + 1;

        typename std::vector<Pixel>::iterator mid50a = img.begin() + q50a;
        typename std::vector<Pixel>::iterator mid50b = img.begin() + q50b;
        typename std::vector<Pixel>::iterator mid25a = img.begin() + q25a;
        typename std::vector<Pixel>::iterator mid25b = img.begin() + q25b;
        typename std::vector<Pixel>::iterator mid75a = img.begin() + q75a;
        typename std::vector<Pixel>::iterator mid75b = img.begin() + q75b;

        // get the 50th percentile, then get the 25th and 75th on the smaller partitions
        std::nth_element(img.begin(), mid50a, img.end());
        std::nth_element(mid50a, mid75a, img.end());
        std::nth_element(img.begin(), mid25a, mid50a);

        // and the adjacent points for each ... use the smallest segments available.
        std::nth_element(mid50a, mid50b, mid75a);
        std::nth_element(mid25a, mid25b, mid50a);
        std::nth_element(mid75a, mid75b, img.end());

        // interpolate linearly between the adjacent values
        double val50a = static_cast<double>(*mid50a);
        double val50b = static_cast<double>(*mid50b);
        double w50a = (static_cast<double>(q50b) - idx50);
        double w50b = (idx50 - static_cast<double>(q50a));
        double median = w50a * val50a + w50b * val50b;

        double val25a = static_cast<double>(*mid25a);
        double val25b = static_cast<double>(*mid25b);
        double w25a = (static_cast<double>(q25b) - idx25);
        double w25b = (idx25 - static_cast<double>(q25a));
        double q1 = w25a * val25a + w25b * val25b;

        double val75a = static_cast<double>(*mid75a);
        double val75b = static_cast<double>(*mid75b);
        double w75a = (static_cast<double>(q75b) - idx75);
        double w75b = (idx75 - static_cast<double>(q75a));
        double q3 = w75a * val75a + w75b * val75b;

        return MedianQuartileReturn(median, q1, q3);
    } else if (n == 1) {
        return MedianQuartileReturn(img[0], img[0], img[0]);
    } else {
        return MedianQuartileReturn(NaN, NaN, NaN);
    }
}

template <typename Validator, typename Algorithm>
class SinglePassStatistics {
public:
    SinglePassStatistics(const Validator &validator = Validator(), size_t baseCaseSize = 100)
            : _baseCaseSize(baseCaseSize), _validator(validator), _externalData() {}

    template <typename ImageT, typename MaskT, typename WeightT, typename VarianceT,
              typename... AlgorithmArgs>
    Result operator()(ImageT const &image, MaskT const &mask, WeightT const &weight,
                      VarianceT const &variance) {
        _externalData.goodPixels.reserve(image.size());
        return collect(image.begin(), mask.begin(), weight.begin(), variance.begin(), image.size())
                .reduce(_externalData);
    }

private:
    template <typename ImageIter, typename MaskIter, typename WeightIter, typename VarianceIter,
              typename... AlgorithmArgs>
    Algorithm collect(ImageIter img, MaskIter msk, WeightIter wgt, VarianceIter var, size_t n) {
        if (n < _baseCaseSize) {
            Algorithm algorithm;

            algorithm.collect(img, msk, wgt, var, n, _validator, _externalData);

            return algorithm;
        } else {
            size_t mid = n / 2;
            auto lhs = collect(img, msk, wgt, var, mid);
            lhs.combineWith(collect(img + mid, msk + mid, wgt + mid, var + mid, n - mid));
            return lhs;
        }
    }

    const size_t _baseCaseSize;
    Validator _validator;
    typename Algorithm::ExternalData _externalData;
};

template <bool useMask, bool useWeight, bool useVariance, bool computeRange, bool computeMedian>
class StandardStatistics {
public:
    struct ExternalData {
        std::vector<double> rejectedWeightsByBit;
        std::vector<double> maskPropagationThresholds;
        std::vector<double> goodPixels;
    };

    explicit StandardStatistics()
            : _count(0),
              _sumwx(0.0),
              _sumwx2(0.0),
              _sumw(0.0),
              _sumw2(0.0),
              _sumw2v(0.0),
              _min(MAX_DOUBLE),
              _max(-MAX_DOUBLE),
              _allPixelOrMask(0x0) {}

    StandardStatistics(StandardStatistics const &) = default;
    StandardStatistics(StandardStatistics &&) = default;
    StandardStatistics &operator=(StandardStatistics const &) = default;
    StandardStatistics &operator=(StandardStatistics &&) = default;

    void combineWith(const StandardStatistics &rhs) {
        _count += rhs._count;
        _sumwx += rhs._sumwx;
        _sumwx2 += rhs._sumwx2;
        _sumw += rhs._sumw;
        _sumw2 += rhs._sumw2;
        _sumw2v += rhs._sumw2v;
        _min = std::min(_min, rhs._min);
        _max = std::max(_max, rhs._max);
        _allPixelOrMask |= rhs._allPixelOrMask;
    }

    template <typename ImageIter, typename MaskIter, typename WeightIter, typename VarianceIter,
              typename Validator>
    void collect(ImageIter img, MaskIter msk, WeightIter wgt, VarianceIter var, size_t n,
                 Validator const &validator, ExternalData &externalData) {
        for (size_t i = 0; i < n; ++i) {
            auto const image = *img;
            auto const mask = *msk;
            auto const weight = *wgt;
            auto const variance = *var;

            if (validator(*img, *msk, *wgt, *var)) {
                ++_count;
                if (useWeight) {
                    auto const w2 = weight * weight;
                    auto const wi = weight * image;
                    _sumwx += wi;
                    _sumwx2 += wi * image;
                    _sumw += weight;
                    _sumw2 += w2;
                    if (useVariance) {
                        _sumw2v += w2 * variance;
                    }
                } else {
                    _sumwx += image;
                    _sumwx2 += image * image;
                    if (useVariance) {
                        _sumw2v += variance;
                    }
                }

                if (computeRange) {
                    if (image < _min) _min = image;
                    if (image > _max) _max = image;

                    _allPixelOrMask |= mask;
                }

                if (computeMedian) {
                    externalData.goodPixels.push_back(image);
                }
            } else {
                for (int bit = 0, nBits = externalData.maskPropagationThresholds.size(); bit < nBits; ++bit) {
                    if (mask & (1 << bit)) {
                        externalData.rejectedWeightsByBit[bit] += weight;
                    }
                }
            }
            ++img;
            ++msk;
            ++wgt;
            ++var;
        }
    }

    Result reduce(ExternalData &externalData) {
        Result result;

        result.count = _count;

        if (useWeight) {
            result.mean = _sumwx / _sumw;

            const double mean2 = result.mean * result.mean;
            result.biasedVariance = (_sumwx2 / _sumw) - mean2;

            const double ws2 = _sumw * _sumw;
            result.variance = (ws2 / (ws2 - _sumw2)) * result.biasedVariance;  // debias

            for (int bit = 0, nBits = externalData.maskPropagationThresholds.size(); bit < nBits; ++bit) {
                const double hypotheticalTotalWeight = _sumw + externalData.rejectedWeightsByBit[bit];
                externalData.rejectedWeightsByBit[bit] /= hypotheticalTotalWeight;
                if (externalData.rejectedWeightsByBit[bit] > externalData.maskPropagationThresholds[bit]) {
                    result.allPixelOrMask |= (1 << bit);
                }
            }
        } else {
            result.mean = _sumwx / _count;
        }

        result.min = _min;
        result.max = _max;

        if (computeMedian) {
            //        result.median = percentile(externalData.goodPixels, 0.5);
            auto temp = medianAndQuartiles(externalData.goodPixels);
            result.median = std::get<0>(temp);
            result.iqrange = std::get<2>(temp) - std::get<1>(temp);
        }

        return result;
    }

private:
    size_t _count;
    double _sumwx;
    double _sumwx2;
    double _sumw;
    double _sumw2;
    double _sumw2v;
    double _min;
    double _max;
    typename image::MaskPixel _allPixelOrMask;
};

template <bool useMask, bool useWeight, bool useVariance, bool computeRange, bool computeMedian,
          bool sigmaClipped, typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const &mask, WeightT const &weight,
                          VarianceT const &variance, NewStatisticsControl const &sctrl) {
    if (sigmaClipped) {
        auto result =
                SinglePassStatistics<AlwaysTrue,
                                     StandardStatistics<useMask, useWeight, useVariance, false, true>>()(
                        image, mask, weight, UnusedParameter());

        const double median = result.median;
        const double iqrange = result.iqrange;

        double center = median;
        double hwidth = sctrl.numSigmaClip * IQ_TO_STDEV * iqrange;
        for (int i = 0; i < sctrl.numIter - 1; ++i) {
            auto checker =
                    makeCombinedChecker(CheckMask(sctrl.andMask), CheckFinite(), CheckRange(center, hwidth));
            result = SinglePassStatistics<decltype(checker),
                                          StandardStatistics<useMask, useWeight, useVariance, false, false>>(
                    checker, sctrl.baseCaseSize)(image, mask, weight, UnusedParameter());
            center = result.mean;
            hwidth = sctrl.numSigmaClip * std::sqrt(result.variance);
        }

        auto checker =
                makeCombinedChecker(CheckMask(sctrl.andMask), CheckFinite(), CheckRange(center, hwidth));
        result = SinglePassStatistics<decltype(checker),
                                      StandardStatistics<useMask, useWeight, useVariance, true, false>>(
                checker, sctrl.baseCaseSize)(image, mask, weight, variance);

        // These are not computed with clipping
        result.median = median;
        result.iqrange = iqrange;

        return result;
    } else {
        auto result = SinglePassStatistics<
                AlwaysTrue, StandardStatistics<useMask, useWeight, useVariance, computeRange, computeMedian>>(
                AlwaysTrue())(image, mask, weight, variance);
        return result;
    }
}

// Selection chain to turn run time options into compile time constants
template <bool useWeight, bool useVariance, bool computeRange, bool computeMedian, bool sigmaClipped,
          typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const *mask, WeightT const &weight,
                          VarianceT const &variance, NewStatisticsControl const &sctrl) {
    if (mask) {
        return standardStatistics<true, useWeight, useVariance, computeRange, computeMedian, sigmaClipped>(
                image, *mask, weight, variance, sctrl);
    } else {
        return standardStatistics<false, useWeight, useVariance, computeRange, computeMedian, sigmaClipped>(
                image, UnusedParameter(), weight, variance, sctrl);
    }
}

template <bool useVariance, bool computeRange, bool computeMedian, bool sigmaClipped, typename ImageT,
          typename MaskT, typename WeightT, typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const *mask, WeightT const *weight,
                          VarianceT const &variance, NewStatisticsControl const &sctrl) {
    if (weight) {
        return standardStatistics<true, useVariance, computeRange, computeMedian, sigmaClipped>(
                image, mask, *weight, variance, sctrl);
    } else {
        return standardStatistics<false, useVariance, computeRange, computeMedian, sigmaClipped>(
                image, mask, UnusedParameter(), variance, sctrl);
    }
}

template <bool computeRange, bool computeMedian, bool sigmaClipped, typename ImageT, typename MaskT,
          typename WeightT, typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const *mask, WeightT const *weight,
                          VarianceT const *variance, NewStatisticsControl const &sctrl) {
    if (variance) {
        return standardStatistics<true, computeRange, computeMedian, sigmaClipped>(image, mask, weight,
                                                                                   *variance, sctrl);
    } else {
        return standardStatistics<false, computeRange, computeMedian, sigmaClipped>(image, mask, weight,
                                                                                    UnusedParameter(), sctrl);
    }
}

template <bool computeRange, bool computeMedian, typename ImageT, typename MaskT, typename WeightT,
          typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const *mask, WeightT const *weight,
                          VarianceT const *variance, bool sigmaClipped, NewStatisticsControl const &sctrl) {
    if (sigmaClipped) {
        return standardStatistics<computeRange, computeMedian, true>(image, mask, weight, variance, sctrl);
    } else {
        return standardStatistics<computeRange, computeMedian, false>(image, mask, weight, variance, sctrl);
    }
}

template <bool computeRange, typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const *mask, WeightT const *weight,
                          VarianceT const *variance, bool computeMedian, bool sigmaClipped,
                          NewStatisticsControl const &sctrl) {
    if (computeMedian) {
        return standardStatistics<computeRange, true>(image, mask, weight, variance, sigmaClipped, sctrl);
    } else {
        return standardStatistics<computeRange, false>(image, mask, weight, variance, sigmaClipped, sctrl);
    }
}

template <typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const *mask, WeightT const *weight,
                          VarianceT const *variance, bool computeRange, bool computeMedian, bool sigmaClipped,
                          NewStatisticsControl const &sctrl) {
    if (computeRange) {
        return standardStatistics<true>(image, mask, weight, variance, computeMedian, sigmaClipped, sctrl);
    } else {
        return standardStatistics<false>(image, mask, weight, variance, computeMedian, sigmaClipped, sctrl);
    }
}

// Explicit instantiations
template Result standardStatistics(ndarray::Array<double, 1, 1> const &,
                                   ndarray::Array<typename image::MaskPixel, 1, 1> const *,
                                   ndarray::Array<double, 1, 1> const *, ndarray::Array<float, 1, 1> const *,
                                   bool, bool, bool, NewStatisticsControl const &);

template Result standardStatistics(std::vector<double> const &, std::vector<std::uint16_t> const *,
                                   std::vector<double> const *, std::vector<float> const *, bool, bool, bool,
                                   NewStatisticsControl const &);

}  // math
}  // afw
}  // lsst
