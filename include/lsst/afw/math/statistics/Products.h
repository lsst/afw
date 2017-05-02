// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2017 AURA/LSST.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_MATH_STATISTICS_PRODUCTS_H
#define LSST_AFW_MATH_STATISTICS_PRODUCTS_H

#include <vector>
#include <iostream>

#include "ndarray.h"

#include "lsst/pex/config.h"
#include "lsst/pex/exceptions.h"

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/statistics/Framework.h"

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

/**
 * Product to calculate number of accumulated values
 */
class Count : public Product<None> {
public:
    Count() : _count(0){};

    template <typename Product>
    void combine(Product const& other) {
        _count += extract<Count>(other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT, MaskPixelT, VariancePixelT, WeightPixelT) {
        ++_count;
    }

    template <typename Product>
    size_t get(Product const&) const {
        return _count;
    };

private:
    size_t _count;
};

/**
 * Product to calculate minimum pixel value
 */
class Min : public Product<None> {
public:
    Min() : _min(std::numeric_limits<double>::max()){};

    template <typename Product>
    void combine(Product const& other) {
        _min = std::min(_min, extract<Min>(other));
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT img, MaskPixelT, VariancePixelT, WeightPixelT) {
        _min = std::min(_min, static_cast<double>(img));
    }

    template <typename Product>
    double get(Product const&) const {
        return _min;
    };

private:
    double _min;
};

/**
 * Product to calculate maximum pixel value
 */
class Max : public Product<None> {
public:
    Max() : _max(std::numeric_limits<double>::min()){};

    template <typename Product>
    void combine(Product const& other) {
        _max = std::max(_max, extract<Max>(other));
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT img, MaskPixelT, VariancePixelT, WeightPixelT) {
        _max = std::max(_max, static_cast<double>(img));
    }

    template <typename Product>
    double get(Product const&) const {
        return _max;
    };

private:
    double _max;
};

/**
 * Product to calculate sum of (weighted) pixel values
 */
class Sum : public Product<None> {
public:
    Sum() : _sum(0.0){};

    template <typename Product>
    void combine(Product const& other) {
        _sum += extract<Sum>(other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT img, MaskPixelT, VariancePixelT, WeightPixelT wgt) {
        _sum += img * wgt;
    }

    template <typename Product>
    double get(Product const&) const {
        return _sum;
    };

private:
    double _sum;
};

/**
 * Product to calculate sum of (weighted) squared pixel values
 */
class SumSquared : public Product<None> {
public:
    SumSquared() : _sum(0.0){};

    template <typename Product>
    void combine(Product const& other) {
        _sum += extract<SumSquared>(other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT img, MaskPixelT, VariancePixelT, WeightPixelT wgt) {
        _sum += img * img * wgt;
    }

    template <typename Product>
    double get(Product const&) const {
        return _sum;
    };

private:
    double _sum;
};

/**
 * Product to calculate sum of weights
 */
class SumWeights : public Product<None> {
public:
    SumWeights() : _sum(0.0){};

    template <typename Product>
    void combine(Product const& other) {
        _sum += extract<SumWeights>(other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT, MaskPixelT, VariancePixelT, WeightPixelT wgt) {
        _sum += wgt;
    }

    template <typename Product>
    double get(Product const&) const {
        return _sum;
    };

private:
    double _sum;
};

/**
 * Product to calculate sum of (weighted) variance pixels
 */
class SumVariance : public Product<None> {
public:
    SumVariance() : _sum(0.0){};

    template <typename Product>
    void combine(Product const& other) {
        _sum += extract<SumVariance>(other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT, MaskPixelT, VariancePixelT var, WeightPixelT wgt) {
        _sum += var * wgt * wgt;
    }

    template <typename Product>
    double get(Product const&) const {
        return _sum;
    };

private:
    double _sum;
};

/**
 * Product to calculate sum of squared weights
 */
class SumWeightsSquared : public Product<None> {
public:
    SumWeightsSquared() : _sum(0.0){};

    template <typename Product>
    void combine(Product const& other) {
        _sum += extract<SumWeightsSquared>(other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT, MaskPixelT, VariancePixelT, WeightPixelT wgt) {
        _sum += wgt * wgt;
    }

    template <typename Product>
    double get(Product const&) const {
        return _sum;
    };

private:
    double _sum;
};

/**
 * Product to calculate all pixel OR mask
 */
class OrMask : public Product<None> {
public:
    OrMask() : _orMask(0x0){};

    template <typename Product>
    void combine(Product const& other) {
        _orMask |= extract<OrMask>(other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT, MaskPixelT msk, VariancePixelT, WeightPixelT) {
        _orMask |= msk;
    }

    template <typename Product>
    image::MaskPixel get(Product const&) const {
        return _orMask;
    };

private:
    image::MaskPixel _orMask;
};

/**
 * Product to calculate (weighted) Mean
 */
class Mean : public Product<Sum, SumWeights> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return extract<Sum>(p) / extract<SumWeights>(p);
    }
};

/**
 * Product to calculated (weighted) mean of squared values
 */
class MeanSquared : public Product<SumSquared, SumWeights> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return extract<SumSquared>(p) / extract<SumWeights>(p);
    }
};

/**
 * Product to calculated variance
 */
class BiasedVariance : public Product<MeanSquared, Mean> {
public:
    template <typename Product>
    double get(Product const& p) const {
        using std::pow;
        return extract<MeanSquared>(p) - pow(extract<Mean>(p), 2);
    }
};

/**
 * Product to calculate debiased variance
 */
class Variance : public Product<BiasedVariance, SumWeightsSquared> {
public:
    template <typename Product>
    double get(Product const& p) const {
        auto variance = extract<BiasedVariance>(p);
        auto sumw = extract<SumWeights>(p);
        auto sumw2 = extract<SumWeightsSquared>(p);
        return variance * sumw * sumw / (sumw * sumw - sumw2);
    }
};

/**
 * Product to calculate standard deviation
 */
class Stddev : public Product<Variance> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return std::sqrt(extract<Variance>(p));
    }
};

/**
 * Product to gather all valid pixel values (in a (shared) `std::vector`)
 */
class Gather : public Product<None> {
public:
    Gather() : _vec(std::make_shared<std::vector<double>>()) {}

    template <typename Product>
    void combine(Product const& other) {
        auto ov = extract<Gather>(other);
        if (ov != _vec) {
            _vec->reserve(_vec->size() + ov->size());
            std::copy(ov->begin(), ov->end(), std::back_inserter(*_vec));
        }
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT img, MaskPixelT, VariancePixelT, WeightPixelT) {
        _vec->push_back(img);
    }

    template <typename Product>
    std::shared_ptr<std::vector<double>> get(Product const&) const {
        return _vec;
    };

private:
    std::shared_ptr<std::vector<double>> _vec;
};

/**
 * Product to calculate median
 */
class Median : public Product<Gather> {
public:
    Median() : _median(), _cached(false) {}

    template <typename Product>
    double get(Product const& p) const {
        if (!_cached) {
            auto v = extract<Gather>(p);
            _median = percentile(*v, 0.5);
            _cached = true;
        }
        return _median;
    };

private:
    mutable double _median;
    mutable bool _cached;
};

/**
 * Product to calculate median and quartiles
 */
class Quartiles : public Product<Gather> {
public:
    Quartiles() : _quartiles(), _cached(false) {}

    template <typename Product>
    MedianQuartileReturn get(Product const& p) const {
        if (!_cached) {
            auto v = extract<Gather>(p);
            _quartiles = medianAndQuartiles(*v);
            _cached = true;
        }
        return _quartiles;
    };

private:
    mutable MedianQuartileReturn _quartiles;
    mutable bool _cached;
};

/**
 * Product to calculate variance of the mean.
 *
 * The calculation may use either the input variance (SumVariance specialization)
 * or the calculated variance (Variance specialization).
 */
template <typename T>
class MeanVar;

template <>
class MeanVar<SumVariance> : public Product<SumWeights, SumVariance> {
public:
    template <typename Product>
    double get(Product const& p) const {
        double sumw = extract<SumWeights>(p);
        return extract<SumVariance>(p) / (sumw * sumw);
    }
};

template <>
class MeanVar<Variance> : public Product<SumWeights, SumWeightsSquared, Variance> {
public:
    template <typename Product>
    double get(Product const& p) const {
        double sumw = extract<SumWeights>(p);
        return extract<Variance>(p) * extract<SumWeightsSquared>(p) / (sumw * sumw);
    }
};

/**
 * Product to calculate variance of the variance
 */
class VarianceVar : public Product<Variance, Count> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return varianceError(extract<Variance>(p), extract<Count>(p));
    }
};

/**
 * Product to calculate the standard error of the mean
 */
template <typename T>
class MeanErr : public Product<MeanVar<T>> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return std::sqrt(extract<MeanVar<T>>(p));
    }
};

/**
 * Product to calculate the standard error of the variance
 */
class VarianceErr : public Product<VarianceVar> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return std::sqrt(extract<VarianceVar>(p));
    }
};

/**
 * Product to calculate the standard error of the standard deviation
 */
class StddevErr : public Product<Stddev, Variance> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return 0.5 * std::sqrt(extract<Variance>(p)) / extract<Stddev>(p);
    }
};

/**
 * Product to calculate the standard error of the mean of the squared pixel values
 */
class MeanSquaredErr : public Product<MeanSquared, Count> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return std::sqrt(2 * std::pow(extract<MeanSquared>(p) / extract<Count>(p), 2));  // assumes Gaussian
    }
};

/**
 * Product to calculate the standard error of the median
 */
class MedianErr : public Product<Median, Variance, Count> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return std::sqrt(geom::HALFPI * extract<Variance>(p) / extract<Count>(p));  // assumes Gaussian
    }
};

/**
 * Product to calculate the sigma clipped mean
 */
class ClippedMean : public Product<Mean> {
public:
    ClippedMean() : _clippedMean(NaN), _computed(false){};

    // Extract Mean from separately calculated ConcreteProduct,
    // this is the only way to construct a ClippedMean
    template <typename Product>
    ClippedMean& operator=(Product const& p) {
        _clippedMean = extract<Mean>(p);
        _computed = true;
        return *this;
    }

    template <typename Product>
    double get(Product const& p) const {
        if (_computed) {
            return _clippedMean;
        } else {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "ClippedMean has not been calculated");
        }
    }

private:
    double _clippedMean;
    bool _computed;
};

/**
 * Product to calculate the sigma clipped variance
 */
class ClippedVariance : public Product<Variance> {
public:
    ClippedVariance() : _clippedVariance(NaN), _computed(false){};

    // Extract Variance from separately calculated ConcreteProduct,
    // this is the only way to construct a ClippedVariance
    template <typename Product>
    ClippedVariance& operator=(Product const& p) {
        _clippedVariance = extract<Variance>(p);
        _computed = true;
        return *this;
    }

    template <typename Product>
    double get(Product const& p) const {
        if (_computed) {
            return _clippedVariance;
        } else {
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                              "ClippedVariance has not been calculated");
        }
    }

private:
    double _clippedVariance;
    bool _computed;
};

/**
 * Product to calculate the sigma clipped standard deviation
 */
class ClippedStddev : public Product<ClippedVariance> {
public:
    template <typename Product>
    double get(Product const& p) const {
        return std::sqrt(extract<ClippedVariance>(p));
    }
};

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
