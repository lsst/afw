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

#ifndef LSST_AFW_MATH_STATISTICS_FRAMEWORK_H
#define LSST_AFW_MATH_STATISTICS_FRAMEWORK_H

#include <tuple>
#include <type_traits>

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

/**
 * @file Framework for composable optimal statistics.
 *
 * Provides the following:
 *   - `Product`, (intermediate) statistic (e.g. `Count` or `Mean`)
 *   - `ProductSet`, list (tuple) of products
 *   - `ConcreteProduct`, (optimal packing of (intermediate) `Product`s that is actually used in a computation
 *   - `Validator`, (concept for a functor that validates input pixels, weights, etc.)
 *   - helper (type)functions
 *
 * `Product`s (and `Validator`s) are composable (e.g. `Mean` depends on `Sum` and `SumWeights`).
 * The framework collapses intermediate products together so that all products are calculated only once,
 * in a single pass through the data, and only take up the stack space of their individual components.
 *
 * Each `Product` (optionally) supplies an `accumulate` (for valid values) and an `accumulateClipped` member,
 * a `ConcreteProduct`s `accumulate` (and `accumulateClipped`) then delegates to each of its component
 * members.
 *
 * All these calls are compiled away and what is left is equivalent to having only the computations required
 * for each of the products (without dupplicates).
 * This approach is similar, but not quite identical to expression templates.
 */

/**
 * Set of Products
 */
template <typename... Ts>
using ProductSet = std::tuple<Ts...>;

/**
 * Class to represent "no product".
 *
 * To be used for empty intermediary products.
 */
class None {};

/**
 * Type trait. If `T` is part of `Ts...`, provides the member constant value equal to `true`.
 * Otherwise value is `false`.
 */
template <typename... Ts>
struct IsIn;

template <typename T, typename U>
struct IsIn<T, U> {
    static const bool value = std::is_same<T, U>::value;
};

template <typename T, typename U, typename... Ts>
struct IsIn<T, U, Ts...> {
    static const bool value = std::is_same<T, U>::value || IsIn<T, Ts...>::value;
};

/**
 * Type trait. If `T` is part of `ProductSet<Ts...>`, provides the member constant value equal to `true`.
 * Otherwise value is `false`.
 */
template <typename T, typename... Ts>
struct IsInProductSet;

template <typename T, typename... Ts>
struct IsInProductSet<T, ProductSet<Ts...>> {
    static const bool value = IsIn<T, Ts...>::value;
};

/**
 * Type function to add type `T` to `ProductSet<Ts...>`.
 *
 * If `T` is not `None` and not already in `Ts...` then `type` is `ProductSet<T, Ts...>`,
 * otherwise it is just `ProductSet<Ts...>`.
 *
 * Keeps the invariant that if `ProductSet<Ts...>` is a set of unique (and not `None`) types,
 * so is `InsertProductSet<T, ProductSet<Ts...>>::type`.
 */
template <typename T, typename... Ts>
struct ProductSetInsert;

template <typename T, typename... Ts>
struct ProductSetInsert<T, ProductSet<Ts...>> {
    using type =
            typename std::conditional<IsIn<T, Ts...>::value, ProductSet<Ts...>, ProductSet<T, Ts...>>::type;
};

template <typename... Ts>
struct ProductSetInsert<None, ProductSet<Ts...>> {
    using type = ProductSet<Ts...>;
};

/**
 * Type function to add types `Ts...` to `ProductSet` `PS`.
 *
 * Keeps the invariant that if `PS` is a set of unique (and not `None`) types,
 * so is `ProductSetInsertMultiple<PS, Ts...>::type`.
 */
template <typename PS, typename... Ts>
struct ProductSetInsertMultiple;

template <typename PS, typename T>
struct ProductSetInsertMultiple<PS, T> {
    using type = typename ProductSetInsert<T, PS>::type;
};

template <typename PS, typename T, typename... Ts>
struct ProductSetInsertMultiple<PS, T, Ts...> {
    using type = typename ProductSetInsert<T, typename ProductSetInsertMultiple<PS, Ts...>::type>::type;
};

/**
 * Type function to union a `ProductSet` `PS` with a `ProductSet<Ts...>`.
 *
 * Keeps the invariant that if `PS` is a set of unique (and not `None`) types,
 * so is `ProductSetInsertMultiple<PS, ProductSet<Ts...>>::type`.
 */
template <typename PS, typename... Ts>
struct ProductSetUnion;

template <typename PS, typename... Ts>
struct ProductSetUnion<PS, ProductSet<Ts...>> {
    using type = typename ProductSetInsertMultiple<PS, Ts...>::type;
};

/**
 * Type function to flatten a `ProductSet`.
 *
 * Provides the member typedef `type` which is the union of the input `ProductSet` and
 * the intermediate `ProductSet`s of all products therein.
 * Only one level of flattening is performed.
 *
 * For example, let:
 *
 *  - `A::Intermediate = ProductSet<C, D, E>`
 *  - `B::Intermediate = ProductSet<E, F>`
 *
 * then `FlattenOne<ProductSet<A, B>>::type` is a `ProductSet<A, C, D, E, B, F>`.
 */
template <typename T>
struct FlattenOne;

template <>
struct FlattenOne<ProductSet<None>> {
    using type = ProductSet<None>;
};

template <typename T>
struct FlattenOne<ProductSet<T>> {
    using type = typename ProductSetUnion<ProductSet<T>, typename T::Intermediate>::type;
};

template <typename T, typename... Ts>
struct FlattenOne<ProductSet<T, Ts...>> {
    using type = typename ProductSetUnion<typename FlattenOne<ProductSet<T>>::type,
                                          typename FlattenOne<ProductSet<Ts...>>::type>::type;
};

/**
 * Helper class for `getByType`
 */
template <typename T, size_t N, typename... Ts>
struct GetHelper;

template <typename T, size_t N, typename... Ts>
struct GetHelper<T, N, T, Ts...> {
    static constexpr size_t value = N;
};

template <typename T, size_t N, typename U, typename... Ts>
struct GetHelper<T, N, U, Ts...> {
    static constexpr size_t value = GetHelper<T, N + 1, Ts...>::value;
};

/**
 * Get value from `ProductSet` indexed by type.
 *
 * Note, this is only needed util we can use C++14.
 * After which we can replace it with `std::get<T>`.
 */
template <typename T, typename... Ts>
constexpr T const& getByType(ProductSet<Ts...> const& t) {
    return std::get<GetHelper<T, 0, Ts...>::value>(t);
};

template <typename T, typename... Ts>
constexpr T& getByType(ProductSet<Ts...>& t) {
    return std::get<GetHelper<T, 0, Ts...>::value>(t);
};

/**
 * Helper class to call `accumulate` member function for every member of a `ProductSet`.
 */
template <typename T, size_t N>
struct AccumulateHelper {
    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    static void accumulate(T& t, ImagePixelT img, MaskPixelT msk, VariancePixelT var, WeightPixelT wgt) {
        AccumulateHelper<T, N - 1>::accumulate(t, img, msk, var, wgt);
        std::get<N - 1>(t).accumulate(img, msk, var, wgt);
    }
};

template <typename T>
struct AccumulateHelper<T, 1> {
    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    static void accumulate(T& t, ImagePixelT img, MaskPixelT msk, VariancePixelT var, WeightPixelT wgt) {
        std::get<0>(t).accumulate(img, msk, var, wgt);
    }
};

/**
 * Call `accumulate` member function for every member of `ProductSet` argument `t`.
 */
template <typename... Args, typename ImagePixelT, typename MaskPixelT, typename VariancePixelT,
          typename WeightPixelT>
void accumulateHelper(ProductSet<Args...>& t, ImagePixelT img, MaskPixelT msk, VariancePixelT var,
                      WeightPixelT wgt) {
    AccumulateHelper<decltype(t), sizeof...(Args)>::accumulate(t, img, msk, var, wgt);
}

/**
 * Helper class to call `accumulateClipped` member function for every member of a `ProductSet`.
 */
template <typename T, size_t N>
struct AccumulateClippedHelper {
    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    static void accumulateClipped(T& t, ImagePixelT img, MaskPixelT msk, VariancePixelT var,
                                  WeightPixelT wgt) {
        AccumulateClippedHelper<T, N - 1>::accumulateClipped(t, img, msk, var, wgt);
        std::get<N - 1>(t).accumulateClipped(img, msk, var, wgt);
    }
};

template <typename T>
struct AccumulateClippedHelper<T, 1> {
    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    static void accumulateClipped(T& t, ImagePixelT img, MaskPixelT msk, VariancePixelT var,
                                  WeightPixelT wgt) {
        std::get<0>(t).accumulateClipped(img, msk, var, wgt);
    }
};

/**
 * Call `accumulateClipped` member function for every member of `ProductSet` argument `t`.
 */
template <typename... Args, typename ImagePixelT, typename MaskPixelT, typename VariancePixelT,
          typename WeightPixelT>
void accumulateClippedHelper(ProductSet<Args...>& t, ImagePixelT img, MaskPixelT msk, VariancePixelT var,
                             WeightPixelT wgt) {
    AccumulateClippedHelper<decltype(t), sizeof...(Args)>::accumulateClipped(t, img, msk, var, wgt);
}

/**
 * Helper class to call `combine` member function for every member of a `ProductSet`.
 */
template <typename T, size_t N>
struct CombineHelper {
    template <typename Product>
    static void combine(T& t, Product const& other) {
        CombineHelper<T, N - 1>::combine(t, other);
        std::get<N - 1>(t).combine(other);
    }
};

template <typename T>
struct CombineHelper<T, 1> {
    template <typename Product>
    static void combine(T& t, Product const& other) {
        std::get<0>(t).combine(other);
    }
};

/**
 * Call `combine` member function for every member of `ProductSet` argument `t`.
 */
template <typename... Args, typename Product>
void combineHelper(ProductSet<Args...>& t, Product const& other) {
    CombineHelper<decltype(t), sizeof...(Args)>::combine(t, other);
}

/**
 * Base class for statistics products (e.g. Sum, Count, Variance, etc.).
 *
 * A Product is templated on the intermediate Products it requires (or None),
 * and defines an `accumulate` member function (default noop).
 */
template <typename... Ts>
class Product {
public:
    using Intermediate = typename FlattenOne<ProductSet<Ts...>>::type;

    /**
     * Combine two products
     */
    template <typename T>
    void combine(T const&) { /* noop for abstract products */
    }

    /**
     * Process valid value (pixel, mask, variance and/or weight)
     */
    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT, MaskPixelT, VariancePixelT, WeightPixelT) { /* noop for abstract products */
    }

    /**
     * Process clipped value (pixel, mask, variance and/or weight)
     */
    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulateClipped(ImagePixelT, MaskPixelT, VariancePixelT,
                           WeightPixelT) { /* noop for abstract products */
    }
};

/**
 * Class that represents a concrete Product.
 *
 * Creates an optimal packing of unique intermediate products and provides
 * an `accumulate` member function that accumulates each one.
 *
 * Results may be extracted from such objects using the non-member `extract` function.
 */
template <typename... Ts>
class ConcreteProduct {
public:
    using Intermediate = typename FlattenOne<ProductSet<Ts...>>::type;

    /**
     * Get value from intermediate product.
     */
    template <typename T>
    T const& get() const {
        return getByType<T>(_x);
    }

    template <typename T>
    T& get() {
        return getByType<T>(_x);
    }

    template <typename T>
    void combine(T const& other) {
        combineHelper(_x, other);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulate(ImagePixelT img, MaskPixelT msk, VariancePixelT var, WeightPixelT wgt) {
        accumulateHelper(_x, img, msk, var, wgt);
    }

    template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT, typename WeightPixelT>
    void accumulateClipped(ImagePixelT img, MaskPixelT msk, VariancePixelT var, WeightPixelT wgt) {
        accumulateClippedHelper(_x, img, msk, var, wgt);
    }

private:
    Intermediate _x;
};

/**
 * Extract result of statistic T from Product.
 */
template <typename T, typename Product,
          typename std::enable_if<IsInProductSet<T, typename Product::Intermediate>::value, int>::type = 0>
auto extract(Product const& p) -> decltype(std::declval<T>().get(p)) {
    return p.template get<T>().get(p);
}

/**
 * Extract result of statistic T from Product.
 */
template <typename T, typename Product,
          typename std::enable_if<IsInProductSet<T, typename Product::Intermediate>::value, int>::type = 0>
auto extractOptional(Product const& p) -> decltype(std::declval<T>().get(p)) {
    return p.template get<T>().get(p);
}

template <typename T, typename Product,
          typename std::enable_if<!IsInProductSet<T, typename Product::Intermediate>::value, int>::type = 0>
auto extractOptional(Product const& p) -> decltype(std::declval<T>().get(p)) {
    throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                      "Cannot extract type from product that does not contain it");
    return {};
}

/**
 * Functor that returns `true` if both its component `Validator`s do
 *
 * Building block for composable `Validator`s
 */
template <typename First, typename Second>
class ValidateBoth {
public:
    ValidateBoth(First const& fst, Second const& snd) : _fst(fst), _snd(snd){};
    ValidateBoth(First&& fst, Second&& snd) : _fst(std::move(fst)), _snd(std::move(snd)){};

    ValidateBoth(ValidateBoth const&) = default;
    ValidateBoth(ValidateBoth&&) = default;
    ValidateBoth& operator=(ValidateBoth const&) = default;
    ValidateBoth& operator=(ValidateBoth&&) = default;

    template <typename... Args>
    bool operator()(Args&&... args) const {
        return _fst(std::forward<Args>(args)...) && _snd(std::forward<Args>(args)...);
    }

private:
    First _fst;
    Second _snd;
};

/**
 * Functor that returns `true` if all its component `Validator's` do
 */
template <typename... T>
struct CombinedValidator;

template <typename T>
struct CombinedValidator<T> {
    using type = T;
};

template <typename T, typename... Ts>
struct CombinedValidator<T, Ts...> {
    using type = ValidateBoth<T, typename CombinedValidator<Ts...>::type>;
};

/**
 * Make a new `CombinedValidator` from a an arbitrary number of input `Validator`s
 */
template <typename T>
T makeCombinedValidator(T t) {
    return t;
}

template <typename T, typename... Ts>
typename CombinedValidator<T, Ts...>::type makeCombinedValidator(T&& t, Ts&&... ts) {
    return {std::forward<T>(t), makeCombinedValidator(std::forward<Ts>(ts)...)};
}

/**
 * General limits and constants
 *
 * @note should probably move these somewhere else.
 */
double const NaN = std::numeric_limits<double>::quiet_NaN();
double const MAX_DOUBLE = std::numeric_limits<double>::max();
double const IQ_TO_STDEV = 0.741301109252802;  // 1 sigma in units of iqrange (assume Gaussian)

/** percentile()
 *
 * @brief A wrapper using the nth_element() built-in to compute percentiles for an image
 *
 * @param img       an afw::Image
 * @param quartile  the desired percentile.
 *
 */
template <typename Pixel>
double percentile(std::vector<Pixel>& img, double const fraction) {
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
MedianQuartileReturn medianAndQuartiles(std::vector<Pixel>& img) {
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

// Return the variance of a variance, assuming a Gaussian
// There is apparently an attempt to correct for bias in the factor (n - 1)/n.  RHL
inline double varianceError(double const variance, int const n) {
    return 2 * (n - 1) * variance * variance / static_cast<double>(n * n);
}

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
