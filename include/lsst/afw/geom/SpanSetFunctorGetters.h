
// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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

 #ifndef LSST_AFW_GEOM_SPANSETFUNCTORGETTERS_H
 #define LSST_AFW_GEOM_SPANSETFUNCTORGETTERS_H

#include <type_traits>
#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/image.h"
#include "lsst/pex/exceptions.h"

namespace lsst { namespace afw { namespace geom { namespace details {

/* These variadic functions exist because of a current limitation in c++ where
 * calls of the form foo(x)... are not possibe unless it is used as a parameter
 * for a second function. The following functions take in a variadic parameter
 * pack, and make recursive calls until the corresponding class method has been
 * called on all the variadic parameters
 */

template<typename T>
void variadicSpanSetter(Span const spn, T & x) {
    x.setSpan(spn);
}

template <typename T, typename... Args>
void variadicSpanSetter(Span const spn, T & first, Args&... x) {
    variadicSpanSetter(spn, first);
    variadicSpanSetter(spn, x...);
}

template<typename T>
void variadicBoundChecker(Box2I const box, int area, T const & x) {
    x.checkExtents(box, area);
}

template<typename T, typename... Args>
void variadicBoundChecker(Box2I const box, int area, T const & first, Args&... x) {
    variadicBoundChecker(box, area, first);
    variadicBoundChecker(box, area, x...);
}

template<typename T>
void variadicIncrementPosition(T & x) {
    x.increment();
}

template<typename T, typename... Args>
void variadicIncrementPosition(T & first, Args&... x) {
    variadicIncrementPosition(first);
    variadicIncrementPosition(x...);
}

/* Getter classes exist to provide a common API (duck-type) for accessing data from
 * different data-types. This common API is used by the SpanSets applyFunctor method
 * for passing the correct references into the supplied functor.
 */

template <typename T>
class IterGetter {
    /* Getter class to manage retrieving values from a generic iterator
       !!! be careful !!! Because there is no way to easily check bounds
       for a generic iterator, it is possible to pass an iterator too
       short, in which case the apply functor will run off the end.

       Also note that because this is a generic iterator, the iterator
       must always provide the next desired value when it is incremented.
       For instance when iterating over a two dimensional object should
       automaticall wrap to the next line when the ++ operator reaches
       the end of the line. In this way all generic inputs must be treated
       as 1d arrays, of length at least the number of elements in the
       SpanSet which originates the applyFunctor call
     */
 public:
    using type = typename std::iterator_traits<T>::value_type;
    explicit IterGetter(T iter): _iter(iter) {}

    // There is no good way to check the extents of a generic iterator, so make
    // a no-op function to satisfy api
    void checkExtents(Box2I const & bbox, int area) const {}

    void setSpan(Span const & span) {}

    void increment() { ++_iter;}

    typename std::iterator_traits<T>::reference get() const {return *_iter;}

 private:
    T _iter;
};

template <typename T>
class ConstantGetter {
    // Getter class which takes in a constant value, and simply returns that value
    // for each iteration
 public:
    explicit ConstantGetter(T constant): _const(constant) {}

    // Constants are simply repeated, so no need to check extents, make no-op
    // function
    void checkExtents(Box2I const & bbox, int area) const {}

    void setSpan(Span const & span) {}

    // No need to increment, make a no-op function
    void increment() {}

    T get() const {return _const;}
 private:
    T _const;
};

template <typename T, int N, int C>
class ImageNdGetter {
    // Getter class to manage iterating though an ndarray which is interpreted as a 2D image
 public:
    using Reference = typename ndarray::Array<T, N, C>::Reference::Reference;

    ImageNdGetter(ndarray::Array<T, N, C> const & array, geom::Point2I const & xy0): _array(array), _xy0(xy0) {}

    void checkExtents(Box2I const & bbox, int area) const {
        // If the bounding box lays outside the are of the image, throw an error
        geom::Box2I arrayBBox(_xy0, _xy0+geom::Extent2I(_array.template getSize<1>(),
                                                        _array.template getSize<0>()));
        if (!arrayBBox.contains(bbox)) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                              "SpanSet bounding box lands outside array");
        }
    }

    void setSpan(Span const & span) {
        auto _iterY = _array.begin() + (span.getY() - _xy0.getY());
        _iterX = (*_iterY).begin() + (span.getMinX() - _xy0.getX());
    }

    void increment() { ++_iterX; }

    Reference get() {
        return *_iterX;}

 private:
    ndarray::Array<T, N, C>  _array;
    geom::Point2I _xy0;
    typename ndarray::Array<T, N, C>::Reference::Iterator _iterX = nullptr;
};

template <typename T>
class FlatNdGetter {
    // Getter class to manage iterating though an ndarray which is interpreted as a 2D image
 public:
    using Reference = typename ndarray::Array<T, 1, 1>::Reference;

    explicit FlatNdGetter(ndarray::Array<T, 1, 1> const & array): _array(array), _iter(_array.begin()) {}

    void checkExtents(Box2I const & bbox, int area) const {
        // If the area of the array is greater than the size of the array, throw an error
        // as the iteration dimensions will not match
        auto shape = _array.getShape();
        if (area > static_cast<int>(shape[0])) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                              "SpanSet has dimensionality greater than array");
        }
    }

    void setSpan(Span const & span) {}

    void increment() { ++_iter; }

    Reference get() const {
        return *_iter;}

 private:
    ndarray::Array<T, 1, 1>  _array;
    typename ndarray::Array<T, 1, 1>::Iterator _iter;
};

/*
 * The MakeGetter function serves as a dispatch agent for the applyFunctor method.
 * The overloaded function accepts various types of objects and initialized the
 * corresponding "Getter" classes. These classes exist to structure calls to the
 * various data-types in such a way that they all share a common API. This allows
 * the applyFunctor method to operate on each of these data-types as if they were all
 * the same. (aka they all correspond to the same duck-type)
 */

template <typename T>
FlatNdGetter<T> makeGetter(FlatNdGetter<T> & getter) {
    // This function simply passes through any FlatNdGetter passed to it
    return getter;
}

template <typename T, int inA, int inB>
ImageNdGetter<T, inA, inB> makeGetter(ImageNdGetter<T, inA, inB> & getter) {
    // This function simply passes though any imageNdGetter passed to it
    return getter;
}

template <typename T>
ImageNdGetter<T, 2, 1> makeGetter(lsst::afw::image::Image<T> & image) {
    // Function to create a ndarray getter from an afw image
    return ImageNdGetter<T, 2, 1>(image.getArray(), image.getXY0());
}

template <typename T>
ImageNdGetter<T const, 2, 1> makeGetter(lsst::afw::image::Image<T> const & image) {
    // Function to create a ndarray getter from an afw image
    return ImageNdGetter<T const, 2, 1>(image.getArray(), image.getXY0());
}

template <typename T>
ImageNdGetter<T, 2, 1> makeGetter(lsst::afw::image::Mask<T> & image) {
    // Function to create a ndarray getter from an afw image
    return ImageNdGetter<T, 2, 1>(image.getArray(), image.getXY0());
}

template <typename T>
ImageNdGetter<T const, 2, 1> makeGetter(lsst::afw::image::Mask<T> const & image) {
    // Function to create a ndarray getter from an afw image
    return ImageNdGetter<T const, 2, 1>(image.getArray(), image.getXY0());
}

template <typename T>
ConstantGetter<T> makeGetter(T num, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0) {
    // Function to create a getter for constant numeric types. Use template type checking to ensure the
    // type is an integral type, or floating point type
    return ConstantGetter<T>(num);
}

// There is no type trait in the standard library to check for iterator types, so we declare one here
// Template specialization is used here. If type T can be mapped to an iterator trait, then it should
// be considered as an iterator. If c++11 supported concepts this would be a perfect place for it. this
// is essentially a duck-type type checking mechanism
template <typename T, typename = void>
struct is_iterator : std::false_type {};

template <typename T>
struct is_iterator<T, typename std::enable_if<!std::is_same<typename std::iterator_traits<T>::value_type,
                                                            void>::value>::type> :std::true_type {};

template <typename T>
IterGetter<T> makeGetter(T iter, typename std::enable_if<is_iterator<T>::value, int>::type = 0) {
    // Use defined type trait checker to create an iterator getter if the template is an iterator type
    return IterGetter<T>(iter);
}
}}}} // Close namespace lsst::afw::geom::details

namespace ndarray {
    // These function are placed in the ndarray namespace to enable function argument namespace lookup
    // This means the function can be used on an ndarray without the need to specify the namespace of
    // the function itself
    namespace details = lsst::afw::geom::details;
    namespace afwGeom = lsst::afw::geom;
    template <typename T, int inA, int inB>

    /** @brief Marks a ndarray to be interpreted as a 1D vector when applying a functor from a SpanSet
     *
     * @param array - ndarray which will be used in functor calls
     *
     * @tparam T - The datatype of a pixel in the ndarray
     * @tparam inA - The number of dimensions of the array
     * @tparam inB - Number of guaranteed row-major contiguous dimensions, starting from the end
     */
    details::FlatNdGetter<T> ndFlat(ndarray::Array<T, inA, inB> const & array) {
        // Function to mark a ndarray to be treated as a flat vector by the applyFunctor method
        return details::FlatNdGetter<T>(array);
    }

    /** @brief Marks a ndarray to be interpreted as an image when applying a functor from a SpanSet
     *
     * @param array - ndarray which will be used in functor calls
     *
     * @tparam T - The datatype of a pixel in the ndarray
     * @tparam inA - The number of dimensions of the array
     * @tparam inB - Number of guaranteed row-major contiguous dimensions, starting from the end
     */
    template <typename T, int inA, int inB>
    details::ImageNdGetter<T, inA, inB> ndImage(ndarray::Array<T, inA, inB> const & array,
                                                afwGeom::Point2I xy0 = afwGeom::Point2I()) {
        // Function to mark a ndarray to be treated as a 2D image by the applyFunctor method
        return details::ImageNdGetter<T, inA, inB>(array, xy0);
    }
} // Close namespace ndarray

 #endif // LSST_AFW_GEOM_SPANSETFUNCTORGETTERS_H
