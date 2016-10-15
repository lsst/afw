
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


template<typename T>
void variadicSetter(Span spn, T & x) {
    x.setSpan(spn);
}

template <typename T, typename... Args>
void variadicSetter(Span spn, T & first, Args&... x) {
    variadicSetter(spn, first);
    variadicSetter(spn, x...);
}

template<typename T>
void variadicChecker(Box2I box, int area, T & x) {
    x.checkExtents(box, area);
}

template<typename T, typename... Args>
void variadicChecker(Box2I box, int area, T & first, Args&... x) {
    variadicChecker(box, area, first);
    variadicChecker(box, area, x...);
}

template<typename T>
void variadicIncrement(T & x) {
    x.increment();
}

template<typename T, typename... Args>
void variadicIncrement(T & first, Args&... x) {
    variadicIncrement(first);
    variadicIncrement(x...);
}

template <typename T>
class IterGetter {
    /* Getter class to manage retreving values from a generic iterator
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
    // Getter class whic takes in a constant value, and simply returns that value
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

    ImageNdGetter(ndarray::Array<T, N, C> const & array, geom::Point2I xy0): _array(array), _xy0(xy0) {}

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
    typename ndarray::Array<T, N, C>::Reference::Iterator _iterX;
};

template <typename T>
class FlatNdGetter {
    // Getter class to manage iterating though an ndarray which is interpreted as a 2D image
 public:
    using Reference = typename ndarray::Array<T, 1, 1>::Reference;

    explicit FlatNdGetter(ndarray::Array<T, 1, 1> const & array): _array(array) {
        _iter = _array.begin();
    }

    void checkExtents(Box2I const & bbox, int area) {
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
ImageNdGetter<T, 2, 1> makeGetter(lsst::afw::image::Image<T> const & image) {
    // Function to create a ndarray getter from an afw image
    return ImageNdGetter<T, 2, 1>(image.getArray(), image.getXY0());
}

template <typename T>
ImageNdGetter<T, 2, 1> makeGetter(lsst::afw::image::Mask<T> & image) {
    // Function to create a ndarray getter from an afw image
    return ImageNdGetter<T, 2, 1>(image.getArray(), image.getXY0());
}

template <typename T>
ImageNdGetter<T, 2, 1> makeGetter(lsst::afw::image::Mask<T> const & image) {
    // Function to create a ndarray getter from an afw image
    return ImageNdGetter<T, 2, 1>(image.getArray(), image.getXY0());
}

template <typename T>
ConstantGetter<T> makeGetter(T num, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0) {
    // Function to create a getter for constant numeric types. Use template type checking to ensure the
    // type is an integral type, or floating point type
    return ConstantGetter<T>(num);
}

// There is no type trait in the standard library to check for iterator types, so we declare one here
// Template specialization is used here. If type T can be mapped to an iterator trait, then it should
// be concidered as an iterator. If c++11 supported concepts this would be a perfect place for it. this
// is essensially a duck-type type checking mechanism
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
    namespace geom = lsst::afw::geom;
    template <typename T, int inA, int inB>
    details::FlatNdGetter<T> ndFlat(ndarray::Array<T, inA, inB> & array) {
        // Function to mark a ndarray to be treated as a flat vector by the applyFunctor method
        return details::FlatNdGetter<T>(array);
    }

    template <typename T, int inA, int inB>
    details::ImageNdGetter<T, inA, inB> ndImage(ndarray::Array<T, inA, inB> & array,
                                                geom::Point2I xy0 = geom::Point2I()) {
        // Function to mark a ndarray to be treated as a 2D image by the applyFunctor method
        return details::ImageNdGetter<T, inA, inB>(array, xy0);
    }
} // Close namespace ndarray

 #endif // LSST_AFW_GEOM_SPANSETFUNCTORGETTERS_H
