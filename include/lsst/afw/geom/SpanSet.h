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

#ifndef LSST_AFW_GEOM_SPANSET_H
#define LSST_AFW_GEOM_SPANSET_H

#include <vector>
#include <algorithm>
#include <functional>
#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/SpanSetFunctorGetters.h"

namespace lsst { namespace afw { namespace geom {

enum class Stencil { CIRCLE, BOX, MANHATTAN };

class SpanSet {
 public:
    typedef std::vector<Span>::const_iterator const_iterator;
    typedef std::vector<Span>::size_type size_type;
    typedef Span value_type;
    typedef value_type const & const_reference;
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;
    const_reference front() const;
    const_reference back() const;
    size_type size() const;
    bool empty() const;

    /// Default constructor creates a null set, useful as a placeholder and for the move constructor
    SpanSet();

    /// Construct a SpanSet from a region defined by a box
    explicit SpanSet(Box2I const & box);

    /// Construct a SpanSet from a container of Spans
    template <typename iter>
    SpanSet(iter begin, iter end, bool normalize = true):spanVector(begin, end) {
        if (normalize) {
            runNormalize();
        }
        initialize();
    }

    /// Construct a SpanSet from a std vector by copying
    explicit SpanSet(const std::vector<Span> & vec, bool normalize = true);

    /// Construct a SpanSet by moving the contents of a vector of Spans
    explicit SpanSet(std::vector<Span> && vec, bool normalize = true);

    /// Define class methods
    size_type getArea() const;

    Box2I getBBox() const;

    bool isContiguous() const;

    SpanSet shiftedBy(int x, int y) const;
    SpanSet shiftedBy(Extent2I const & offset) const;

    SpanSet clippedTo(Box2I const & box) const;

    SpanSet transformedBy(LinearTransform const & t) const;
    SpanSet transformedBy(AffineTransform const & t) const;
    SpanSet transformedBy(XYTransform const & t) const;

    bool overlaps(SpanSet const & other) const;

    bool contains(SpanSet const & other) const;
    bool contains(Point2I const & point) const;

    Point2D computeCentroid() const;

    ellipses::Quadrupole computeShape() const;

    SpanSet dilate(int r, Stencil s = Stencil::CIRCLE) const;
    SpanSet dilate(SpanSet const & other) const;

    SpanSet erode(int r, Stencil s = Stencil::CIRCLE) const;
    SpanSet erode(SpanSet const & other) const;

    template <typename Pixel, int inC>
    ndarray::Array<Pixel, 1, 1> flatten(ndarray::Array<Pixel, 2, inC>  & input,
                                                       Point2I const xy0 = Point2I()) const {
        // Populate a one dimensional array with the values from input at taken at the points of SpanSet
        ndarray::Array<Pixel, 1, 1> outputArray = ndarray::allocate(ndarray::makeVector(getArea()));
        outputArray.deep() = 0;
        flatten(outputArray, input, xy0);
        return outputArray;
    }

    template <typename Pixel, int outC, int inC>
    void flatten(
            ndarray::Array<Pixel, 1, outC> & output,
            ndarray::Array<Pixel, 2, inC> & input,
            Point2I const xy0 =  Point2I()) const {
        // Populate array output with values from input at positions given by SpanSet
        applyFunctor([](Point2I point, int n, typename details::FlatNdGetter<Pixel>::Reference out,
                        typename details::ImageNdGetter<Pixel, 2, inC>::Reference in) {out = in;},
                     ndarray::ndFlat(output), ndarray::ndImage(input, xy0));
    }

    template <typename Pixel, int inC>
    ndarray::Array<Pixel, 2, 2> unflatten(ndarray::Array<Pixel, 1, inC> & input) const {
        // Create a two dimensional array the size of the bounding box. Populate values from input, placed at
        // locations corresponding to SpanSet, offset by the lower corner of the bounding box
        ndarray::Array<Pixel, 2, 2> outputArray = ndarray::allocate(bBox.getHeight(), bBox.getWidth());
        outputArray.deep() = 0;
        unflatten(outputArray, input, Point2I(bBox.getMinX(), bBox.getMinY()));
        return outputArray;
    }

    template <typename Pixel, int outC, int inC>
    void unflatten(ndarray::Array<Pixel, 2, outC> & output,
                                  ndarray::Array<Pixel, 1, inC> & input,
                                  Point2I const xy0 = Point2I()) const {
        // Populate 2D ndarray output with values from input, at locations defined by SpanSet, optionally
        // offset by xy0
        applyFunctor([](Point2I point, int n, typename details::ImageNdGetter<Pixel, 2, outC>::Reference out,
                        typename details::FlatNdGetter<Pixel>::Reference in) {out = in;},
                        ndarray::ndImage(output, xy0), ndarray::ndFlat(input));
    }

    template <typename Functor, typename...T>
    void applyFunctor(Functor && func, T&&...x) const {
        /* Use a variadic template to take a functor object, and an arbitrary number
           of parameters. For each of the arguments, construct a Getter class using
           a function (makeGetter) which is overloaded to all the types applyFunctorImpl
           supports: Images, MaskedImages, Exposures, ndarrays, numeric values, and
           iterators. A note a ndarray can either be interpreted as a two dimensional
           image, or a one dimensional vector, as such the user must break the ambiguity
           by using either the ndarray::ndImage or ndarray::ndFlat functions on the array.
           The functor and the getters are then passed to the implementation of applyFunctor
           where the values of the input arguments are intelegently generated at each point
           in SpanSet, and passed to the functor object for evaluation. The functor object
           must operate inplace. The first two arguments of the functor must be the point
           in the SpanSet where the operation is occuring and an integer n representing
           the number of points that have been operated on. i.e. if the SpanSet has an
           area of 10 pixels, the functor will be passed a value of 9 on the last
           invocation. All other arguments to the functor are generated from the getters
           and will be supplied in the order in which they are passed to applyFunctor
         */
        applyFunctorImpl(func, details::makeGetter(x)...);
    }

    template <typename T>
    void setMask(lsst::afw::image::Mask<T> & target, T bitmask) const {
        // Use a lambda to set bits in a mask at the locations given by SpanSet
        auto targetArray = target.getArray();
        applyFunctor([](Point2I point, int n, typename details::ImageNdGetter<T, 2, 1>::Reference maskVal,
                        T bitmask){maskVal |= bitmask;}, ndarray::ndImage(targetArray), bitmask);
    }

    template <typename T>
    void clearMask(lsst::afw::image::Mask<T> & target, T bitmask) const {
        // Use a lambda to clear bits in a mask at the locations given by SpanSet
        auto targetArray = target.getArray();
        applyFunctor([](Point2I point, int n, typename details::ImageNdGetter<T, 2, 1>::Reference maskVal,
                        T bitmask){maskVal &= ~bitmask;}, ndarray::ndImage(targetArray), bitmask);
    }

    // SpanSet functions
    SpanSet intersect(SpanSet const & other) const;
    SpanSet intersectNot(SpanSet const & other) const;
    SpanSet union_(SpanSet const & other) const;

    /// Compareison Operators
    bool operator==(SpanSet const & other) const;
    bool operator!=(SpanSet const & other) const;

    static SpanSet spanSetFromShape(int r, Stencil s = Stencil::CIRCLE);

 private:
    void runNormalize();
    void initialize();
    inline bool spansOverlap(Span const & a, Span const & b, bool compareY = true) const {
        bool yTruth(true);
        if (compareY) {
            yTruth = a.getY() == b.getY();
        }
        return (yTruth && ((a.getMaxX() >= b.getMinX() && a.getMinX() <= b.getMinX()) ||
                (b.getMaxX() >= a.getMinX() && b.getMinX() <= a.getMinX())))
                ? true : false;
    }
    inline bool spansContiguous(Span const & a, Span const & b, bool compareY = true) const {
        bool yTruth(true);
        if (compareY) {
            yTruth = a.getY() == b.getY();
        }
        return (yTruth && ((a.getMaxX()+1 >= b.getMinX() && a.getMinX() <= b.getMinX()) ||
                (b.getMaxX()+1 >= a.getMinX() && b.getMinX() <= a. getMinX())))
                ? true: false;
    }
    SpanSet makeShift(int x, int y) const;

    template <typename F, typename ...T>
    void applyFunctorImpl(F f, T... x) const {
        /* Implementation for appling functors, loop over each of the spans, and then
         * each point. Use the get method in the getters to fetch the value and pass
         * the point, the iteration index, and the values to the functor
         */
        std::size_t n = 0;
        // make sure that the SpanSet is within the bounds of functor arguments
        details::variadicChecker(bBox, area, x...);
        for (auto const & spn : spanVector) {
            /* The following function call is only used so that the the variadic
             * template can be implicityly expanded. It is currently a
             * limitation in variadic templates that a function call on each
             * parameter can only happen in a function call or initializer list
             */
            // Set the current span in the getter, useful for optimizing value lookups
            details::variadicSetter(spn, x...);
            for (auto point : spn) {
                f(point, n, x.get()...);
                details::variadicIncrement(x...);
            }
            ++n;
        }
     }

    std::vector<Span> spanVector;
    Box2I bBox;
    std::size_t area;
};

 }}} // Close namespace lsst::afw::geom

#endif // LSST_AFW_GEOM_SPANSET_H
