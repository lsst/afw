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
/*
  Implements a compact representation of a collection of pixels
 */

#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/geom/SpanSetFunctorGetters.h"

namespace lsst { namespace afw { namespace geom { namespace details {
    /* Functor object to be used with maskToSpanSet function
     */
    template <typename T>
    class AnyBitSetFunctor {
     public:
        bool operator()(T const & pixelValue) {
            return pixelValue !=0;
        }
    };

}}}} // end lsst::afw::geom::details



namespace lsst { namespace afw { namespace geom {

/** @brief An enumeration class which describes the shapes

           A stencil is a shape used in creating SpanSets, erosion kernels,
           or dilation kernels. CIRCLE creates a circle shape, BOX
           creates a box shape, and MANHATTAN creates a diamond shape.
 */
enum class Stencil { CIRCLE, BOX, MANHATTAN };

// Forward declaration of the SpanSet class
class SpanSet;

/** @brief Create a SpanSet from a mask.
 *
 * Create a SpanSet from a class. The default behaivor is to include any pixels which have any
 * bits set. More complex selection/filtering of bit patterns can be done by supplying a comparator
 * function.
 *
 * @param mask - mask to convert to a SpanSet
 * @param comparator - Functor object to use in the decision to include pixel in SpanSet. Should return
 *                     true when a given pixel in the mask should be part of the SpanSet, and false
 *                     otherwise. The functor takes a single value taken from the mask at the
 *                     pixel under consideration. Defaults to evaluating true if the mask has bits set,
 *                     and false otherwise.
 *
 * @tparam T - Pixel type of the Mask
 * @tparam F - Type of the functor
 */
template <typename T, typename UnaryPredicate = details::AnyBitSetFunctor<T>>
std::shared_ptr<geom::SpanSet> maskToSpanSet(image::Mask<T> const & mask,
                               UnaryPredicate p = details::AnyBitSetFunctor<T>() ) {
    std::vector<Span> tempVec;
    std::size_t startValue{0};
    bool started{false};
    auto const & maskArray = mask.getArray();
    auto const & minPoint = mask.getBBox().getMin();
    auto dimensions = maskArray.getShape();
    for (size_t y = 0; y < dimensions[0]; ++y) {
        startValue = 0;
        started = false;
        for (size_t x = 0; x < dimensions[1]; ++x) {
            // If a new span has not been started, and a given x matches the functor condition
            // start a new span
            if (p(maskArray[y][x]) && !started) {
                started = true;
                startValue = x;
            }
            // If a span has been started, and the functor condition is false, that means the
            // Span being created should be stopped, and appended to the Span vector
            // Offset the x, y position by the minimum point of the mask
            else if (started && !p(maskArray[y][x])) {
                tempVec.push_back(Span(y + minPoint.getY(),
                                       startValue + minPoint.getX(),
                                       x-1+minPoint.getX()));
                started = false;
            }
            // If this is the last value in the Span's x range (dimension minux one), and started
            // is still true that means the last value does not evaluate false in the functor
            // and should be included in the Span under construction. The Span should be completed
            // and added to the Span Vector before the next span is concidered.
            // offset the x, y position by the minimum point of the mask
            if (started && x == dimensions[1] - 1) {
                tempVec.push_back(Span(y + minPoint.getY(),
                                       startValue + minPoint.getX(),
                                       x + minPoint.getX()));
            }
        }
    }

    // construct a SpanSet from the spans determined above
    return std::make_shared<SpanSet>(std::move(tempVec));
}

/**
 * @brief A compact representation of a collection of pixels
 *
 * A SpanSet is a collection of Span classes. As each Span encodes a range of pixels
 * on a given row, a SpanSet represents an arbitrary collection of pixels on an image.
 * The SpanSet class also contains mathematical set style operators, for working with
 * the collection of pixels, and helper functions which make use of the area defined
 * to perform localized actions
 */
class SpanSet : public afw::table::io::PersistableFacade<lsst::afw::geom::SpanSet>,
                public afw::table::io::Persistable{
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

    /** @brief Default constructor
     *
     * Construct a null SpanSet, with zero size. This is useful as a placeholder
     */
    SpanSet();

    /** @brief Construct a SpanSet from a box
     *
     * @param box - A integer box that defines the shape for which a span set should be created
     */
    explicit SpanSet(Box2I const & box);

    /** @brief Construct a SpanSet from an iterator
     *
     * This constructor accepts the begin and end points of an arbitrary iterator
     * of a container which contains previously created Spans. These Spans are
     * used to construct the SpanSet.
     *
     * @param begin - Beginning iterator of a container of Spans
     * @param end - End iterator of a container of Spans
     * @param normalize - Controls if the constructor attempts to merge connected or
                          overlapping spans. Defaults to true. Set to false to save
                          computational time when the container is sure to already
                          be normalized.
     */
    template <typename iter>
    SpanSet(iter begin, iter end, bool normalize = true):_spanVector(begin, end) {
        // Return a null SpanSet if spanVector is 0
        if (_spanVector.size() == 0) {
            _bBox = geom::Box2I();
            _area = 0;
        } else {
            if (normalize) {
                _runNormalize();
            }
            _initialize();
        }
    }

    /** @brief Construct a SpanSet from a std vector by copying
     *
     * This constructor accepts a standard vector which contains already created
     * Spans. These Spans are copied into the internal container of spans.
     *
     * @param vec - Standard vector containing Spans
     * @param normalize - Controls if the constructor attempts to merge connected or
                          overlapping spans. Defaults to true. Set to false to save
                          computational time when the container is sure to already
                          be normalized.
    */
    explicit SpanSet(const std::vector<Span> & vec, bool normalize = true);


    /** @brief Construct a SpanSet from a std vector from a move
     *
     * This constructor accepts a standard vector r-reference which contains already
     * created Spans. These Spans become the internal container of spans.
     *
     * @param vec - Standard vector containing Spans
     * @param normalize - Controls if the constructor attempts to merge connected or
                          overlapping spans. Defaults to true. Set to false to save
                          computational time when the container is sure to already
                          be normalized.
    */
    explicit SpanSet(std::vector<Span> && vec, bool normalize = true);

    // Explicitly delete copy and move constructors
    SpanSet(const SpanSet & other) = delete;
    SpanSet(SpanSet && other) = delete;

    // Define class methods
    /** @brief Return the number of pixels in the SpanSet
     */
    size_type getArea() const;

    /** @brief Return a new integer box which is the minimum size to contain
               the pixels
     */
    Box2I getBBox() const;

    /** @brief Defines if the SpanSet is simply contiguous
     *
     * If the pixels can be traversed in such a way that every pixel can be reached
     * without going over a pixel not contained in the SpanSet this method will return
     * true. If the SpanSet is disjoint aka the above is not true and there is more
     * than one region, returns false.
     */
    bool isContiguous() const;

    /** @brief Return a new SpanSet shifted by specified amount
     *
     * @param x - number of pixels to shift in x dimension
     * @param y - number of pixels to shift in y dimension
     */
    std::shared_ptr<SpanSet> shiftedBy(int x, int y) const;
    /** @brief Return a new SpanSet shifted by specified amount
     *
     * @param offset - integer extent which specifies amount to offset in x and y
     */
    std::shared_ptr<SpanSet> shiftedBy(Extent2I const & offset) const;

    /** @brief Return a new SpanSet which has all pixel values inside specified box
     *
     * @param box - Integer box specifying the bounds for which all pixels must be within
     */
    std::shared_ptr<SpanSet> clippedTo(Box2I const & box) const;

    /** @brief Return a new SpanSet who's pixels are the product of applying the specified
               transformation
     *
     * @param t - A linear transform object which will be used to map the pixels
     */
    std::shared_ptr<SpanSet> transformedBy(LinearTransform const & t) const;

    /** @brief Return a new SpanSet who's pixels are the product of applying the specified
               transformation
     *
     * @param t - An affine transform object which will be used to map the pixels
     */
    std::shared_ptr<SpanSet> transformedBy(AffineTransform const & t) const;

    /** @brief Return a new SpanSet who's pixels are the product of applying the specified
               transformation
     *
     * @param t - A XY transform object which will be used to map the pixels
     */
    std::shared_ptr<SpanSet> transformedBy(XYTransform const & t) const;

    /** @brief Specifies if this SpanSet overlaps with another SpanSet
     *
     * @param other - A SpanSet for which overlapping comparison will be made
     */
    bool overlaps(SpanSet const & other) const;

    /** @brief Check if a SpanSet instance entirely contains another SpanSet
     *
     * @param other - The SpanSet who's membership is to be tested for
     */
    bool contains(SpanSet const & other) const;

    /** @brief Check if a point is contained within the SpanSet instance
     *
     * @param point - An integer point object for which membership is to be tested
     */
    bool contains(Point2I const & point) const;

    /** @brief Compute the point about which the SpanSet is symmetrically distributed
     */
    Point2D computeCentroid() const;

    /** @brief Compute the shape parameters for the distribution of points in the SpanSet
     */
    ellipses::Quadrupole computeShape() const;

    /** @brief Perform a set dilation operation, and return a new object
     *
     * Dilate a SpanSet with a kernel specified with the stencil parameter
     *
     * @param r - radius of the stencil, the length is inclusive i.e. 3 ranges from -3 to 3
     * @param s - must be an enumeration of type geom::Stencil. Specifies the shape of the
                  dilation kernel. May be CIRCLE, MANHATTAN, or BOX
     */
    std::shared_ptr<SpanSet> dilate(int r, Stencil s = Stencil::CIRCLE) const;

    /** @brief Perform a set dilation operation, and return a new object
     *
     * Dilate a SpanSet with a kernel specified by another SpanSet
     *
     * @param other - A SpanSet which specifies the kernel to use for dilation
     */
    std::shared_ptr<SpanSet> dilate(SpanSet const & other) const;

    /** @brief Perform a set erosion, and return a new object
     *
     * Erode a SpanSet with a kernel specified with the stencil parameter
     *
     * @param r - radius of the stencil, the length is inclusive i.e. 3 ranges from -3 to 3
     * @param s - must be an enumeration of type geom::Stencil. Specifies the shape of the
                  erosion kernel. May be CIRCLE, MANHATTAN, or BOX
     */
    std::shared_ptr<SpanSet> erode(int r, Stencil s = Stencil::CIRCLE) const;

    /** @brief Perform a set erosion operation, and return a new object
     *
     * Erode a SpanSet with a kernel specified by another SpanSet
     *
     * @param other - A SpanSet which specifies the kernel to use for erosion
     */
    std::shared_ptr<SpanSet> erode(SpanSet const & other) const;

    /** @brief Create 1d array at points given by SpanSet
     *
     * Take values from an array at points defined by SpanSet and use them to populate a new 1D ndarray
     *
     * @param input - The ndarray from which the values will be taken
     * @param xy0 - A point object with is used as the origin point for the SpanSet coordinate system
     *
     * @tparam Pixel - The datatype for the ndarray
     * @tparam inC - Number of guaranteed row-major contiguous dimensions, starting from the end
     */
    template <typename Pixel, int inC>
    ndarray::Array<Pixel, 1, 1> flatten(ndarray::Array<Pixel, 2, inC>  & input,
                                        Point2I const & xy0 = Point2I()) const {
        // Populate a one dimensional array with the values from input taken at the points of SpanSet
        ndarray::Array<Pixel, 1, 1> outputArray = ndarray::allocate(ndarray::makeVector(getArea()));
        outputArray.deep() = 0;
        flatten(outputArray, input, xy0);
        return outputArray;
    }

    /** @brief Populate 1d array at points given by SpanSet
     *
     * Take values from an array at points defined by SpanSet and use them to populate a new 1D ndarray
     *
     * @param output - The 1d ndarray which will be populated with output parameters, will happen in place
     * @param input - The ndarray from which the values will be taken
     * @param xy0 - A point object which is used as the origin point for the SpanSet coordinate system
     *
     * @tparam Pixel - The data-type for the ndarray
     * @tparam outC - Number of guaranteed row-major contiguous dimensions in the output array,
                      starting from the end
     * @tparam inC - Number of guaranteed row-major contiguous dimensions in the input array,
                     starting from the end
     */
    template <typename Pixel, int outC, int inC>
    void flatten(
            ndarray::Array<Pixel, 1, outC> const & output,
            ndarray::Array<Pixel, 2, inC> const & input,
            Point2I const & xy0 =  Point2I()) const {
        auto ndAssigner = []
                          (Point2I const & point,
                           typename details::FlatNdGetter<Pixel>::Reference out,
                           typename details::ImageNdGetter<Pixel, 2, inC>::Reference in)
                          {out = in;};
        // Populate array output with values from input at positions given by SpanSet
        applyFunctor(ndAssigner, ndarray::ndFlat(output), ndarray::ndImage(input, xy0));
    }

    /** @brief Create 2d array at pionts given by SpanSet
     *
     * Take values from a 1d array and insert them in a new 2D array at points defined by the SpanSet,
     * offset by the lower left hand corner of the bounding box of the SpanSet
     *
     * @param input - The ndarray from which the values will be taken
     *
     * @tparam Pixel - The data-type for the ndarray
     * @tparam inC - Number of guaranteed row-major contiguous dimensions, starting from the end
     */
    template <typename Pixel, int inC>
    ndarray::Array<Pixel, 2, 2> unflatten(ndarray::Array<Pixel, 1, inC> & input) const {
        // Create a two dimensional array the size of the bounding box. Populate values from input, placed at
        // locations corresponding to SpanSet, offset by the lower corner of the bounding box
        ndarray::Array<Pixel, 2, 2> outputArray = ndarray::allocate(_bBox.getHeight(), _bBox.getWidth());
        outputArray.deep() = 0;
        unflatten(outputArray, input, Point2I(_bBox.getMinX(), _bBox.getMinY()));
        return outputArray;
    }

    /** @brief Populate 2d array at points given by SpanSet
     *
     * Take values from a 1d array and insert them in a new 2D array at points defined by the SpanSet,
     * offset by the xy0 parameter
     *
     * @param output - The 1d ndarray which will be populated with output parameters, will happen in place
     * @param input - The ndarray from which the values will be taken
     * @param xy0 - A point object with is used as the origin point for the SpanSet coordinate system
     *
     * @tparam Pixel - The datatype for the ndarray
     * @tparam outC - Number of guaranteed row-major contiguous dimensions in the output array,
                      starting from the end
     * @tparam inC - Number of guaranteed row-major contiguous dimensions in the input array,
                     starting from the end
     */
    template <typename Pixel, int outC, int inC>
    void unflatten(ndarray::Array<Pixel, 2, outC> & output,
                                  ndarray::Array<Pixel, 1, inC> & input,
                                  Point2I const & xy0 = Point2I()) const {
        // Populate 2D ndarray output with values from input, at locations defined by SpanSet, optionally
        // offset by xy0
        auto ndAssigner = []
                          (Point2I const & point,
                           typename details::ImageNdGetter<Pixel, 2, outC>::Reference out,
                           typename details::FlatNdGetter<Pixel>::Reference in)
                          {out = in;};
        applyFunctor(ndAssigner, ndarray::ndImage(output, xy0), ndarray::ndFlat(input));
    }

    /** @brief Apply functor on individual elements from the supplied parameters
     *
     * Use a variadic template to take a functor object, and an arbitrary number of parameters.
     * Parameters may be of type(s) Image, MakedImage, ndarray, numeric value and generic iterators.
     * For most of these types bound checking is done to ensure execution safety, but because
     * a generic iterator can have any behavior, bounds checking is not possible. As such the iterator
     * must be valid for at least the number of pixels contained in the SpanSet. Numeric values are
     * also different in that they will not be iterated but the value will be passed to the functor
     * for each point in the SpanSet.

     * Because a ndarray can either be interpreted as either a two dimensional image, or a one
     * dimensional vector a user must break the ambiguity by using the functions ndarray::ndImage,
     * or ndarray::ndFlat respectively.
     *
     * The functor object must operate in-place on any data, no return values are captured. No exceptions
     * are handled, if a functor throws one it will propagate back out of the applyFunctor call, possibly
     * leaving the output array in an incomplete state. The first argument of the functor must be
     * a Point2I which will be the point in the SpanSet where the operation is occurring. All the
     * remaining arguments to the functor will be individual values generated from the input arguments. For two
     * dimensional types (Image, MaskedImage, ndarray) arguments will be the value taken from the input type
     * at the location given by the SpanSet. I.e. if the SpanSet has the point (2,3) in it, then a 2 dimensional
     * ndarray will have ndarray[2][3] passed to the functor. Thus all SpanSet coordinates must be addressable in
     * two dimensional types. For one dimensional types (ndarray, iterator) arguments will correspond to the
     * value at the location defined by number of pixels which have been iterated over. I.e. if there are 3 pixels
     * defined in the SpanSet a one dimensional ndarray will have the value ndarray[0] passed to the functor on
     * on the first functor call, ndarray[1] passed on the second call, and ndarray[2] passed on the last functor
     * call. The total length of a one dimensional type will be equal to the area of the SpanSet, and therefore
     * the data-type must be at least that length. The order of the parameters supplied to the functor will
     * be the same order as they are passed to the applyFunctor method.
     *
     * @param ...args - Variadic arguments, may be of type Image, MaskedImage, ndarrays, numeric values
     *                  and iterators. ndarrays must be specified as either image or vector with the
     *                  ndarray::ndImage or ndarray::ndFlat functions
     *
     * @tparam ...Args - Variadic type specification
     */
    template <typename Functor, typename...Args>
    // Normally std::forward would be used with a universal reference, however
    // this function does not use one because without std::forward the
    // compiler is forced to keep any r-value references alive for the
    // duration of the function call
    void applyFunctor(Functor && func, Args && ...args) const {
        /* Use a variadic template to take a functor object, and an arbitrary number
           of parameters. For each of the arguments, construct a Getter class using
           a function (makeGetter) which is overloaded to all the types applyFunctorImpl
           supports: Images, MaskedImages, Exposures, ndarrays, numeric values, and
           iterators. The functor and the getters are then passed to the implementation of
           applyFunctor where the values of the input arguments are intelligently
           generated at each point in SpanSet, and passed to the functor object for evaluation.
         */
        applyFunctorImpl(func, details::makeGetter(args)...);
    }

    /** @brief Set a Mask at pixels defined by the SpanSet
     *
     * @param target - Mask in which values will be set
     * @param bitmask - The bit pattern to set in the mask
     *
     * @tparam T - data-type of a pixel in the Mask plane
     */
    template <typename T>
    void setMask(lsst::afw::image::Mask<T> & target, T bitmask) const {
        // Use a lambda to set bits in a mask at the locations given by SpanSet
        auto targetArray = target.getArray();
        auto xy0 = target.getBBox().getMin();
        auto maskFunctor = []
                           (Point2I const & point,
                            typename details::ImageNdGetter<T, 2, 1>::Reference maskVal,
                            T bitmask)
                           {maskVal |= bitmask;};
        applyFunctor(maskFunctor, ndarray::ndImage(targetArray, xy0), bitmask);
    }

    /** @brief Unset a Maks at pixels defined by the SpanSet
     *
     * @param target - Mask in which a bit pattern will be unset
     * @param bitmask - The bit pattern to clear in the mask
     *
     * @tparam T - data-type of a pixel in the Mask plane
     */
    template <typename T>
    void clearMask(lsst::afw::image::Mask<T> & target, T bitmask) const {
        // Use a lambda to clear bits in a mask at the locations given by SpanSet
        auto targetArray = target.getArray();
        auto xy0 = target.getBBox().getMin();
        auto clearMaskFunctor = []
                                (Point2I const & point,
                                 typename details::ImageNdGetter<T, 2, 1>::Reference maskVal,
                                 T bitmask)
                                {maskVal &= ~bitmask;};
        applyFunctor(clearMaskFunctor, ndarray::ndImage(targetArray, xy0), bitmask);
    }

    // SpanSet functions
    /** @brief Determine the common points between two SpanSets, and create a new SpanSet
     *
     * @param other - The other SpanSet with which to intersect with
     */
    std::shared_ptr<SpanSet> intersect(SpanSet const & other) const;

    /** @brief Determine the common points between a SpanSet and a Mask with a given bit pattern
     *
     * @param other - Mask with which to calculate intersection
     * @param bitmask - The bit value to concider when intersecting
     *
     * @tparam T - Pixel type of the Mask
     */
    template <typename T>
    std::shared_ptr<SpanSet> intersect(image::Mask<T> const & other, T const & bitmask) const {
        auto comparator = [bitmask]
                          (T pixelValue)
                          {return (pixelValue & bitmask) == bitmask;};
        auto spanSetFromMask = geom::maskToSpanSet(other, comparator);
        return intersect(*spanSetFromMask);
    }

    /** @brief Determine the common points between a SpanSet and the logical inverse of a second SpanSet
     *         and return them in a new SpanSet.
     *
     * @param other - The spanset which will be logically inverted when computing the intersection
     */
    std::shared_ptr<SpanSet> intersectNot(SpanSet const & other) const;

    /** @brief Determine the common points between a SpanSet and the logical inverse of a Mask for a
               given bit pattern
     *
     * @param other - Mask with which to calculate instersection
     * @param bitmask - The bit value to concider when intersecting
     *
     * @tparam T - Pixel type of the Mask
     */
    template <typename T>
    std::shared_ptr<SpanSet> intersectNot(image::Mask<T> const & other, T const & bitmask) const {
        auto comparator = [bitmask]
                          (T pixelValue)
                          {return (pixelValue & bitmask) == bitmask;};
        auto spanSetFromMask = geom::maskToSpanSet(other, comparator);
        return intersectNot(*spanSetFromMask);
    }

    /** @brief Create a new SpanSet that contains all points from two SpanSets
     *
     * @param other - The SpanSet from which the union will be calculated
     */
    std::shared_ptr<SpanSet> union_(SpanSet const & other) const;


    /** @brief Determine the union between a SpanSet and a Mask for a given bit pattern
     *
     * @param other - Mask with which to calculate instersection
     * @param bitmask - The bit value to concider when intersecting
     *
     * @tparam T - Pixel type of the Mask
     */
    template <typename T>
    std::shared_ptr<SpanSet> union_(image::Mask<T> const & other, T const & bitmask) const {
        auto comparator = [bitmask]
                          (T pixelValue)
                          {return (pixelValue & bitmask) == bitmask;};
        auto spanSetFromMask = geom::maskToSpanSet(other, comparator);
        return union_(*spanSetFromMask);
    }

    // Comparison Operators
    /** @brief Compute equality between two SpanSets
     *
     * @param other - The SpanSet for which equality will be computed
     */
    bool operator==(SpanSet const & other) const;

    /* @brief Compute inequality between two SpanSets
     *
     * @param other - The SpanSet for which inequality will be computed
     */
    bool operator!=(SpanSet const & other) const;

    /** @brief Factory function for creating SpanSets from a Stencil
     *
     * @param r - radius of the stencil, the length is inclusive i.e. 3 ranges from -3 to 3
     * @param s - must be an enumeration of type geom::Stencil. Specifies the shape of the
                  newly created SpanSet. May be CIRCLE, MANHATTAN, or BOX
    */
    static std::shared_ptr<geom::SpanSet> spanSetFromShape(int r, Stencil s = Stencil::CIRCLE);

    /** @brief Split a discontinuous SpanSet into multiple SpanSets which are contiguous
     */
    std::vector<std::shared_ptr<geom::SpanSet>> split() const;

    bool isPersistable() const override { return true; }

    /**
     * @brief Select pixels within the SpanSet which touch its edge
     *
     */
     std::shared_ptr<geom::SpanSet> findEdgePixels() const;

private:
    /* Returns the name used by the persistence layer to identify the SpanSet class
     */
    std::string getPersistenceName() const override;

    /* Return a string corresponding to the python module that SpanSets lives in
     */
    inline std::string getPythonModule() const override { return "lsst.afw.geom"; }

    /* Writes the representation of the class out to an output archive
     */
    void write(OutputArchiveHandle & handle) const override;

    /* A class which is used by the persistence layer to restore SpanSets from an archive
     */
    friend class SpansSetFactory;

    /* A function to combine overlapping Spans in a SpanSet into a single Span
     */
    void _runNormalize();

    /* Initializes the SpanSet class. Contains code that is common to multiple constructors
     */
    void _initialize();

    /* Label Spans according to contiguous group. If the SpanSet is contiguous, all Spans will be labeled 1.
     * If there is more than one group each group will receive a label one higher than the previous.
     */
    void _label(geom::Span const & spn, std::vector<std::size_t> & labelVector, std::size_t currentLabel) const;
    std::pair<std::vector<std::size_t>, std::size_t> _makeLabels() const;

    /* Determine if two spans overlap
     *
     * a - First Span in comparison
     * b - Second Span in comparison
     * compareY - a boolean to control if the comparison takes into account the y position of the spans
     */
    inline bool spansOverlap(Span const & a, Span const & b, bool compareY = true) const {
        bool yTruth(true);
        if (compareY) {
            yTruth = a.getY() == b.getY();
        }
        return (yTruth && ((a.getMaxX() >= b.getMinX() && a.getMinX() <= b.getMinX()) ||
                (b.getMaxX() >= a.getMinX() && b.getMinX() <= a.getMinX())))
                ? true : false;
    }


    /* Determine if two spans are contiguous, that is they can overlap or the end of one span is
     * one pixel before the beginning of the next
     *
     * a - First Span in comparison
     * b - Second Span in comparison
     * compareY - a boolean to control if the comparison takes into account the y position of the spans
     */
    inline bool spansContiguous(Span const & a, Span const & b, bool compareY = true) const {
        bool yTruth(true);
        if (compareY) {
            yTruth = a.getY() == b.getY();
        }
        return (yTruth && ((a.getMaxX()+1 >= b.getMinX() && a.getMinX() <= b.getMinX()) ||
                (b.getMaxX()+1 >= a.getMinX() && b.getMinX() <= a. getMinX())))
                ? true: false;
    }
    std::shared_ptr<SpanSet> makeShift(int x, int y) const;

    template <typename F, typename ...T>
    void applyFunctorImpl(F f, T... args) const {
        /* Implementation for applying functors, loop over each of the spans, and then
         * each point. Use the get method in the getters to fetch the value and pass
         * the point, and the values to the functor
         */
        // make sure that the SpanSet is within the bounds of functor arguments
        details::variadicBoundChecker(_bBox, _area, args...);
        for (auto const & spn : _spanVector) {
            // Set the current span in the getter, useful for optimizing value lookups
            details::variadicSpanSetter(spn, args...);
            for (auto point : spn) {
                f(point, args.get()...);
                details::variadicIncrementPosition(args...);
            }
        }
     }

    // Vector to hold the Spans contained in the SpanSet
    std::vector<Span> _spanVector;

    // Box that is large enough to bound all pixels in the SpanSet
    Box2I _bBox;

    // Number of pixels in the SpanSet
    std::size_t _area;
};

 }}} // Close namespace lsst::afw::geom

#endif // LSST_AFW_GEOM_SPANSET_H
