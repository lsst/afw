// -*- lsst-c++ -*-
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
#ifndef LSST_AFW_GEOM_SpanRegion_h_INCLUDED
#define LSST_AFW_GEOM_SpanRegion_h_INCLUDED

#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/Box.h"

namespace lsst { namespace afw { namespace geom {

class SpanRegionInverse;
class SpanRegionBuilder;

/// Enum used to specify different kernels for dilate and erode operations.
enum class Stencil { CIRCLE, BOX, MANHATTAN };

/**
 *  A region in an image defined by a sequence of row Spans.
 *
 *  SpanRegion contains much of the functionality of the old Footprint class, with many minor
 *  interfaces differences and two major conceptual differences:
 *   - a SpanRegion does not contain peaks;
 *   - a SpanRegion is not necessarily contiguous.
 *  We'll still have a Footprint class, which will continue to hold peaks and be contiguous,
 *  but it will delegate much of its old capabilities to a SpanRegion member; the new Footprint
 *  will be little more than a combination of a SpanRegion and a PeakCatalog, with an additional
 *  guarantee that the SpanRegion it holds is contiguous.
 *
 *  Because a SpanRegion isn't guaranteed to be contiguous, we avoid many of the problems
 *  with set-topological and mathematical morphology (i.e. dilate/erode) operations on Footprints;
 *  while these methods had the potential to merge or split Footprints, they mostly had to operate
 *  on FootprintSets, not Footprints, which was inconvenient for users who either didn't care
 *  about contiguousness or could guarantee in advance that it would be preserved.  We'll still
 *  have to provide many of those operations on FootprintSets, of course, but many users will now
 *  prefer the SpanRegion versions of these operators, and we can implement the FootprintSet
 *  methods by delegating to the SpanRegion versions (along with calls to SpanRegion's split() and
 *  combine() methods).
 *
 *  SpanRegion will almost certainly be implemented using the copy-on-write pattern: it'll have
 *  a shared_ptr to a private Impl object, and any mutator will first check to see that it
 *  is the sole owner, and if not, make a deep-copy at that point.  This reflects the fact that
 *  modifications to SpanRegions are almost always atomic operations that require all of the
 *  spans to be rewritten, and most of the time we don't modify them at all.  The exception, of
 *  course, is when a SpanRegion is first being built, for which we've provided the
 *  SpanRegionBuilder helper class.  As a result, SpanRegion should typically be passed by reference
 *  or by value, not by shared_ptr (because it already has an internal shared_ptr, and there's no
 *  polymorphism, the only reason to use an external shared_ptr is if two objects need to share
 *  the same SpanRegion and one object needs to receive the modifications the other will make to
 *  the SpanRegion).
 *
 *  Perhaps more unusually, SpanRegion will be immutable from its Python interface; you'll note that
 *  all mutating methods below are within #ifndef SWIG blocks (as are a few other methods I don't
 *  think we can/should make available in Python).  That's because SpanRegion's most important use
 *  is as an attribute of Footprint - a ton of Footprint's functionality will be accessed via
 *  its getSpans() method (and, in Python, I think that should also be available as a .spans property),
 *  and there's simply no way to get Swig to protect the constness of the SpanRegion that returns.
 *  So we're left with several unpalatable options:
 *   - Ignore the constness failure, as we do most other places in the stack (which I think needs to
 *     change at some point).  That allows users to modify the internals of objects in ways the class
 *     authors attempted to disallow, leading to everything from silently incorrect behavior to
 *     segfaults.
 *   - Return a copy.  That's what Schema accessors in Table and Record objects do, and it's
 *     something we could do here (Schema, like SpanRegion, is copy-on-write, so those copies are
 *     generally cheap).  However, nothing stops a Python user from modifying the copy they got
 *     back, *thinking* that they're modifying e.g. the Footprint's internals.  Which again leads
 *     to silently incorrect (from the user's perspective) behavior (this happens all the time
 *     with Schemas)
 *   - <insert comment about replacing swig with something else here>
 *   - Make the Python-side object immutable, even if the C++ object isn't.  This still won't prevent
 *     Python users from passing what should be const references to Swigged C++ methods that take
 *     non-const references, but those are very rare compared to ways to modify a SpanRegion via
 *     its own methods.
 *  I think the last approach is best for SpanRegion particularly because it is so important as
 *  a member of another class, and all of its mutating methods have return-a-copy counterparts.
 *  I even think we can get the augmented assignment operators working naturally in Python without
 *  actually wrapping them, by having them reset the Python variable to a new C++ object (which
 *  Python itself allows us to prevent from working on properties, and will automatically be
 *  disallowed on objects returned directly by functions).
 */
class SpanRegion {
public:

    //@{
    /**
     *  STL container interface.
     *
     *  SpanRegion is iterable, but has no operator[] or rbegin/rend (because we could
     *  use std::forward_list or std::list instead of std::vector if we wanted to).
     *  We also have no non-const iterator, because we don't want to have to deal with
     *  maintaining invariants while exposing references to innards.
     *  Any mutating operation may invalidate iterators or Span references.
     */
    typedef <unspecified> const_iterator;
    typedef <unspecified> size_type;
    typedef Span value_type;
    typedef value_type const & const_reference;
    const_iterator begin() const;   // iterators available only via __iter__ in Python
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;
    const_reference front() const;
    size_type size() const;    // also available as __len__ in Python
    bool empty() const;        // also available as !__nonzero__ in Python
    void swap(SpanRegion & rhs);
    //@}

    /// Create an empty region.
    SpanRegion();

    /// Create from a box.
    explicit SpanRegion(Box2I const & box);

#ifndef SWIG
    /// Create from a special builder object (see SpanRegionBuilder).
    SpanRegion(SpanRegionBuilder && builder);
#endif

    /// SpanRegion is Copyable (almost certainly copy-on-write, but that's an implementation detail).
    SpanRegion(SpanRegion const & other);
#ifndef SWIG
    SpanRegion & operator=(SpanRegion const & other);
#endif

#ifndef SWIG
    /// SpanRegion is Moveable.
    SpanRegion(SpanRegion && other);
    SpanRegion & operator=(SpanRegion && other);
#endif

    /**
     *  Create from a range of iterators to Spans.
     *
     *  In Python, this will accept any Python iterable that yields spans.
     */
    template <typename Iter>
    static SpanRegion fromSpans(Iter first, Iter last);

#ifndef SWIG
    /// Create from an explicit set of Spans.
    static SpanRegion fromSpans(std::initializer_list<Span> spans);
#endif

    /// Create a single (possibly noncontiguous) SpanRegion from multiple (possibly overlapping) SpanRegions.
    static SpanRegion combine(std::vector<SpanRegion> const & inputs);

    /// Split into multiple contiguous footprints.
    std::vector<SpanRegion> split() const;

    //@{
    /**
     *  Add a new Span to the region.
     *
     *  If multiple Spans are to be added, consider using SpanRegionBuilder for
     *  a more efficient approach; these methods have additional overhead to
     *  maintain the bounding box, no-overlapping-Span constraint, and
     *  copy-on-write behavior for each Span.
     */
    void push_back(Span const & span);      // will be renamed to "append" in Python
    void addSpan(int y, int x0, int x1);
    //@}

    /// Return the number of pixels in the region (stored; updated when region is modified).
    size_type getArea() const;

    /// Return the smallest possible bounding box for the region (stored; updated when region is modified).
    Box2I getBBox() const;

    /// Return true if all pixels are connected (diagonal-only connections don't count).
    bool isContiguous() const;

    /// Move the SpanRegion by the given offset.
    SpanRegion shiftedBy(Extent2I const & offset) const; // return shifted copy
#ifndef SWIG
    SpanRegion & shiftBy(Extent2I const & offset);               // in-place
#endif

    /// Faster implementation of *this & SpanRegion(box).
    SpanRegion clippedTo(Box2I const & box) const; // return clipped copy
#ifndef SWIG
    SpanRegion & clipTo(Box2I const & box);                // in-place
#endif

    /// Transform to a new coordinate system (approximate).
    SpanRegion transformedBy(XYTransform const & t) const;     // return transformed copy
    SpanRegion transformedBy(AffineTransform const & t) const; // return transformed copy
#ifndef SWIG
    SpanRegion & transformBy(XYTransform const & t);                   // in-place
    SpanRegion & transformBy(AffineTransform const & t);               // in-place
#endif

    /// Return true if any pixel in this is in other.
    bool overlaps(SpanRegion const & other) const;

    /// Return true if all pixels in other are also in this.
    bool contains(SpanRegion const & other) const;

    /// Return true if the region contains the given point.
    bool contains(Point2I const & point) const;

    /// Return the centroid of the region, computed as the mean of pixel centers.
    Point2D computeCentroid() const;

    /// Return the 2nd-moments ellipse of the Footprint
    ellipses::Quadrupole computeShape() const;

    // Equality comparison.
    bool operator==(SpanRegion const & rhs) const;
    bool operator!=(SpanRegion const & rhs) const;

    //@{
    /**
     *  Mathematical morphology operations.
     *
     *  These are essentially binary convolution (dilate) or deconvolution (erode) operations,
     *  which take the place of the old "grow" methods (erode and dilate are much more precise
     *  mathematical names for these operations).
     */
    SpanRegion dilatedBy(int r, Stencil=Stencil.CIRCLE) const; // return dilated copy
    SpanRegion erodedBy(int r, Stencil=Stencil.CIRCLE) const;  // return eroded copy
#ifndef SWIG
    SpanRegion & dilateBy(int r, Stencil=Stencil.CIRCLE);      // in-place
    SpanRegion & erodeBy(int r, Stencil=Stencil.CIRCLE);       // in-place
#endif
    //@}

    //@{
    /**
     *  Topological operations
     *
     *  All of these have versions that return new objects and versions that operate in-place, but
     *  because we expect the class to be copy-on-write and we can't actually write the in-place
     *  versions in a more efficient waya, the performance of the two versions will frequently be
     *  very similar.
     */
    // Return the inverse of the region (see SpanRegionInverse for more information).
    SpanRegionInverse operator~() const;

    // Compute the intersection of two regions.
    SpanRegion operator&(SpanRegion const & rhs) const;
#ifndef SWIG
    SpanRegion & operator&=(SpanRegion const & rhs);
#endif

    /// Compute the union of two regions.
    SpanRegion operator|(SpanRegion const & rhs) const;
#ifndef SWIG
    SpanRegion & operator|=(SpanRegion const & rhs);
#endif

    // Compute the intersection of a region and the inverse of a region.
    SpanRegion operator&(InverseSpanRegion const & rhs) const;
#ifndef SWIG
    SpanRegion & operator&=(InverseSpanRegion const & rhs);
#endif
    //@}

    //@[
    /**
     *  Copy a 2-d image-like array to/from a 1-d array containing only the values in the SpanRegion.
     *
     *  These are equivalent to to the current afw::detection::flattenArray/afw::detection::expandArray
     *  free functions, but I've switched the order of arguments for the overloads that take an output
     *  array to match our our coding standards (which say output arguments should go first).  I'm
     *  ambivalent about whether that's a good idea, in part because I'm not sure it's a good thing to
     *  try to standardize.  I've also renamed the "expand" operation to "unflatten", as I think that's
     *  a little less ambiguous.
     *
     *  We could also add overloads that take/write images, but it is trivial to call these with
     *  images as-is (just pass getArray() and getXY0()), and we haven't needed them so far.
     */
    template <typename Pixel, int inC>
    ndarray::Array<Pixel,1,1> flatten(
        ndarray::Array<Pixel const,2,inC> const & input,
        Point2I const & xy0=Point2I()
    ) const;

    template <typename Pixel, int outC, int inC>
    void flatten(
        ndarray::Array<Pixel,1,outC> const & output,
        ndarray::Array<Pixel const,2,inC> const & input,
        Point2I const & xy0=Point2I()
    ) const;

    template <typename Pixel, int outC>
    void unflatten(
        ndarray::Array<Pixel,2,outC> const & output,
        ndarray::Array<Pixel const,1,inC> const & input,
        Point2I const & xy0=Point2I()
    ) const;
    //@}

#ifndef SWIG
    //@{
    /**
     *  Apply a functor to one or more image-like or flattened arrays within the SpanRegion area.
     *
     *  Unlike the current FootprintFunctor, functors do not need to inherit from a particular base
     *  class or implement multiple methods - all that's needed is a function call operator with
     *  the appropriate signature (so function pointers, lambdas, and std::function objects may
     *  all be used).  This approach also differs from the current FootprintFunctor in not
     *  requiring a virtual function call for every pixel, which may dramatically improve the
     *  compiler's ability to optimize some operations.
     *
     *  These methods all take one or more "target" arguments, which can be:
     *   - an Image, Mask, or MaskedImage object that fully contains the SpanRegion's pixels.
     *   - a 1-d ndarray::Array object, representing a flattened view of the
     *     SpanRegion's pixels (see flatten() and unflatten()).
     *
     *  The functor is called on every pixel, with a Point2I object as its first argument, followed
     *  by a pixel reference for every object (T& for Image<T> or Array<T,...>, but afw::image::Pixel for
     *  MaskedImage).  Each pixel reference will be const if and only if the corresponding target
     *  argument is.
     *
     *  Functors may have state, which they can use to compute aggregate quantities.  Because we're using
     *  C++11's universal argument syntax, we can detect whether functors are being passed by reference or
     *  by value and handle either case appropriately.  It'd would probably also be possible to replace the
     *  explicit target arguments with variadic templates, but I don't think that's worth the trouble here.
     *
     *  These methods will not be available in Python.  They aren't likely to be needed there, and both
     *  the generic-functor interface and the generic-target interface would be very hard to translate
     *  to Python.
     *
     *  These methods are expected to replace the footprintAndMask, intersectMask, and copyWithinFootprint
     *  methods of the old Footprint class - those can now be written more clearly and generically by
     *  combining a lambda with one of these methods (we won't be able to call them from Python, but
     *  we currently don't).  Several methods that *could* be replaced by these methods that we do
     *  utilize in Python are defined explicitly below, both to provide a less verbose interface in C++
     *  and to make them available in Python.
     */
    template <typename Functor, typename T1>
    void applyFunctor(Functor && func, T1 && target1) const;

    template <typename Functor, typename T1, typename T2>
    void applyFunctor(Functor && func, T1 && target1, T2 && target2) const;

    template <typename Functor, typename T1, typename T2, typename T3>
    void applyFunctor(Functor && func, T1 && target1, T2 && target2, T3 && target3) const;
    //@}
#endif

    /// OR the given bitmask into the target mask for all pixels within the region.
    template <typename T>
    void setMask(afw::image::Mask<T> & target, T bitmask) const;

    /// Clear the given bits (&= ~bitmask) for all pixels within the region.
    template <typename T>
    void clearMask(afw::image::Mask<T> & target, T bitmask) const;

};


/**
 *  A helper class that represents the topological inverse of a SpanRegion.
 *
 *  Because the inverse has infinite area, you can't do anything with it
 *  besides intersect it with another SpanRegion - but that's the whole
 *  reason it exists:
 *  @code
 *  a &= (~b)
 *  @endcode
 *  is much clearer than e.g.
 *  @code
 *  a.assignDifferenceIntersection(b).
 *  @endcode
 */
class SpanRegionInverse {
public:

    // Return the original region this is an inverse of.
    SpanRegion operator~() const;

    // Compute the intersection of a region and the inverse of a region.
    SpanRegion operator&(SpanRegion const & rhs) const;

};

#ifndef SWIG
/**
 *  A helper class that allows SpanRegions to be built more efficiently.
 *
 *  SpanRegionBuilder is essentially an "unnormalized" SpanRegion - it doesn't
 *  store its own bounding box or area, and it doesn't guarantee that its
 *  Spans don't overlap.  Since a regular SpanRegion has to maintain all of
 *  those things every time a new Span is added, adding Spans directly to
 *  a SpanRegion is slow.  By using SpanRegionBuilder, we do all of that
 *  only once, when we build the SpanRegion from the SpanRegionBuilder.
 *  Because we can assume anyone building a SpanRegion span-by-span in
 *  Python doesn't care about performance, this class is not exposed to Python.
 */
class SpanRegionBuilder {
public:

    // Construct an empty SpanRegionBuilder.
    SpanRegionBuilder();

    // SpanRegionBuilder is not copyable (only moveable).
    SpanRegionBuilder(SpanRegionBuilder const &) = delete;
    SpanRegionBuilder & operator=(SpanRegionBuilder const &) = delete;

    // Add a new Span to the in-progress SpanRegion.
    void push_back(Span const & span);
    void addSpan(int y, int x0, int x1);

    // Compute the bounding box of the in-progress SpanRegion (computed on-the-fly).
    Box2I computeBBox() const;

    // Compute the number of pixels in the in-progress SpanRegion (computed on-the-fly).
    int computeArea() const;

};
#endif

}}} // namespace lsst::afw::region

#endif // !LSST_AFW_GEOM_SpanRegion_h_INCLUDED
