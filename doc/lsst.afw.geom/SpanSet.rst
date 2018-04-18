#######
SpanSet
#######

A ``SpanSet`` is a class that represents regions of pixels on an image.  As the
name implies, a ``SpanSet`` is a collection of ``Span`` objects which behave as
a mathematical set. The class provides methods for doing set operations, as well
as convenience method for operating on data products at the locations defined
by the ``SpanSet``.

Creating SpanSets
=================

The following example shows a few different ways to create a ``SpanSet`` in
Python (C++ is similar but with differences in syntax).

.. code-block:: python

    from lsst.afw.geom import Span, SpanSet, Stencil

    # Construct a SpanSet from a list of Spans
    # Spans are constructed as y, begin x, end x coordinates
    spanList = [Span(1, 0, 2), Span(2, 0, 2), Span(3, 0, 2)]
    spanSetFromList = SpanSet(spanList)

    # Construct the same SpanSet from a shape
    radius = 1
    spanSetFromShape = SpanSet.fromShape(radius, Stencil.BOX, offset=(1, 2))

    # Verify that the two are the same
    assert(spanSetFromList == spanSetFromShape)

fromShape
---------

The last example introduced a useful factory method for creating ``SpanSets``
that are worth exploring further. ``fromShape`` can be called in one of two
ways, specifying a radius, stencil (defaults to a circle), and an offset
(defaults to 0, 0), or passing in an ``afw.geom.ellipse.Ellipse`` object. There
are three options when creating a ``SpanSet`` from a ``afw.geom.Stencil``;
``BOX``, ``CIRCLE``, and ``MANHATTAN`` (a diamond). Each shape is centered on
the 0, 0 pixel and extends the given radius in all directions according to
the shape selected. If an origin other than 0, 0 is desired an x, y offset to
the origin can be provided.

SpanSets are immutable
======================

When working with ``SpanSet`` objects it is important to note that they are
immutable once they are created. As such, all method calls on a ``SpanSet``
object which operate on a ``SpanSet`` create, modify and return a new
``SpanSet`` object. This design is intended to offer a safety guarantee to
programmers. If two different bits of code each have a shared pointer to the
same ``SpanSet`` one can not modify the region and possible corrupt what the
other bit of code is trying to do.

SpanSets do not need to be Spatially contiguous
===============================================

Another important note is that regions defined by a ``SpanSet`` are not
guaranteed to be contiguous. There is however a method which will return if a
given ``SpanSet`` is contiguous, and a method which will split a
non-contiguous ``SpanSet`` into a standard vector (or list in Python) of
contiguous ``SpanSet`` objects. The following example demonstrates this behavior
alongside set operations.

.. code-block:: python

    from lsst.afw.geom import SpanSet, Stencil

    # Construct some SpanSets
    radius = 5
    ss1 = SpanSet.fromShape(radius)
    ss2 = SpanSet.fromShape(radius, offset=(6, 6))

    # Create a union of the immutable sets
    ss3 = ss1.union(ss2)

    # Erode the combine SpanSet with a circular kernel
    newRadius = 4
    ssEroded = ss3.eroded(newRadius, Stencil.CIRCLE)

    # Result is now non-contiguous
    assert(ssEroded.isContiguous() == False)

    # Split the non contiguous SpanSet
    spanSetSplitList = ssEroded.split()
    assert(len(spanSetSplitList) == 2)


SpanSets, Masks, Images, and Ndarray
====================================

Masks
-----

Masks and ``SpanSet``\s can often be used in similar fashion to represent areas
of interest in an image. As such ``SpanSet``\s provide many utility functions to
make transporting this information between these two data types similar. As
Masks are similar to ``SpanSet``\s they can even participate in ``SpanSet``
mathematical operations. Examples for some of the ways masks and ``SpanSet``\s
work together can be found in the example below.

.. code-block:: python

    from lsst.afw.geom import SpanSet
    from lsst.afw.image import MaskU

    # Create a mask to be populated
    size = 10
    mask = MaskU(size, size)

    # Create a SpanSet which represents the pixels to be set in the mask, and
    # set bit two
    radius = 4
    ss = SpanSet.fromShape(radius, offset=(4, 4))
    bitMask = 2
    ss.setMask(mask, bitMask)

    # Intersect not (~) the SpanSet with the mask, the result should be a null
    # SpanSet
    ssIntersectNot = ss.intersectNot(mask)

    # Convert the mask into a SpanSet and verify it evaluates equal to the
    # original
    newSS = SpanSet.fromMask(mask)
    assert(newSS == ss)

Images
------

As mentioned above the ``SpanSet`` class is used to encode sets of x, y
locations on an image. These locations can be used to interact with
additional images through a series of convenience methods demonstrated in the
subsequent example.

.. code-block:: python

    from lsst.afw.geom import SpanSet
    from lsst.afw.image import ImageI

    # Define two different spans sets of differing sized centered at different
    # positions
    radius1 = 3
    radius2 = 2
    spanSet1 = SpanSet.fromShape(radius1, offset=(3, 3))
    spanSet2 = SpanSet.fromShape(radius2, offset=(7, 7))

    # Create two different Images, of the same size
    imageSize = 10
    image1 = ImageI(imageSize, imageSize)
    image2 = ImageI(imageSize, imageSize)

    # Use the SpanSets to set pixels in each image to an arbitrary value
    spanSet1.setImage(image1, 10)
    spanSet2.setImage(image2, 15)

    # Use the second SpanSet to copy the values from image2 into image1 at the
    # positions defined in spanSet2
    spanSet2.copyImage(image2, image1)

    # Show the results
    print(image1.getArray())

    # Output:
    #[[ 0  0  0 10  0  0  0  0  0  0]
    # [ 0 10 10 10 10 10  0  0  0  0]
    # [ 0 10 10 10 10 10  0  0  0  0]
    # [10 10 10 10 10 10 10  0  0  0]
    # [ 0 10 10 10 10 10  0  0  0  0]
    # [ 0 10 10 10 10 10  0 15  0  0]
    # [ 0  0  0 10  0  0 15 15 15  0]
    # [ 0  0  0  0  0 15 15 15 15 15]
    # [ 0  0  0  0  0  0 15 15 15  0]
    # [ 0  0  0  0  0  0  0 15  0  0]]

Ndarrays
--------

A ``SpanSet`` can also be used to extract or insert values from / into ndarrays
while expanding or reducing dimensionality. The ``flatten`` method extracts
data from an array at locations defined by the ``SpanSet`` and returns (or
inserts into) an ``ndarray`` with one less dimension. The ``unflatten`` method
does the opposite. The ``flatten`` method takes the first two dimensions of the
ndarray as the dimensions to flattened, indexed at the locations of the
``SpanSet``. If a ``SpanSet`` is defined to cover a 5x5 area, and is used to
flatten a 5x5x4x10 array, the resulting array will be 25x4x10. Below is small
example.

.. code-block:: python

    import numpy as np
    from lsst.afw.geom import SpanSet, Stencil

    # Create a 2D array with ascending values and a SpanSet of a sub region
    dims = 5
    array = np.arange(dims * dims).reshape(dims, dims)
    radius = 1
    ss = SpanSet.fromShape(radius, Stencil.BOX, offset=(1, 1))

    # Extract the sub region into a flattened array
    flat = ss.flatten(array)

    # Show the flattned values
    print(flat.shape)

    # Output:
    # (9,)

    print(flat)

    # Output:
    # [ 0  1  2  5  6  7 10 11 12]

Using indices (Python only)
===========================

A ``SpanSet`` is a representation of coordinates that is very efficient in
terms of memory usage. This however does not always lend itself to the
Python / Numpy style of programming, owing to the need to do a double loop to
access the actual coordinates. In order to support a more natural way of
programming with Python / Numpy the ``SpanSet`` class provides an ``indices``
method. This method, when called, returns a tuple of two lists. The first list
contains the y coordinate for each point in the ``SpanSet``, and the second
provides the corresponding x coordinates. Note this is different that the x, y
ordering common through other parts of the LSST code base, but was chosen to be
similar to numpy.indices, and the ordering of Numpy arrays. This representation
is less memory efficient and should be used thoughtfully, but enables coding
styles similar to the following example.

.. code-block:: python

    import numpy as np
    from lsst.afw.geom import SpanSet, Stencil

    # Create a numpy array to work with
    arrayDim = 5
    dataArray = np.zeros((arrayDim, arrayDim))

    # Create a SpanSet which indexes all the x, y locations in the data array
    radius = 2
    ss = SpanSet.fromShape(2, Stencil.BOX, offset=(2, 2))

    # Get the indices corresponding to the SpanSet and use it to set values in
    # the data array
    yind, xind = ss.indices()
    dataArray[yind, xind] = 9

    # Show the modified data array
    print(dataArray)

    # Output:
    # [[ 9.  9.  9.  9.  9.]
    #  [ 9.  9.  9.  9.  9.]
    #  [ 9.  9.  9.  9.  9.]
    #  [ 9.  9.  9.  9.  9.]
    #  [ 9.  9.  9.  9.  9.]]


Using applyFunctor (C++ only)
=============================
When a ``SpanSet`` class is used in C++ there is a useful convenience function
that is unavailable from the Python interface called ``applyFunctor``. This
method is meant to simplify the complexities of doing operations at the
locations defined by a ``SpanSet`` on data of mixed types and shapes. The key
point to this functionality is specifying some function-like object that
contains the operation to be done as if each pixel only represented a single
value. The ``applyFunctor`` method then takes this operation, and the data to
be operated on, and iterates over the data types in such a way that the
operator is supplied only one set of values at a time. The method can handle
data types of ``Image``, ``MaskedImage``, ``ndarrays``, numeric values (i.e. a
float), and iterators. Any number of data types may be supplied (constrained by
the number of arguments the supplied function operator takes). The following
contains a snippet of C++ code as a demonstration, a full working example can
be found in the C++ unit test, and more detail on syntax can be found in the
applyFunctor doxygen.

.. code-block:: cpp

    afwImage::Image<int> image(5, 5, 1);
    afwImage::Image<int> outputImage(5, 5, 0);
    std::vector<int> vec(5*5, 2);
    ndarray::Array<int, 2, 2> ndAr = ndarray::allocate(ndarray::makeVector(5,5));
    ndAr.deep() = 1;
    int constant = 2;
    auto ss = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::BOX,
                                          offset=afwGeom::Box2I(2,2))
    // The Point2I argument says where in the SpanSet the operator is being applied
    // but is unused in this example
    ss.applyFunctor([](
        afGeom::Point2I const &, int & out, int const & inIm, int const & inVec,
        int const & ndAr, int number){
            out = inIm * inVec * ndAr / number;
        }, outputImage, image, vec, ndarray::ndImage(ndAr), constant);
