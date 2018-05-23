
###############################################
Image Indexing, Array Views, and Bounding Boxes
###############################################

Pixel Indexing Conventions
==========================

.. currentmodule:: lsst.afw.image

LSST's image classes (`Image`, `Mask`, `MaskedImage`, and `Exposure`) use a pixel indexing convention that is different from both the convention used by `numpy.ndarray` objects and the convention used in FITS images.

.. currentmodule:: lsst.afw.geom

Like FITS but unlike NumPy, points and pixel indices in LSST software are always ordered ``(x, y)`` (with x the column index and y the row index); this includes both geometry objects (`Point2D`, `Point2I`, `Extent2D`, `Extent2I`) and the images classes themselves.

Like NumPy and unlike FITS, LSST typically labels the center of the lower left pixel of an image ``(0, 0)``.
Note that because we label the centers of pixels with integer coordinates, the exact coordinate bounding box of an image (which correspond to the edges of pixels) have half-integer values.

.. currentmodule:: lsst.afw.image

LSST image classes also have the ability to use a custom origin, which we frequently refer to as ``xy0``.
The image coordinate system that uses ``xy0`` as the coordinates of the lower left pixel is called the `PARENT` coordinate system, because when a subimage is created, it allows the subimage to use the same coordinate system as that of the "parent" image it is derived from.
Most image operations can also utilize the `LOCAL` system, which uses
``(0,0)`` as the coordinates of the lower left pixel regardless of the value of ``xy0``.

.. warning::

    NumPy array indices are ordered `(y, x)` and always start at `(0, 0)`, while LSST image indices are ordered `(x, y)` and sometimes have a nonzero origin.

In Python, all image operations use `PARENT` coordinates by default.
In C++, most image operations use `PARENT`, including all those with direct equivalents in Python; only iterator and ``operator()-based`` direct pixel access do not (purely for historical/backwards-compatibility reasons).
The coordinate system to use is typically indicated via the `afw.image.ImageOrigin` enum type, which has two values, `PARENT` and `LOCAL`.

To illustrate how this works, let's start with a 10×12 image with ``xy0=(0,0)``:

>>> import numpy as np
>>> from lsst.afw.image import Image
>>> from lsst.afw.geom import Box2I, Point2I, Extent2I
>>> img = Image(Extent2I(x=10, y=12), dtype=np.float32)
>>> print(img.getBBox(LOCAL))
(minimum=(0, 0), maximum=(9, 11))
>>> print(img.getBBox(PARENT))
(minimum=(0, 0), maximum=(9, 11))

Because ``xy0=(0,0)``, the bounding box is the same in both coordinate systems.
We'll now extract a subimage:

>>> box1 = Box2I(minimum=Point2I(x=2, y=3), maximum=Point2I(x=7, y=9))
>>> sub1 = img[box1]

This image has a nonzero ``xy0``:

>>> print(sub1.getXY0())
(2, 3)

This makes sense; the lower left pixel in the subimage corresponds to pixel
``(2, 3)`` in the original image.

As expected, the bounding box of the subimage is the same as the one we used to construct it:

>>> print(sub1.getBBox())
(minimum=(2, 3), maximum=(7, 9))

This is the `PARENT` bounding box; the `LOCAL` bounding box has the same
dimensions but a different offset:

>>> print(sub1.getBBox(LOCAL))
Box2I(minimum=Point2I(0, 0), dimensions=Extent2I(6, 7))

.. note::

    The `PARENT` bounding box's minimum point is ``xy0``, while the `PARENT` bounding box's minimum point is ``(0, 0)``; this is *always* true.

The operation that creates a subimage can also accept an `ImageOrigin` argument:

>>> sub1a = img[box1, LOCAL]

This argument indicates which of the *original* image's coordinate systems the given box is in.
But in this case, the original image has ``xy0 = (0, 0)``, and hence those two coordinate systems are the same, so `sub1a` is exactly the same subimage as `sub1`.

That's not the case if we make a subimage of our subimage:

>>> box2 = Box2I(minimum=Point2I(x=3, y=4), maximum=Point2I(x=5, y=5))
>>> sub2a = sub1[box2, PARENT]   # same as no ImageOrigin argument
>>> sub2b = sub1[box2, LOCAL]
>>> sub2a.getBBox(PARENT)
(minimum=(3, 4), maximum=(5, 5))
>>> sub2b.getBBox(PARENT)
(minimum=(5, 7), maximum=(7, 8))
>>> sub2a.getBBox(LOCAL)
(minimum=(0, 0), maximum=(2, 1))
>>> sub2b.getBBox(LOCAL)
(minimum=(0, 0), maximum=(2, 1))

As in the previous case, when we make a subimage using a box in `PARENT`coordinates, the PARENT bounding box of the result is that same box.
When we make a subimage using a box in `LOCAL` coordinates, that input box is different from both the resulting subimage's `LOCAL` bounding box and its `PARENT` bounding box.

.. note::

    We strongly recommend using the `PARENT` convention whenever possible (which usually just means not explicitly selecting `LOCAL`, of course, since `PARENT` is the default).


FITS Reading and Writing
========================

The flexibility of our ``xy0`` functionality makes it possible to make LSST images use the FITS convention by setting ``xy0 = (1, 1)``, but LSST code does not do this, even when reading and writing FITS images.
Instead, we read a FITS image with an origin of ``(1, 1)`` into LSST image objects with an origin of ``(0, 0)`` (and do the reverse when writing, of course).

We also adjust any FITS WCS in the image headers to account for this change in conventions, and *also* write an extra (trivial, shift-only) WCS that offsets the pixel grid by ``xy0``, providing FITS access to our `PARENT` coordinate system.


Floating-Point and Integer Bounding Boxes
=========================================

.. currentmodule:: lsst.afw.geom

One consequence of using integer labels for pixel centers is that integer boxes (`Box2I`) behave fundamentally differently from floating-point bounding boxes (`Box2D`).

The width and height of an image's integer bounding box are of course the same as those of the image itself:

>>> img = Image(Extent2I(x=10, y=12), dtype=np.float32)
>>> boxI = img.getBBox()
>>> print(boxI.getDimensions())
(10, 12)

But this is not the same as the difference between the minimum and maximum points of that box:

>>> print(boxI.getMax() - boxI.getMin())
(9, 11)

That's because those values correspond to the *centers* of the minimum and maximum pixels, and hence this naive subtraction does not include that half-pixel-width boundary.

This same discrepancy can also be seen when converting a `Box2I` to a `Box2D`:

>>> from lsst.afw.geom import Box2D, Point2D, Extent2D
>>> boxD = Box2D(boxI)
>>> print(boxD)
(minimum=(-0.5, -0.5), maximum=(9.5, 11.5))
>>> print(boxD.getDimensions())
(10, 12)

When converting a `Box2I` to a `Box2D`:

 - the dimensions are preserved;
 - the minimum and maximum points are not (instead, they are expanded by half a pixel in all directions).

This means that the difference between the minimum and maximum points of a `Box2D` *is* equivalent to its size:

>>> print(boxD.getMax() - boxD.getMin())
(10, 12)

The conversion from a `Box2D` to a `Box2I` is not as straightforward, because there are many `Box2D` regions that cannot be represented exactly by `Box2I` objects.
Instead, the `Box2I.EdgeHandlingEnum` is used to specify whether the `Box2I` is the smallest integer box that contains the `Box2D` (`Box2I.EXPAND`), or the `Box2I` is the largest integer box that is contained by the `Box2D` (`Box2I.SHRINK`).

In fact, because of the half-pixel boundary discrepancy noted above, `Box2D` objects with integer-valued minimum and maximum points are among those that cannot be converted exactly to `Box2Is <Box2I>`, even though it *looks* like they are (when `Box2I.EXPAND` is used):

>>> smallBox = Box2D(Point2D(0.0, 0.0), Point2D(10.0, 12.0))
>>> expandedBox = Box2I(smallBox, Box2I.EXPAND)
>>> print(smallBox)
(minimum=(0, 0), maximum=(10, 12))
>>> print(expandedbox)
(minimum=(0, 0), maximum=(10, 12))

While ``smallBox`` and ``expandedBox`` appear to have the same minimum and maximum points, they actually represent different regions: `smallBox` does not enclose that half-pixel boundary around the edges, and this is reflected by their dimensions:

>>> print(smallBox.getDimensions())
(10, 12)
>>> print(expandedBox.getDimensions())
(11, 13)

Converting with `Box2I.SHRINK` of course creates a box that is smaller than the `Box2D`:

>>> shrunkBox = Box2I(smallBox, Box2I.SHRINK)
>>> print(shrunkBox)
(minimum=(1, 1), maximum=(9, 11))
>>> print(shrunkBox.getDimensions())
(9, 11)

.. currentmodule:: lsst.afw.image


Image Slicing
=============

We've already shown how `Box2I` objects can be used to access subimage views.
This is usually the most concise syntax, and we recommend using it when a box object is available.

It's also possible, however, to create subimages using Python's built-in slice syntax; the ``sub1`` and ``sub2`` views below are thus equivalent:

>>> box = Box2I(minimum=Point2I(x=2, y=3), maximum=Point2I(x=7, y=9))
>>> sub1 = img[box]
>>> sub2 = img[2:8, 3:9]

Note that again that `Box2I` maximum points are inclusive, while slice upper endpoints are exclusive.
The indices still follow the LSST conventions: the slices are ordered ``(x, y)`` and assumed to be `PARENT` coordinates unless `LOCAL` is explicitly passed as the last argument.

It is also possible to use scalar indices or `Point2I` objects when indexing `Images <Image>` and `Masks <Mask>`:

>>> scalar = img[3, 4]
>>> scalar = img[Point2I(x=3, y=4)]

Indexing with a slice for one dimension and a scalar for the other is not supported, because LSST image objects are intrinsically 2-d.
1-d array views can be obtained by first accessing the 2-d array view and slicing that (see :ref:`array_views_to_images`).

.. note::
    Python slicing typically allows negative indices to be used to indicate positions relative to the end of the sequence.
    This is supported when slicing LSST image objects when either the `LOCAL` coordinate system is used or ``xy0`` is nonnegative.
    When ``xy0`` is negative, negative indices in `PARENT` coordinates could be either positions relative to the end or true negative pixel indices, and to avoid confusion image classes will raise `IndexError` instead of assuming either.
    To obtain a subimage containing a region that includes negative-index pixels, use a `Box2I`.

.. _array_views_to_images:

Array Views to Images
=====================

The `Image` and `Mask` classes also provide NumPy views to their internal data via an ``array`` property.
These are writeable views that can be used to modify the contents of the `Image` or `Mask`.

Because these are just `numpy.ndarray` objects, these views conform to NumPy's conventions, not LSST's: indices are ordered ``(y, x)``, and are ignorant of ``xy0``.
That means the following two array views are equivalent:

>>> view1 = img[x1:x2, y1:y2, LOCAL].array
>>> view2 = img.array[y1:y2, x1:x2]
