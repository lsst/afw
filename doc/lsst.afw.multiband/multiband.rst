#################
Multiband Objects
#################

The majority of the LSST pipeline is based on calibrations and measurements
on single band objects like `Image`, `Mask`, `Exposure`, `HeavyFootprint`, etc.
Once exposures in multiple bands have been co-added it can be convenient to have
multiband classes that act as containers and references to the single band objects,
with special syntactical sugar to simplify working with multiband objects.
For all of the classes described below it is assumed that the single band objects
used to initialize their multiband counterparts are resampled onto a grid that
is the same size, with the same number of pixels (in other words the co-adds are
all projected onto the same reference frame) although they are *NOT* required to
have the same PSF.
All multiband classes inherit from `MultibandBase`, which contains the basic methods for
slicing a multiband object into either a single band object or another multiband object
that is a view into the original.

MultibandImage
==============

To understand how multiband objects we begin with `MultibandImage`.

Construction
------------

All objects that inherit from `MultibandBase` require a list of `filters`,
the name of the filter in each band, as the first argument.
The second argument is always an optional list of `singles`, single band
objects in each band. For example, a `MultibandImage` can be initialized
using:

.. code-block:: python

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import ImageF, MultibandImage

    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    filters = ["G", "R", "I", "Z", "Y"]
    images = [ImageF(bbox, n) for n,f in enumerate(filters)]
    mImage = MultibandImage(filters, images)

In the case of some objects, like `MultibandImage`, multiband objects
can also be created from numpy arrays, for example:

.. code-block:: python

    import numpy as np

    from lsst.afw.image import MultibandImage, ImageD
    from lsst.geom import Point2I, Box2I

    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    filters = ["G", "R", "I", "Z", "Y"]
    images = np.random.rand(len(filters), 100, 200).astype(np.float32)
    mImage = MultibandImage(filters, array=images, bbox=bbox)

Indexing and Slicing
--------------------

All objects that inherit from `MultibandBase` can be sliced in the filter
dimension to return a single band object by either specifying the filter name
or the numerical index.
For example slicing a `MultibandImage` gives

.. code-block:: python

    import numpy as np

    from lsst.afw.image import MultibandImage

    filters = ["G", "R", "I", "Z", "Y"]
    images = np.random.rand(len(filters), 100, 200).astype(np.float32)
    mImage = MultibandImage(filters, array=images)

    print(mImage["G"])
    print(mImage[0])
    print(mImage[:1].__repr__())

    # Output:
    #<lsst.afw.image.image.image.ImageF object at 0x7f69a7a73298>
    #<lsst.afw.image.image.image.ImageF object at 0x7f69a7a73298>
    #<MultibandImage, filters=('G',), bbox=Box2I(minimum=Point2I(0, 0), dimensions=Extent2I(200, 100))>

Notice that the first two slices, `mImage["G"]` and `mImage[0]`,
are identical `ImageF` objects, but if a slice is used instead the result is
a `MultibandImage`. This is true of all objects that inherit from `MultibandBase`
and can be used to return a multiband object when necessary.

For `MultibandImage` objects and any classes that inherit from it,
the multiband object can also be sliced in the spatial dimensions.

.. warning::
    The LSST stack uses the convention [x,y] for indexing image data,
    which differs from the standard python and C++ [y,x] notation.

The examples in the remainder of this section assume that the following
code has been executed:

.. code-block:: python

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import ImageF, MultibandImage

    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    filters = ["G", "R", "I", "Z", "Y"]
    images = [ImageF(bbox, n) for n,f in enumerate(filters)]
    mImage = MultibandImage(filters, images)

For example, if we want to extract a small subset in the `R` and `I`
bands in the `MultibandImage` we can use

.. code-block:: python

    subset = mImage[1:3, 1000:1005, 2000:2003]
    print(subset.__repr__())
    print(subset.array)

    # Output
    #<MultibandImage, filters=('R', 'I'), bbox=Box2I(minimum=Point2I(1000, 2000), dimensions=Extent2I(5, 3))>
    #[[[ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]]

    # [[ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]]]

Notice that the `XY0` (or minimal bounding box coordinate) has been subtracted
from the spatial indices. So the same subset can also be extracted with
all of the following methods:

.. code-block:: python

    subset = mImage[1:3, :1005, :2003]
    subset = mImage[1:3, :-194, :-96]
    subset = mImage[1:3, Box2I(Point2I(1000, 2000), Extent2I(5, 3))]

.. warning::
    Negative indices can only be used in a pythonic fashion if `XY0 >= 0`,
    otherwise negative indices are interpreted to be negative coordinates.

However notice if we use indices less than `XY0` we get an error:

.. code-block:: python

    subset = mImage[1:3, 1:1006, 1:2004]

    # Output
    #---------------------------------------------------------------------------
    #IndexError                                Traceback (most recent call last)
    #<ipython-input-25-9adb2171fc95> in <module>()
    #----> 1 subset = mImage[1:3, 1:1005, 1:2003]
    #
    #~/lsst/code/afw/python/lsst/afw/multiband.py in __getitem__(self, args)
    #    203             return result
    #    204 
    #--> 205         return self._slice(filters=filters, filterIndex=filterIndex, indices=indices[1:])
    #    206 
    #    207     def filterToIndex(self, filterIndex):
    #
    #~/lsst/code/afw/python/lsst/afw/image/multiband.py in _slice(self, filters, filterIndex, indices)
    #    291         if len(indices) > 0:
    #    292             allSlices = [filterIndex, slice(None), slice(None)]
    #--> 293             sy, sx = self.imageIndicesToNumpy(indices)
    #    294             if sy is not None:
    #    295                 allSlices[-2] = sy
    #
    #~/lsst/code/afw/python/lsst/afw/multiband.py in imageIndicesToNumpy(self, indices)
    #    300 
    #    301             if sx is not None:
    #--> 302                 sx = self._removeOffset(sx, x0, bbox.getMaxX())
    #    303             if sy is not None:
    #    304                 sy = self._removeOffset(sy, y0, bbox.getMaxY())
    #
    #~/lsst/code/afw/python/lsst/afw/multiband.py in _removeOffset(self, index, x0, xf)
    #    343                 start = None
    #    344             else:
    #--> 345                 start = _applyBBox(index.start, x0, xf)
    #    346             if index.stop is None:
    #    347                 stop = None
    #
    #~/lsst/code/afw/python/lsst/afw/multiband.py in _applyBBox(index, x0, xf)
    #    329             if index > 0 and (index < x0 or index > xf):
    #    330                 err = "Indices must be <0 or between {0} and {1}, received {2}"
    #--> 331                 raise IndexError(err.format(x0, xf, index))
    #    332             newIndex = index - x0
    #    333             if index < 0:
    #
    #IndexError: Indices must be <0 or between 1000 and 1199, received 1
    #


Conversion to numpy indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`MultibandImage` objects have an `array` property to access the 3D array (filter, y, x)
used to fill the single band objects (in fact the single band `Image` objects are initialized
with pointers to the `Multiband.array`).
Accessing this property may be necessary since images can not be set directly and must be
updated by using the `array` property.
In order to have consistent behavior the `imageIndicesToNumpy` method can be used to convert
coordinates in the LSST image frame to the numpy frame:

.. code-block:: python

    ay, ax = mImage.imageIndicesToNumpy((1001, 2002))
    print(ay, ax)
    mImage.array[0, ay, ax] = 7
    print(mImage[0, 1001, 2002])

    # Output
    #2 1
    #7.0

The inverse can be accomplished using the `origin` property:

.. code-block:: python

    import numpy as np
    iy, ix = mImage.origin + np.array([2, 1])
    print(iy, ix)
    mImage.array[0, 2, 1] = 14
    print(mImage[0, ix, iy])

    # Output
    #2002 1001
    #14.0

MultibandPixel
==============

It is unlikely a user will construct a `MultibandPixel` from scratch.
Instead the `MultibandPixel` is returned when a single pixel is sliced
from a multiband image. For example:

.. code-block:: python

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import ImageF, MultibandImage

    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    filters = ["G", "R", "I", "Z", "Y"]
    images = [ImageF(bbox, n) for n,f in enumerate(filters)]
    mImage = MultibandImage(filters, images)

    subset = mImage[:, mImage.getXY0()]
    print(subset.__repr__())
    print(subset)

    # Output
    #<MultibandPixel, filters=('G', 'R', 'I', 'Z', 'Y'), bbox=Point2I(1000, 2000)>
    #[ 0.  1.  2.  3.  4.]

`MultibandPixel` objects can only be sliced in the filter dimension (since there is only one
pixel) and choosing a single band returns an element of the array. For example, using `subset`
as defined above:

    print(subset[:2])
    print(subset[0])
    print(subset[:1])

    # Output
    #[ 0.  1.]
    #0.0
    #[ 0.]

Another difference between `MultibandPixel` and other multiband classes is that
`MultibandPixel.singles` is just a numpy array with the pixel value in each filter,
not a single band object from the stack.

MultibandMask
=============

`MultibandMask` inherits from `MultibandImage` and has the same behavior
except that it also contains a `maskPlaneDict` that contains information
about the binary values contained in the "image".

Construction
------------

We construct a new `MultibandMask` similar to a `MultibandImage` with
either a single band `Mask` or a 3D data array:

.. code-block:: python

    import numpy as np

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import Mask, MaskPixel, MultibandMask

    filters = ["G", "R", "I"]

    # Construct a MultibandMask from a collection of afw.image.Mask objects
    mask = Mask[MaskPixel]
    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    singles = [mask(bbox) for f in range(len(filters))]
    for n in range(len(singles)):
        singles[n].set(n)
    mMask = MultibandMask(filters, singles)

    # Construct a MultibandMask from an array
    masks = np.zeros((3, 100, 200), dtype=np.int32)
    for n in range(len(filters)):
        masks[n] = n
    mMask = MultibandMask(filters=filters, array=masks, bbox=bbox)

Mask Planes
-----------

In addition to the standard multiband image functionality,
`MultibandMask` also has support for the standard `Mask`
methods and operators, including:

.. code-block:: python

    print(mMask.getMaskPlaneDict())
    print(mMask.getMaskPlane("BAD"))

    # Output
    #{'BAD': 0, 'CR': 3, 'DETECTED': 5, 'DETECTED_NEGATIVE': 6, 'EDGE': 4, 'INTRP': 2, 'NO_DATA': 8, 'SAT': 1, 'SUSPECT': 7}
    #0

Operators
---------

The binary operators used to update `Mask` objects also work for `MultibandMask` objects:

.. code-block:: python

    masks = np.zeros((3, 100, 200), dtype=np.int32)
    for n in range(len(filters)):
        masks[n] = n
    mMask1 = MultibandMask(filters=filters, array=masks, bbox=bbox)

    masks = np.zeros((3, 100, 200), dtype=np.int32)
    for n in range(len(filters)):
        masks[n] = n+1
    mMask2 = MultibandMask(filters=filters, array=masks, bbox=bbox)

    print(mMask1[:, -1, -1])
    print(mMask2[:, -1, -1])

    # Output
    #[0 1 2]
    #[1 2 3]

    mMask1 |= mMask2
    print(mMask1[:, -1, -1])

    # Output
    #[1 3 3]

    mMask1 ^= mMask2
    print(mMask1[:, -1, -1])

    # Output
    #[0 1 0]

    mMask1 &= mMask2
    print(mMask1[:, -1, -1])

    # Output
    #[0 0 0]

MultibandMaskedImage
====================

`MultibandMaskedImage` is different from the other multiband classes in that
it does not have an `array` property, since it is actually a collection of
three arrays: `image` is a `MultibandImage`, `mask` is a `MultibandMask`,
and `variance` is a `MultibandImage` that describes the variance of the
pixels in `image`.

Construction
------------

A new `MultibandMaskedImage` can be constructed in the following ways:

.. code-block:: python

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import MultibandMask, MultibandImage, MultibandMaskedImage
    from lsst.afw.image import Mask, Image

    # Setup the image, mask, and variance
    filters = ["G", "R", "I"]
    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    images = [Image(bbox, n, dtype=np.float32) for n in range(len(filters))]
    masks = [Mask(bbox) for f in filters]
    for n, mask in enumerate(masks):
        mask.set(2**n)
    _variance = np.random.rand(3, 100,200).astype(np.float32) * 1e-1
    variance = [Image(_variance[n], xy0=bbox.getMin(), dtype=np.float32) for n in range(len(filters))]

    # Construct a MultibandMaskedImage using single band images
    mMaskedImage = MultibandMaskedImage(filters, image=images, mask=masks, variance=variance)

    # Construct a MultibandMaskedImage using multiband objects
    mImage = MultibandImage(filters, singles=images)
    mMask = MultibandMask(filters, singles=masks)
    mVariance = MultibandImage(filters, singles=variance)
    mMaskedImage = MultibandMaskedImage(filters, image=mImage, mask=mMask, variance=mVariance)

The remaining sections assume that the above `mMaskedImage` has been initialized.

Indexing and Slicing
--------------------

Like `MultibandImage`, using a single filter index returns a single band version of
the object, in this case a `MaskedImage`, while slicing in the filter dimension returns
a new `MultibandMaskedImage`:

.. code-block:: python

    print(mImage["G"])
    print(mImage[0])
    print(mImage[:1].__repr__())

    # Output
    #<lsst.afw.image.image.image.ImageF object at 0x7fa9fcf247a0>
    #<lsst.afw.image.image.image.ImageF object at 0x7fa9fcf247a0>
    #<MultibandImage, filters=('G',), bbox=Box2I(minimum=Point2I(1000, 2000), dimensions=Extent2I(200, 100))>

Slices in the image x,y dimensions are performed in all bands, for example:

.. code-block:: python

    subset = mMaskedImage[1:3, :1005, :2003]
    print(subset.__repr__())
    print("image:\n", subset.image.array)
    print("mask:\n", subset.mask.array)
    print("variance:\n", subset.variance.array)

    # Output
    #<MultibandMaskedImage, filters=('R', 'I'), bbox=Box2I(minimum=Point2I(1000, 2000), dimensions=Extent2I(5, 3))>
    #image:
    # [[[ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]]
    #
    # [[ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]]]
    #mask:
    # [[[2 2 2 2 2]
    #  [2 2 2 2 2]
    #  [2 2 2 2 2]]
    #
    # [[4 4 4 4 4]
    #  [4 4 4 4 4]
    #  [4 4 4 4 4]]]
    #variance:
    # [[[ 0.0029152   0.00433466  0.00061036  0.00972021  0.00367536]
    #   [ 0.00997514  0.00166059  0.00602718  0.00395029  0.00816098]
    #   [ 0.00350882  0.00489013  0.00632092  0.00879703  0.000716  ]]
    #
    # [[ 0.00228322  0.00872436  0.00210415  0.00585763  0.0099331 ]
    #  [ 0.00727455  0.00358093  0.0075652   0.00454849  0.00338826]
    #  [ 0.00599099  0.0098431   0.00771584  0.00207854  0.00840227]]]

MultibandExposure
=================

Like `MultibandMaskedImage`, `MultibandExposure` has a multiband
image, mask, and variance image, and can be indexed and sliced in the same manner.
In addition to the properties and methods of a `MultibandMaskedImage`,
a `MultibandExposure` also has a PSF in each band.
Typically these are stored in the single band `Exposure` objects, but the
entire PSF image can be built using the `getPsfImage` method:

.. code-block:: python

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import MultibandExposure
    from lsst.afw.image import Mask, Image
    from lsst.afw.detection import GaussianPsf

    filters = ["G", "R", "I"]
    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    images = [Image(bbox, n, dtype=np.float32) for n in range(len(filters))]
    masks = [Mask(bbox) for f in filters]
    for n, mask in enumerate(masks):
        mask.set(2**n)
    _variance = np.random.rand(3, 100,200).astype(np.float32) * 1e-1
    variance = [Image(_variance[n], xy0=bbox.getMin(), dtype=np.float32) for n in range(len(filters))]

    kernelSize = 5
    psfs = [GaussianPsf(kernelSize, kernelSize, 4.0) for f in filters]

    mExposure = MultibandExposure(filters, image=images, mask=masks, variance=variance, psfs=psfs)

    print(mExposure.getPsfImage())
    
    # Output
    #[[[ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.0398913   0.04381203  0.04520277  0.04381203  0.0398913 ]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]]
    #
    # [[ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.0398913   0.04381203  0.04520277  0.04381203  0.0398913 ]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]]
    #
    # [[ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.0398913   0.04381203  0.04520277  0.04381203  0.0398913 ]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]]]

It is also possible to set the PSF in a single band:

    gPsf = GaussianPsf(kernelSize, kernelSize, 1.0)
    rPsf = GaussianPsf(kernelSize, kernelSize, 2.0)
    mExposure.setPsf(gPsf, "G")
    mExposure.setPsf(rPsf, 1)
    print(mExposure.getPsfImage())
    
    # Output
    #[[[ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.02193823  0.09832033  0.16210282  0.09832033  0.02193823]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]]
    #
    # [[ 0.02324684  0.03382395  0.03832756  0.03382395  0.02324684]
    #  [ 0.03382395  0.04921356  0.05576627  0.04921356  0.03382395]
    #  [ 0.03832756  0.05576627  0.06319146  0.05576627  0.03832756]
    #  [ 0.03382395  0.04921356  0.05576627  0.04921356  0.03382395]
    #  [ 0.02324684  0.03382395  0.03832756  0.03382395  0.02324684]]
    #
    # [[ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.0398913   0.04381203  0.04520277  0.04381203  0.0398913 ]
    #  [ 0.03866398  0.04246407  0.04381203  0.04246407  0.03866398]
    #  [ 0.03520395  0.03866398  0.0398913   0.03866398  0.03520395]]]

or set all of the PSFs together:

.. code-block:: python

    psfs = [GaussianPsf(kernelSize, kernelSize, n/2) for n in range(len(filters))]
    mExposure.setAllPsfs(psfs)
    print(mExposure.getPsfImage())

    # Output
    #[[[ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.02193823  0.09832033  0.16210282  0.09832033  0.02193823]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]]
    #
    # [[ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.02193823  0.09832033  0.16210282  0.09832033  0.02193823]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]]
    #
    # [[ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.02193823  0.09832033  0.16210282  0.09832033  0.02193823]
    #  [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
    #  [ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]]]

`MultibandExposure` also has a `fromButler` method that makes it possible
to load an exposure from a file:

.. code-block:: python

    from lsst.afw.image import MultibandExposure
    from lsst.daf.persistence import Butler

    # This is an example dataset on lsstdev which may be out of date,
    # replace with a local dataset
    DATA_DIR = "/datasets/hsc/repo/rerun/RC/w_2018_22/DM-14547"
    butler = Butler(inputs=DATA_DIR)

    filters = ["G","R","I"]
    hscFilters = ["HSC-"+f for f in filters]
    mExposure = MultibandExposure.fromButler(butler, hscFilters, None, "deepCoadd_calexp",
                                             patch="1,1", tract=9813)

MultibandFootprint
==================

A `MultibandFootprint` is a collection of `HeavyFootprint` objects, one in each band,
that do not necessarily need to have the same (or even overlapping) `SpanSet`s.
If `SpanSet` is not the same for all of the single band `HeavyFootprint` objects,
the `MultibandFootprint` will have a `SpanSet` that is the union of all of the
single band `SpanSet`s.

Construction
------------

If a `HeavyFootprint` already exists for each band, a `MultibandFootprint` can
be initialized using the list of `HeavyFootprint` objects as `singles`:

.. code-block:: python

    from lsst.afw.detection import Footprint, makeHeavyFootprint, MultibandFootprint
    from lsst.afw.geom import SpanSet, Stencil
    from lsst.afw.image import ImageF, MaskedImageF

    singles = []
    for n in range(len(filters)):
        spans = SpanSet.fromShape(1, Stencil.CIRCLE, offset=(2*(n+1),2*(n+1)))
        footprint = Footprint(spans)
        image = ImageF(spans.getBBox())
        image.set(n+1)
        image = MaskedImageF(image)
        heavy = makeHeavyFootprint(footprint, image)
        singles.append(heavy)
    mFoot = MultibandFootprint(filters, singles)
    print(mFoot.getArray())

    # Output
    #[[[ 0.  1.  0.  0.  0.  0.  0.]
    #  [ 1.  1.  1.  0.  0.  0.  0.]
    #  [ 0.  1.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]]
    #
    # [[ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  2.  0.  0.  0.]
    #  [ 0.  0.  2.  2.  2.  0.  0.]
    #  [ 0.  0.  0.  2.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]]
    #
    # [[ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  3.  0.]
    #  [ 0.  0.  0.  0.  3.  3.  3.]
    #  [ 0.  0.  0.  0.  0.  3.  0.]]]

A `MultibandFootprint` can also be initialized with a list of `Image` objects,
or a `MultibandImage`, and a detection threshold:

.. code-block:: python

    from lsst.afw.detection import Footprint, makeHeavyFootprint, MultibandFootprint
    from lsst.afw.geom import SpanSet, Stencil
    from lsst.afw.image import ImageF, MaskedImageF

    filters = ["G","R","I"]
    images = []
    for n in range(len(filters)):
        spans = SpanSet.fromShape(1, Stencil.CIRCLE, offset=(2*(n+1),2*(n+1)))
        image = ImageF(spans.getBBox())
        spans.setImage(image, n+1)
        image.array[1,1] = 4
        images.append(image)
        print("initial arrays:\n", image.array)
    mFoot = MultibandFootprint(filters, images=images, thresh=1.1)
    print("result:\n", mFoot.getArray())

    # Output
    #initial arrays:
    # [[ 0.  1.  0.]
    # [ 1.  4.  1.]
    # [ 0.  1.  0.]]
    #initial arrays:
    # [[ 0.  2.  0.]
    # [ 2.  4.  2.]
    # [ 0.  2.  0.]]
    #initial arrays:
    # [[ 0.  3.  0.]
    # [ 3.  4.  3.]
    # [ 0.  3.  0.]]
    #result:
    # [[[ 4.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]]
    #
    # [[ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  2.  0.  0.  0.]
    #  [ 0.  2.  4.  2.  0.  0.]
    #  [ 0.  0.  2.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]]
    #
    # [[ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  3.  0.]
    #  [ 0.  0.  0.  3.  4.  3.]
    #  [ 0.  0.  0.  0.  3.  0.]]]

Indexing and Slicing
--------------------

Because a `SpanSet` is more complicated than a 2D array,
it is only possible to slice a `MultibandFootprint` in the
filter dimension, not the spatial dimensions.

Using `mFoot` as defined in the previous section, we see that
filter slicing is identical to the other multiband classes:

.. code-block:: python

    print(mFoot["G"])
    print(mFoot[0])
    print(mFoot[:1])

    # Output
    #<lsst.afw.detection._heavyFootprint.HeavyFootprintF object at 0x7f6690b40260>
    #<lsst.afw.detection._heavyFootprint.HeavyFootprintF object at 0x7f6690b40260>
    #<MultibandFootprint, filters=('G',), bbox=Box2I(minimum=Point2I(2, 2), dimensions=Extent2I(6, 6))>

Peak Catalog
------------

`MultibandFootprint` does have a `peaks` property to load the
`peakCatalog`, which is defined to be the same in all single
band `HeavyFootprint` objects:

.. code-block:: python

    import numpy as np

    from lsst.afw.detection import Footprint, makeHeavyFootprint, MultibandFootprint
    from lsst.afw.geom import SpanSet, Stencil
    from lsst.afw.image import ImageF, MaskedImageF

    filters = ["G","R","I"]
    images = []
    spanSet = SpanSet()
    for n in range(len(filters)):
        spans = SpanSet.fromShape(1, Stencil.CIRCLE, offset=(2*(n+1),2*(n+1)))
        spanSet = spanSet.union(spans)
        image = ImageF(spans.getBBox())
        spans.setImage(image, n+1)
        images.append(image)
    footprint = Footprint(spanSet)
    footprint.addPeak(1, 1, 1)
    footprint.addPeak(3, 3, 2)
    footprint.addPeak(5, 5, 3)
    mFoot = MultibandFootprint(filters, images=images, footprint=footprint)

    print(mFoot.peaks)
    for n, single in enumerate(mFoot.singles):
        msg = "HeavyFootprint {0} has the same peak as the MultibandFootprint: {1}"
        print(msg.format(n, np.all([single.getPeaks()[key]==mFoot.peaks[key]
                                    for key in ["id", "f_x", "f_y"]])))

    # Output
    # id f_x f_y i_x i_y peakValue
    #    pix pix pix pix     ct   
    #--- --- --- --- --- ---------
    # 16 1.0 1.0   1   1       1.0
    # 17 3.0 3.0   3   3       2.0
    # 18 5.0 5.0   5   5       3.0
    #HeavyFootprint 0 has the same peak as the MultibandFootprint: True
    #HeavyFootprint 1 has the same peak as the MultibandFootprint: True
    #HeavyFootprint 2 has the same peak as the MultibandFootprint: True
