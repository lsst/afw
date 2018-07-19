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
The single band instances are stored in the `singles` property
of a multiband object and are usually initialized with a
reference to a multiband data cube that contains the values in
each band, making slicing and modification easy.

For example, `MultibandImage` can be created from a list
of single band images using `MultibandImage.fromImages`:

.. code-block:: python

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import ImageF, MultibandImage

    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    filters = ["G", "R", "I", "Z", "Y"]
    images = [ImageF(bbox, n) for n,f in enumerate(filters)]
    mImage = MultibandImage.fromImages(filters, images)

The default constructor uses an array and bounding box to
initialize the image:

.. code-block:: python

    import numpy as np

    from lsst.afw.image import MultibandImage, ImageD
    from lsst.geom import Point2I, Box2I

    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    filters = ["G", "R", "I", "Z", "Y"]
    images = np.random.rand(len(filters), 100, 200).astype(np.float32)
    mImage = MultibandImage(filters, images, bbox)

Indexing and Slicing
--------------------

All objects that inherit from `MultibandBase` can be indexed in the filter
dimension to return a single band object by specifying the filter name,
or sliced in the filter dimension to return a new multiband object.
This is true for all objects that inherit from `MultibandBase`.
For example slicing a `MultibandImage` gives

.. code-block:: python

    import numpy as np

    from lsst.afw.image import MultibandImage

    filters = ["G", "R", "I", "Z", "Y"]
    images = np.random.rand(len(filters), 100, 200).astype(np.float32)
    mImage = MultibandImage(filters, array=images)

    print(mImage["G"])
    print(repr(mImage[:"R"]))

    # Output:
    #<lsst.afw.image.image.image.ImageF object at 0x7fea5950dca8>
    #<MultibandImage, filters=('G',), bbox=Box2I(minimum=Point2I(0, 0), dimensions=Extent2I(200, 100))>

For `MultibandImage` objects and any classes that inherit from it,
the multiband object can also be sliced in the spatial dimensions.

.. warning::
    The LSST stack uses the convention [x,y] for indexing image data,
    which differs from the standard python and C++ [y,x] notation.

The examples in the remainder of this section assume that the following
code has been executed:

.. code-block:: python

    from lsst.geom import Point2I, Box2I, Extent2I
    from lsst.afw.image import ImageF, MultibandImage, LOCAL, PARENT

    bbox = Box2I(Point2I(1000, 2000), Extent2I(200, 100))
    filters = ["G", "R", "I", "Z", "Y"]
    images = [ImageF(bbox, n) for n,f in enumerate(filters)]
    mImage = MultibandImage.fromImages(filters, images)

For example, if we want to extract a small subset in the `R` and `I`
bands in the `MultibandImage` we can use

.. code-block:: python

    subset = mImage["R":"Z", 1000:1005, 2000:2003]
    print(repr(subset))
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

    subset = mImage["R":"Z", :1005, :2003]
    subset = mImage["R":"Z", :-195, :-97, LOCAL]
    subset = mImage["R":"Z", Box2I(Point2I(1000, 2000), Extent2I(5, 3))]

.. warning::
    Negative indices can only be used in a pythonic fashion if `LOCAL`
    is used for the `origin` (as above), which doesn't respect `XY0`.
    Otherwise, the default `origin=PARENT` will throw an `IndexError`,
    since it is possible for `XY0` to be less than 0:

.. code-block:: python

    subset = mImage["R":"Z", :-195, :-97]

    # Output
    # ... traceback here ...
    # IndexError: Negative indices are not permitted with the PARENT origin. Use LOCAL to use negative to index relative to the end, and Point2I or Box2I indexing to access negative pixels in PARENT coordinates.


Conversion to numpy indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`MultibandImage` objects have an `array` property to access the 3D array (filter, y, x)
used to fill the single band objects (in fact the single band `Image` objects are initialized
with pointers to the `Multiband.array`).
In order to have consistent behavior the `imageIndicesToNumpy` method can be used to convert
coordinates in the LSST image frame to the numpy frame:

.. code-block:: python

    ay, ax, bbox = mImage.imageIndicesToNumpy((1001, 2002))
    print(ay, ax, bbox)
    mImage.array[0, ay, ax] = 7
    print(mImage["G", 1001, 2002])

    # Output
    #2 1 None
    #7.0

The inverse can be accomplished using the `origin` property:

.. code-block:: python

    import numpy as np
    iy, ix = np.array(mImage.origin) + np.array([2, 1])
    print(iy, ix)
    mImage.array[0, 2, 1] = 14
    print(mImage["G", ix, iy])

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
    mImage = MultibandImage.fromImages(filters, images)

    subset = mImage[:, mImage.getXY0()]
    print(repr(subset))
    print(subset)

    # Output
    #<MultibandPixel, filters=('G', 'R', 'I', 'Z', 'Y'), bbox=Box2I(minimum=Point2I(1000, 2000), dimensions=Extent2I(1, 1))>
    #[ 0.  1.  2.  3.  4.]

`MultibandPixel` objects can only be sliced in the filter dimension (since there is only one
pixel) and choosing a single band returns an element of the array. For example, using `subset`
as defined above:

    print(subset[:"I"])
    print(subset["G"])
    print(subset[:"R"])

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
    mMask = MultibandMask.fromMasks(filters, singles)

    # Construct a MultibandMask from an array
    masks = np.zeros((3, 100, 200), dtype=np.int32)
    for n in range(len(filters)):
        masks[n] = n
    mMask = MultibandMask(filters, masks, bbox)

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

    print(mMask1[:, -1, -1, LOCAL])
    print(mMask2[:, -1, -1, LOCAL])

    # Output
    #[0 1 2]
    #[1 2 3]

    mMask1 |= mMask2
    print(mMask1[:, -1, -1, LOCAL])

    # Output
    #[1 3 3]

    mMask1 ^= mMask2
    print(mMask1[:, -1, -1, LOCAL])

    # Output
    #[0 1 0]

    mMask1 &= mMask2
    print(mMask1[:, -1, -1, LOCAL])

    # Output
    #[0 0 0]

MultibandMaskedImage
====================

`MultibandMaskedImage` is different from most other multiband classes in that
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
    np.random.seed(1)
    _variance = np.random.rand(3, 100,200).astype(np.float32) * 1e-1
    variance = [Image(_variance[n], xy0=bbox.getMin(), dtype=np.float32) for n in range(len(filters))]

    # Construct a MultibandMaskedImage using single band images
    mMaskedImage = MultibandMaskedImage(filters, images, masks, variance)


    # Construct a MultibandMaskedImage using multiband objects
    mImage = MultibandImage.fromImages(filters, images)
    mMask = MultibandMask.fromMasks(filters, masks)
    mVariance = MultibandImage.fromImages(filters, variance)
    mMaskedImage = MultibandMaskedImage(filters, mImage, mMask, mVariance)

    # Construct a MultibandMaskedImage using arrays
    img = np.array([image.array for image in images])
    msk = np.array([mask.array for mask in masks])
    var = np.array([v.array for v in variance])
    bbox = images[0].getBBox()
    mMaskedImage = MultibandMaskedImage.fromArrays(filters, img, msk, var, bbox)

The remaining sections assume that the above `mMaskedImage` has been initialized.

Indexing and Slicing
--------------------

Like `MultibandImage`, using a single filter index returns a single band version of
the object, in this case a `MaskedImage`, while slicing in the filter dimension returns
a new `MultibandMaskedImage`:

.. code-block:: python

    print(mMaskedImage["G"])
    print(repr(mMaskedImage[:"R"]))

    # Output
    #<lsst.afw.image.maskedImage.maskedImage.MaskedImageF object at 0x7f915adf6148>
    #<MultibandMaskedImage, filters=('G',), bbox=Box2I(minimum=Point2I(1000, 2000), dimensions=Extent2I(200, 100))>

Slices in the image x,y dimensions are performed in all bands, for example:

.. code-block:: python

    subset = mMaskedImage["R":, :1005, :2003]
    print(repr(subset))
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
    # [[[ 0.0805802   0.05415874  0.08788868  0.0947535   0.08327904]
    #  [ 0.02400039  0.01800653  0.08792792  0.09132284  0.0911031 ]
    #  [ 0.04891231  0.0680088   0.04953354  0.0435614   0.04319233]]
    #
    # [[ 0.08787258  0.004194    0.06818753  0.05020679  0.01482407]
    #  [ 0.09170669  0.00243524  0.07492483  0.02702225  0.05552186]
    #  [ 0.0114406   0.04935426  0.00269435  0.03762079  0.01350806]]]

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

    print(mExposure.computePsfImage())

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

`MultibandExposure` also has a `fromButler` method that makes it possible
to load an exposure from a file:

.. code-block:: python

    from lsst.afw.image import MultibandExposure
    from lsst.daf.persistence import Butler

    # This is an example dataset on lsstdev which may be out of date,
    # replace with a local dataset
    DATA_DIR = "/datasets/hsc/repo/rerun/RC/w_2018_22/DM-14547"
    butler = Butler(inputs=DATA_DIR)

    filters = ["G", "R","I"]
    hscFilters = ["HSC-"+f for f in filters]
    mExposure = MultibandExposure.fromButler(butler, hscFilters, None, "deepCoadd_calexp",
                                             patch="1,1", tract=9813)

MultibandFootprint
==================

A `MultibandFootprint` is a collection of `HeavyFootprint` objects, one in each band,
that are required to have the same `SpanSet`s and `PeakCatalog`.

.. warning::

    To speed up processing there is no check that the `PeakCatalog`s are the
    same, so initializing a `MultibandFootprint` with `HeavyFootprint`s that
    have different `PeakCatalog`s may lead to unexpected results.

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
        spans = SpanSet.fromShape(2, Stencil.CIRCLE)
        footprint = Footprint(spans)
        image = ImageF(spans.getBBox())
        image.set(n+1)
        image = MaskedImageF(image)
        heavy = makeHeavyFootprint(footprint, image)
        singles.append(heavy)
    mFoot = MultibandFootprint(filters, singles)
    print(mFoot.getImage(fill=0).image.array)

    # Output
    #[[[ 0.  0.  1.  0.  0.]
    #  [ 0.  1.  1.  1.  0.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 0.  1.  1.  1.  0.]
    #  [ 0.  0.  1.  0.  0.]]
    #
    # [[ 0.  0.  2.  0.  0.]
    #  [ 0.  2.  2.  2.  0.]
    #  [ 2.  2.  2.  2.  2.]
    #  [ 0.  2.  2.  2.  0.]
    #  [ 0.  0.  2.  0.  0.]]
    #
    # [[ 0.  0.  3.  0.  0.]
    #  [ 0.  3.  3.  3.  0.]
    #  [ 3.  3.  3.  3.  3.]
    #  [ 0.  3.  3.  3.  0.]
    #  [ 0.  0.  3.  0.  0.]]]

A `MultibandFootprint` can also be initialized with a list of `Image` objects,
or a `MultibandImage`, and a detection threshold:

.. code-block:: python

    from lsst.afw.detection import Footprint, makeHeavyFootprint, MultibandFootprint
    from lsst.afw.geom import SpanSet, Stencil
    from lsst.afw.image import ImageI

    filters = ["G","R","I"]
    images = []
    for n in range(len(filters)):
        spans = SpanSet.fromShape(2, Stencil.CIRCLE)
        image = ImageI(spans.getBBox())
        spans.setImage(image, 1)

        spans = SpanSet.fromShape(1, Stencil.CIRCLE)
        image2 = ImageI(image.getBBox())
        spans.setImage(image2, n)
        image += image2
        images.append(image)
        print("initial arrays:\n", image.array)
    mFoot = MultibandFootprint.fromImages(filters, images, thresh=1.1)
    print("result:\n", mFoot.getImage(fill=0).image.array)

    # Output
    #initial arrays:
    # [[0 0 1 0 0]
    # [0 1 1 1 0]
    # [1 1 1 1 1]
    # [0 1 1 1 0]
    # [0 0 1 0 0]]
    #initial arrays:
    # [[0 0 1 0 0]
    # [0 1 2 1 0]
    # [1 2 2 2 1]
    # [0 1 2 1 0]
    # [0 0 1 0 0]]
    #initial arrays:
    # [[0 0 1 0 0]
    # [0 1 3 1 0]
    # [1 3 3 3 1]
    # [0 1 3 1 0]
    # [0 0 1 0 0]]
    #result:
    # [[[0 1 0]
    #  [1 1 1]
    #  [0 1 0]]
    #
    # [[0 2 0]
    #  [2 2 2]
    #  [0 2 0]]
    #
    # [[0 3 0]
    #  [3 3 3]
    #  [0 3 0]]]

Notice here that the threshold was set at `thresh=1.1`, which is above the
level of the outer circle and all of the pixels in the `G` band.
So the outer pixels were trimmed from all of the footprints, however
because the same footprint is used for all bands, the values below the
threshold are still used if they fall within the `SpanSet` of the full
`MultibandFootprint`.

Both `fromImages` and `fromArrays` allow the user to specify a
`Footprint` instead of a threshold, and that `Footprint` is used
for all of the bands. For example:

.. code-block:: python

    from lsst.afw.detection import Footprint, MultibandFootprint
    from lsst.afw.geom import SpanSet, Stencil, Extent2I
    import numpy as np
    from lsst.afw.image import ImageF

    filters = ["G","R","I"]

    spans = SpanSet.fromShape(2, Stencil.CIRCLE, offset=(2, 2))
    footprint = Footprint(spans)
    dimensions = spans.getBBox().getDimensions()
    image = np.ones((len(filters), dimensions.getY(), dimensions.getX()), dtype=np.float32)
    image[1] = 2
    image[2] = 3
    print("Input images:\n", image)
    fpImage = ImageF(footprint.getBBox())
    spans.setImage(fpImage, 1)
    print("Footprint:\n", fpImage.array)
    mFoot = MultibandFootprint.fromArrays(filters, image, footprint=footprint)
    print("result:\n", mFoot.getImage(fill=0).image.array)

    # Output
    #Input images:
    # [[[ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.]]
    #
    # [[ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]
    #  [ 2.  2.  2.  2.  2.]]
    #
    # [[ 3.  3.  3.  3.  3.]
    #  [ 3.  3.  3.  3.  3.]
    #  [ 3.  3.  3.  3.  3.]
    #  [ 3.  3.  3.  3.  3.]
    #  [ 3.  3.  3.  3.  3.]]]
    #Footprint:
    # [[ 0.  0.  1.  0.  0.]
    # [ 0.  1.  1.  1.  0.]
    # [ 1.  1.  1.  1.  1.]
    # [ 0.  1.  1.  1.  0.]
    # [ 0.  0.  1.  0.  0.]]
    #result:
    # [[[ 0.  0.  1.  0.  0.]
    #  [ 0.  1.  1.  1.  0.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 0.  1.  1.  1.  0.]
    #  [ 0.  0.  1.  0.  0.]]
    #
    # [[ 0.  0.  2.  0.  0.]
    #  [ 0.  2.  2.  2.  0.]
    #  [ 2.  2.  2.  2.  2.]
    #  [ 0.  2.  2.  2.  0.]
    #  [ 0.  0.  2.  0.  0.]]
    #
    # [[ 0.  0.  3.  0.  0.]
    #  [ 0.  3.  3.  3.  0.]
    #  [ 3.  3.  3.  3.  3.]
    #  [ 0.  3.  3.  3.  0.]
    #  [ 0.  0.  3.  0.  0.]]]

Indexing and Slicing
--------------------

Because a `SpanSet` is more complicated than a 2D array,
it is only possible to slice a `MultibandFootprint` in the
filter dimension, not the spatial dimensions.

Using `mFoot` as defined in the previous section, we see that
filter slicing is identical to the other multiband classes:

.. code-block:: python

    print(mFoot["G"])
    print(mFoot[:"R"])

    # Output
    #<lsst.afw.detection._heavyFootprint.HeavyFootprintF object at 0x7fa35c0e81f0>
    #<MultibandFootprint, filters=('G',), bbox=Box2I(minimum=Point2I(0, 0), dimensions=Extent2I(5, 5))>

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
