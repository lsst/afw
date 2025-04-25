.. py:currentmodule:: lsst.afw.math

.. _lsst.afw.math-BackgroundExample:

###################################
Example of lsst.afw.math.Background
###################################

The basic strategy is
 - Measure the properties of the image (e.g. the mean level) -- the `Background` object
 - `Interpolate` the `Background` to provide an estimate of the background
 - Or generate an approximation to the `Background`, and use that to estimate the background

Start by importing needed packages

.. code-block:: py

    import os
    import lsst.afw.image as afwImage
    import lsst.afw.math as afwMath
    import lsst.utils

Read a test Image from the DC3a-Sim data set.

.. code-block:: py

    def get_image():
        imagePath = os.path.join(
            lsst.utils.getPackageDir("afwdata"),
            "DC3a-Sim",
            "sci",
            "v5-e0",
            "v5-e0-c011-a00.sci.fits",
        )
        return afwImage.MaskedImageF(imagePath)

.. code-block:: py

    image = get_image()

We'll do the simplest case first.
Start by creating a `BackgroundControl` object that's used to configure the algorithm that's used to estimate the background levels.

.. code-block:: py

    def make_background_control(image, binsize=128, **kwargs):
        nx = int(image.getWidth() / binsize) + 1
        ny = int(image.getHeight() / binsize) + 1
        bctrl = afwMath.BackgroundControl(nx, ny, **kwargs)
        return bctrl

.. code-block:: py

    bctrl = make_background_control(image)

Estimate the background that corresponds to our image.

.. code-block:: py

    bkgd = afwMath.makeBackground(image, bctrl)

We can ask for the resulting heavily-binned image (the *statistics image*):

.. code-block:: py

    stats_image = bkgd.getStatsImage()

or subtract this background estimate from the input image, interpolating our estimated values using a `~Interpolate.NATURAL_SPLINE`:

.. code-block:: py

    image_bgsub = image.clone()
    image_bgsub -= bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE)

We actually have a lot more control over the whole process than that.
We'll start by building a `StatisticsControl` object, and telling it our desires:

.. code-block:: py

    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(3)
    sctrl.setNumIter(4)
    sctrl.setAndMask(image.mask.getPlaneBitMask(["INTRP", "EDGE"]))
    sctrl.setNoGoodPixelsMask(image.mask.getPlaneBitMask("BAD"))
    sctrl.setNanSafe(True)

(actually we could have set most of those options in the constructor, but this is clearer)

We then build the `BackgroundControl` object, passing it ``sctrl`` and also my desired statistic.

.. code-block:: py

    bctrl = make_background_control(image, sctrl=sctrl, prop=afwMath.MEANCLIP)

Making the `Background` is the same as before

.. code-block:: py

    bkgd = afwMath.makeBackground(image, bctrl)

We can get the statistics image, and its variance:

.. code-block:: py

    stats_image = bkgd.getStatsImage()
    stats_image_variance = statsImage.getVariance()

Finally, we can interpolate in a number of ways, e.g.

.. code-block:: py

    AKIMA_SPLINE
    AKIMA_SPLINE_PERIODIC
    CONSTANT
    CUBIC_SPLINE
    CUBIC_SPLINE_PERIODIC
    LINEAR
    NATURAL_SPLINE

If we wish to use an approximation to the background (instead of interpolating the values) we proceed slightly differently.
First we need an object to specify our interpolation strategy:

.. code-block:: py

    order = 2
    actrl = afwMath.ApproximateControl(
        afwMath.ApproximateControl.CHEBYSHEV,
        order,
        order,
    )

and then we can `Approximate <ApproximateF>` the `Background` (in this case with a Chebyshev polynomial)

.. code-block:: py

    approx = bkgd.getApproximate(actrl)

We can get an `~lsst.afw.image.Image` or `~lsst.afw.image.MaskedImage` from ``approx`` with

.. code-block:: py

    approx.getImage()
    approx.getMaskedImage()

or truncate the expansion (as is often a good idea with a Chebyshev expansion); in this case to order one lower than the original fit.

.. code-block:: py

    approx.getImage(order - 1)
