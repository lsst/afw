.. py:currentmodule:: lsst.afw.math

.. _lsst.afw.math-BackgroundExample:

###################################
Example of lsst.afw.math.Background
###################################

Using the `Background` class; the code's in `afw/examples/estimateBackground.py <https://github.com/lsst/afw/blob/master/examples/estimateBackground.py>`_.

The basic strategy is
 - Measure the properties of the image (e.g. the mean level) -- the `Background` object
 - `Interpolate` the `Background` to provide an estimate of the background
 - Or generate an approximation to the `Background`, and use that to estimate the background

Start by importing needed packages

.. code-block:: py

   import os
   import lsst.utils
   import lsst.afw.image as afwImage
   import lsst.afw.math as afwMath

Read an Image

.. code-block:: py

   def getImage():
       imagePath = os.path.join(lsst.utils.getPackageDir("afwdata"),
                                "DC3a-Sim", "sci", "v5-e0", "v5-e0-c011-a00.sci.fits")
       return afwImage.MaskedImageF(imagePath)

.. code-block:: py

   image = getImage()

We'll do the simplest case first.
Start by creating a `BackgroundControl` object that's used to configure the algorithm that's used to estimate the background levels.

.. code-block:: py

   def simpleBackground(image):
       binsize = 128
       nx = int(image.getWidth()/binsize) + 1
       ny = int(image.getHeight()/binsize) + 1
       bctrl = afwMath.BackgroundControl(nx, ny)

Estimate the background levels

.. code-block:: py

   bkgd = afwMath.makeBackground(image, bctrl)

We can ask for the resulting heavily-binned image (but only after casting the base class `Background` to one that includes such an image, a `BackgroundMI`)

.. code-block:: py

or subtract this background estimate from the input image, interpolating our estimated values using a `~Interpolate.NATURAL_SPLINE`

.. code-block:: py

       image -= bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE)

       return bkgd

We actually have a lot more control over the whole process than that.
We'll start by building a `StatisticsControl` object, and telling it our desires:

.. code-block:: py

   sctrl = afwMath.StatisticsControl()
   sctrl.setNumSigmaClip(3)
   sctrl.setNumIter(4)
   sctrl.setAndMask(afwImage.Mask[MaskPixel].getPlaneBitMask(["INTRP",
                                                              "EDGE"]))
   sctrl.setNoGoodPixelsMask(afwImage.Mask[MaskPixel].getPlaneBitMask("BAD"))
   sctrl.setNanSafe(True)

(actually I could have set most of those options in the ctor)

We then build the `BackgroundControl` object, passing it ``sctrl`` and also my desired statistic.

.. code-block:: py

    bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)

Making the `Background` is the same as before

.. code-block:: py

    bkgd = afwMath.makeBackground(image, bctrl)

We can get the statistics image, and its variance:

.. code-block:: py

    statsImage = bkgd.getStatsImage()
    ds9.mtv(statsImage.getVariance())

Finally, we can interpolate in a number of ways, e.g.

.. code-block:: py

If we wish to use an approximation to the background (instead of interpolating the values) we proceed slightly differently.
First we need an object to specify our interpolation strategy:

.. code-block:: py

   order = 2
   actrl = afwMath.ApproximateControl(
       afwMath.ApproximateControl.CHEBYSHEV, order, order)

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
