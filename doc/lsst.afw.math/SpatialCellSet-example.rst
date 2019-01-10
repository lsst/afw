.. py:currentmodule:: lsst.afw.math

.. _lsst.afw.math-SpatialCellSetExample:

#######################################
Example of lsst.afw.math.SpatialCellSet
#######################################

Demonstrate the use of `SpatialCellSet`\ s; the code's in `spatialCellExample.cc`_.

.. _spatialCellExample.cc: https://github.com/lsst/afw/blob/master/examples/spatialCellExample.cc
.. _spatialCellExample.py: https://github.com/lsst/afw/blob/master/examples/spatialCellExample.py
.. _testSpatialCell.h: https://github.com/lsst/afw/blob/master/examples/testSpatialCell.h
.. _testSpatialCell.cc: https://github.com/lsst/afw/blob/master/examples/testSpatialCell.cc

Start by including needed headers, and declaring namespace aliases and a routine ``readImage``

.. code-block:: cpp

   #include <string>
   #include "lsst/geom.h"
   #include "lsst/utils/Utils.h"
   #include "lsst/pex/exceptions.h"
   #include "lsst/daf/base.h"
   #include "lsst/afw/detection.h"
   #include "lsst/afw/image.h"
   #include "lsst/afw/math.h"

   #include "testSpatialCell.h"

   namespace afwDetect = lsst::afw::detection;
   namespace afwImage = lsst::afw::image;
   namespace afwMath = lsst::afw::math;
   typedef float PixelT;

   std::pair<std::shared_ptr<afwImage::MaskedImage<PixelT>>, std::shared_ptr<afwDetect::FootprintSet>>
   readImage();

We start by calling ``readImage``, and use ``boost::tie`` to unpack the ``std::pair``.
The ``tie`` call does what you think, unpacking a ``pair`` into a couple of variables (it works for ``boost::tuple`` too, and is in TR1's ``<tuple>`` header).

.. code-block:: cpp

   void SpatialCellSetDemo() {
       std::shared_ptr<afwImage::MaskedImage<PixelT>> im;
       std::shared_ptr<afwDetect::FootprintSet> fs;
       boost::tie(im, fs) = readImage();

We want to learn something about the objects in this image, and would like to ensure that the ones we study are spread reasonably uniformly.
We accordingly create a `SpatialCellSet`; a collection of `SpatialCell`\ s, each of which will maintain its one list of candidate objects for study.
For example, if we were estimating the PSF we'd want a set of isolated bright stars, but we wouldn't want them to all come from the top right corner of the image.
A `SpatialCellSet` allows us to take the best ``n`` candidates from each `SpatialCell`, ensuring a reasonable distribution across the image.

The constructor's first argument is the image's bounding box --- it'd be nice to simply pass the image, wouldn't it, but that's not currently supported.
The second and third arguments ``260, 200`` define the size (in pixels) of the `SpatialCell`\ s.

If you run the python version of this example, `spatialCellExample.py`_, with ``display = True`` the 6 cells will be shown in green (why 6?  Because the image is 512x512 and you can fit 260x200 into 512x512 6 times.)

.. code-block:: cpp
   :name: dummy-to-force-indent1

       /*
        * Create an (empty) SpatialCellSet
        */
       afwMath::SpatialCellSet cellSet(im->getBBox(), 260, 200);

Our `SpatialCellSet` is empty, so let's insert all the objects in the frame into it.
We have a list of detections in the `FootprintSet` ``fs``, so this is easy.
We package each object into an ``ExampleCandidate``, and insert it into the set.
The `SpatialCellSet` is responsible for putting it into the correct cell, and `SpatialCell` for maintaining an order within each cell; this ordering is defined by a virtual function ``double ExampleCandidate::getCandidateRating() const``.
The ``ExampleCandidate`` class is implemented in `testSpatialCell.h`_ and `testSpatialCell.cc`_

You can store anything you like in your candidate class, the only requirement is that it inherit from `SpatialCellCandidate` or `SpatialCellImageCandidate` (the latter adds some extra virtual methods).
I chose to save a pointer to the parent image, and the object's bounding box.

.. code-block:: cpp
   :name: dummy-to-force-indent2

       /*
        * Populate the cellSet using the detected object in the FootprintSet
        */
       for (afwDetect::FootprintSet::FootprintList::iterator ptr = fs->getFootprints()->begin(),
                                                             end = fs->getFootprints()->end();
            ptr != end; ++ptr) {
           lsst::geom::Box2I const bbox = (*ptr)->getBBox();
           float const xc = (bbox.getMinX() + bbox.getMaxX()) / 2.0;
           float const yc = (bbox.getMinY() + bbox.getMaxY()) / 2.0;
           std::shared_ptr<ExampleCandidate> tc(new ExampleCandidate(xc, yc, im, bbox));
           cellSet.insertCandidate(tc);
       }

It's possible to iterate over all the objects in a `SpatialCellSet` (we'll do so in a moment), but the simplest
way to visit all cells is to pass in a visitor object.
The ``ExampleCandidateVisitor`` object (defined in `testSpatialCell.h`_) counts the candidates and the number of pixels contained in their bounding boxes.

.. code-block:: cpp
   :name: dummy-to-force-indent3

       ExampleCandidateVisitor visitor;

       cellSet.visitCandidates(&visitor);
       std::cout << boost::format("There are %d candidates\n") % visitor.getN();

Now we'll visit each of our objects by explicit iteration.
The iterator returns a base-class pointer so we need a ``dynamic_cast`` (this cast is also available from python via a little swiggery).
We decided that we don't like small objects, defined as those with less than 75 pixels in their bounding boxes, so we'll label
them as `~SpatialCellCandidate.BAD`.

.. code-block:: cpp
   :name: dummy-to-force-indent4

       for (unsigned int i = 0; i != cellSet.getCellList().size(); ++i) {
           std::shared_ptr<afwMath::SpatialCell> cell = cellSet.getCellList()[i];

           for (afwMath::SpatialCell::iterator candidate = cell->begin(), candidateEnd = cell->end();
                candidate != candidateEnd; ++candidate) {
               lsst::geom::Box2I box = dynamic_cast<ExampleCandidate *>((*candidate).get())->getBBox();
               if (box.getArea() < 75) {
                   (*candidate)->setStatus(afwMath::SpatialCellCandidate::BAD);
               }
           }
       }


What does `~SpatialCellCandidate.BAD` mean (other options are `~SpatialCellCandidate.UNKNOWN` and `~SpatialCellCandidate.GOOD`)?
Basically that that object is to be ignored.
It no longer appears in the size of the `SpatialCell`\ s, it is skipped by the iterators, and the visitors pass it by.
You can turn this behaviour off with `~SpatialCellSet.setIgnoreBad`.

Note that we pass the visitor *before* we decide to ignore `~SpatialCellCandidate.BAD` so ``getN()`` and ``getNPix()`` return the number of good objects/pixels.

.. code-block:: cpp
   :name: dummy-to-force-indent5

       for (unsigned int i = 0; i != cellSet.getCellList().size(); ++i) {
           std::shared_ptr<afwMath::SpatialCell> cell = cellSet.getCellList()[i];
           cell->visitCandidates(&visitor);

           cell->setIgnoreBad(false);  // include BAD in cell.size()
           std::cout << boost::format("%s nobj=%d N_good=%d NPix_good=%d\n") % cell->getLabel() % cell->size() %
                                visitor.getN() % visitor.getNPix();
       }

And count the good candidate again

.. code-block:: cpp
   :name: dummy-to-force-indent6

       cellSet.setIgnoreBad(true);  // don't visit BAD candidates
       cellSet.visitCandidates(&visitor);
       std::cout << boost::format("There are %d good candidates\n") % visitor.getN();
   }

Running the example should print

.. code-block:: none

   There are 22 candidates
   Cell 0x0 nobj=2 N_good=2 NPix_good=1858
   Cell 1x0 nobj=2 N_good=1 NPix_good=210
   Cell 0x1 nobj=4 N_good=4 NPix_good=1305
   Cell 1x1 nobj=4 N_good=1 NPix_good=360
   Cell 0x2 nobj=3 N_good=1 NPix_good=99
   Cell 1x2 nobj=7 N_good=2 NPix_good=288
   There are 11 good candidates

----------

Here's the function that reads a FITS file and finds a set of object in it.
It isn't really anything to do with `SpatialCell`\ s, but for completeness...

.. code-block:: cpp

   std::pair<std::shared_ptr<afwImage::MaskedImage<PixelT>>, std::shared_ptr<afwDetect::FootprintSet>>
   readImage() {

First read a part of the FITS file.
We use `lsst.utils.getPackageDir` to find the directory, and only read a part of the image (that's the ``BBox``).
The use of a ``boost::shared_ptr<MaskedImage>`` (written as ``MaskedImage::Ptr``) is because I want to call the actual constructor in the scope of the try block, but I want to use the image at function scope.

.. code-block:: cpp
   :name: dummy-to-force-indent7

       std::shared_ptr<afwImage::MaskedImage<PixelT>> mi;

       try {
           std::string dataDir = lsst::utils::getPackageDir("afwdata");

           std::string filename = dataDir + "/CFHT/D4/cal-53535-i-797722_1.fits";

           lsst::geom::Box2I bbox =
                   lsst::geom::Box2I(lsst::geom::Point2I(270, 2530), lsst::geom::Extent2I(512, 512));

           std::shared_ptr<lsst::daf::base::PropertySet> md;
           mi.reset(new afwImage::MaskedImage<PixelT>(filename, md, bbox));

       } catch (lsst::pex::exceptions::NotFoundError &e) {
           std::cerr << e << std::endl;
           exit(1);
       }

Subtract the background;  the ``try`` block is in case the image is too small for a spline fit.

.. code-block:: cpp
   :name: dummy-to-force-indent8

       /*
        * Subtract the background.  We can't fix those pesky cosmic rays, as that's in a dependent product
        * (meas/algorithms)
        */
       afwMath::BackgroundControl bctrl(afwMath::Interpolate::NATURAL_SPLINE);
       bctrl.setNxSample(mi->getWidth() / 256 + 1);
       bctrl.setNySample(mi->getHeight() / 256 + 1);
       bctrl.getStatisticsControl()->setNumSigmaClip(3.0);
       bctrl.getStatisticsControl()->setNumIter(2);

       std::shared_ptr<afwImage::Image<PixelT>> im = mi->getImage();
       try {
           *mi->getImage() -= *afwMath::makeBackground(*im, bctrl)->getImage<PixelT>();
       } catch (std::exception &) {
           bctrl.setInterpStyle(afwMath::Interpolate::CONSTANT);
           *mi->getImage() -= *afwMath::makeBackground(*im, bctrl)->getImage<PixelT>();
       }

Run an object detector

.. code-block:: cpp
   :name: dummy-to-force-indent9

       /*
        * Find sources
        */
       afwDetect::Threshold threshold(5, afwDetect::Threshold::STDEV);
       int npixMin = 5;  // we didn't smooth
       std::shared_ptr<afwDetect::FootprintSet> fs(
               new afwDetect::FootprintSet(*mi, threshold, "DETECTED", npixMin));
       int const grow = 1;
       bool const isotropic = false;
       std::shared_ptr<afwDetect::FootprintSet> grownFs(new afwDetect::FootprintSet(*fs, grow, isotropic));
       grownFs->setMask(mi->getMask(), "DETECTED");

And return the desired data

.. code-block:: cpp

       return std::make_pair(mi, grownFs);
   }
