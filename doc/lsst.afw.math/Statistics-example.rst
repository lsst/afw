.. py:currentmodule:: lsst.afw.math

.. _lsst.afw.math-StatisticsExample:

###################################
Example of lsst.afw.math.Statistics
###################################

Demonstrate the use of `Statistics`; the code's in `afw/examples/statistics.cc <https://github.com/lsst/afw/blob/master/examples/statistics.cc>`_.

Start by including needed headers and declaring namespace aliases

.. code-block:: cpp

   #include <iostream>
   #include <cmath>

   #include "lsst/geom.h"
   #include "lsst/afw/image/Image.h"
   #include "lsst/afw/image/MaskedImage.h"
   #include "lsst/afw/math/Statistics.h"

   using namespace std;
   namespace image = lsst::afw::image;
   namespace math = lsst::afw::math;

Create an `lsst.afw.image.Image`

.. code-block:: cpp

   typedef image::Image<float> ImageF;

   int main() {
       // First we'll try a regular image
       ImageF img(lsst::geom::Extent2I(10, 40));
       img = 100000.0;

Create a `Statistics` object from that `~lsst.afw.image.Image`, requesting the number of points, the mean, and the standard deviation.

.. code-block:: cpp

   {
       math::Statistics stats = math::makeStatistics(img, math::NPOINT | math::MEAN | math::STDEV);

And print the desired quantities.
Note that we didn't request that the error in the mean be calculated, so a ``NaN`` is returned.

.. code-block:: cpp

       cout << "Npixel: " << stats.getValue(math::NPOINT) << endl;
       cout << "Mean: " << stats.getValue(math::MEAN) << endl;
       cout << "Error in mean: " << stats.getError(math::MEAN) << " (expect NaN)" << endl;
       cout << "Standard Deviation: " << stats.getValue(math::STDEV) << endl << endl;
   }

Here's another way to do the same thing.
We use `makeStatistics` (cf. :cpp:func:`std::make_pair`) to avoid having to specify what sort of `Statistics` we're creating (and in C++0X you'll be able to say

.. code-block:: cpp

   auto stats = math::makeStatistics(img, math::STDEV | math::MEAN | math::ERRORS);

which means that we *never* have to provide information that the compiler has up its sleeve --- very convenient for generic template programming)

.. code-block:: cpp

   {
       math::Statistics stats = math::makeStatistics(img, math::STDEV | math::MEAN | math::ERRORS);

Print the answers again, but this time return that value and its error as a \c std::pair

.. code-block:: cpp

       std::pair<double, double> mean = stats.getResult(math::MEAN);

       cout << "Mean: " << mean.first << " error in mean: " << mean.second << endl << endl;
   }

Don't ask for things that you didn't request.

.. code-block:: cpp

   {
       math::Statistics stats = math::makeStatistics(img, math::NPOINT);
       try {
           stats.getValue(math::MEAN);
       } catch (lsst::pex::exceptions::InvalidParameterError &e) {
           cout << "You didn't ask for the mean, so we caught an exception: " << e.what() << endl;
       }
   }

be tidy and return success (unnecessary; unlike C, C++ will return this 0 for you automatically)

.. code-block:: cpp

       return 0;
   }
