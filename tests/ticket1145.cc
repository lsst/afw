#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Ticket1145

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/MaskedImage.h"

namespace afwImage = lsst::afw::image;

void setImage(afwImage::MaskedImage<float> &image, float im, float var) {
    *image.getImage() = im;
    *image.getMask() = 0x0;
    *image.getVariance() = var;
}

BOOST_AUTO_TEST_CASE(Ticket1145) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    typedef afwImage::MaskedImage<float>::x_iterator XIterator;
    afwImage::MaskedImage<float> image(1, 1);

    XIterator xIter = image.row_begin(0);
    float const imVal = 1.0e10;
    double const divider = 0.5e10;
    
    double const tol = 1e-5;
    {
        setImage(image, imVal, 1.0);
        image /= divider;
        double const v1 = (*xIter).variance();
        std::cout << "using image /= pixel (0,0) is " << *xIter << std::endl;
        
        setImage(image, imVal, 1.0);
        *xIter /= divider;
        double const v2 = (*xIter).variance();
        std::cout << "using *xIter /= pixel (0,0) is " << *xIter << std::endl;
        
        BOOST_CHECK_CLOSE(v1, v2, tol);
    }

    {
        setImage(image, imVal, 1.0e20);
        image /= divider;
        double const v1 = (*xIter).variance();
        std::cout << "using image /= pixel (0,0) is " << *xIter << std::endl;

        setImage(image, imVal, 1.0e20);
        *xIter /= divider;
        double const v2 = (*xIter).variance();
        std::cout << "using *xIter /= pixel (0,0) is " << *xIter << std::endl;
        BOOST_CHECK_CLOSE(v1, v2, tol);
    }
}
