// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Background

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/utils/Utils.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Background.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;
namespace geom = lsst::afw::geom;

typedef image::Image<float> Image;
typedef image::DecoratedImage<float> DecoratedImage;

BOOST_AUTO_TEST_CASE(BackgroundBasic) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    int nX = 40;
    int nY = 40;
    Image img(geom::Extent2I(nX, nY));
    Image::Pixel const pixVal = 10000;
    img = pixVal;

    {
        int xcen = nX/2;
        int ycen = nY/2;
        math::BackgroundControl bgCtrl("AKIMA_SPLINE");
        // test methods native BackgroundControl
        bgCtrl.setNxSample(5);
        bgCtrl.setNySample(5);
        // test methods for public stats objects in bgCtrl
        bgCtrl.getStatisticsControl()->setNumSigmaClip(3);
        bgCtrl.getStatisticsControl()->setNumIter(3);
        PTR(math::Background) back = math::makeBackground(img, bgCtrl);
        double const TESTVAL = boost::dynamic_pointer_cast<math::BackgroundMI>(back)->getPixel(xcen, ycen);

        image::Image<float>::Ptr bImage = back->getImage<float>();
        Image::Pixel const testFromImage = *(bImage->xy_at(xcen, ycen));

        BOOST_CHECK_EQUAL(TESTVAL, pixVal);
        BOOST_CHECK_EQUAL(TESTVAL, testFromImage);

    }

}

BOOST_AUTO_TEST_CASE(BackgroundTestImages) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    {
        vector<string> imgfiles;
        imgfiles.push_back("v1_i1_g_m400_s20_f.fits");
        imgfiles.push_back("v1_i1_g_m400_s20_u16.fits");
        //imgfiles.push_back("v1_i2_g_m400_s20_f.fits");
        //imgfiles.push_back("v1_i2_g_m400_s20_u16.fits");
        //imgfiles.push_back("v2_i1_p_m9_f.fits");
        //imgfiles.push_back("v2_i1_p_m9_u16.fits");
        //imgfiles.push_back("v2_i2_p_m9_f.fits");
        //imgfiles.push_back("v2_i2_p_m9_u16.fits");

        std::string afwdata_dir;
        try {
            afwdata_dir = lsst::utils::getPackageDir("afwdata");
        } catch (lsst::pex::exceptions::NotFoundError) {
            std::cout << "Warning: test skipped because afwdata is not setup" << std::endl;
            return;
        }
        for (vector<string>::iterator imgfile = imgfiles.begin(); imgfile != imgfiles.end(); ++imgfile) {

            string img_path = afwdata_dir + "/Statistics/" + *imgfile;

            // get the image and header
            DecoratedImage dimg(img_path);
            Image::Ptr img = dimg.getImage();
            lsst::daf::base::PropertySet::Ptr fitsHdr = dimg.getMetadata(); // the FITS header

            // get the true values of the mean and stdev
            float reqMean = static_cast<float>(fitsHdr->getAsDouble("MEANREQ"));
            float reqStdev = static_cast<float>(fitsHdr->getAsDouble("SIGREQ"));

            int const width = img->getWidth();
            int const height = img->getHeight();

            // create a background control object
            math::BackgroundControl bctrl(math::Interpolate::AKIMA_SPLINE);
            bctrl.setNxSample(5);
            bctrl.setNySample(5);
            float stdevSubimg = reqStdev / sqrt(width*height/(bctrl.getNxSample()*bctrl.getNySample()));

            // run the background constructor and call the getPixel() and getImage() functions.
            PTR(math::Background) backobj = math::makeBackground(*img, bctrl);

            // test getPixel()
            float testval = boost::dynamic_pointer_cast<math::BackgroundMI>(backobj)->getPixel(width/2, height/2);
            BOOST_REQUIRE( fabs(testval - reqMean) < 2.0*stdevSubimg );

            // test getImage() by checking the center pixel
            image::Image<float>::Ptr bimg = backobj->getImage<float>();
            float testImgval = static_cast<float>(*(bimg->xy_at(width/2, height/2)));
            BOOST_REQUIRE( fabs(testImgval - reqMean) < 2.0*stdevSubimg );
        }
    }
}


BOOST_AUTO_TEST_CASE(BackgroundRamp) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    {
        // make a ramping image (spline should be exact for linear increasing image
        int const nX = 512;
        int const nY = 512;
        image::Image<float> rampimg = image::Image<float>(geom::Extent2I(nX, nY));
        float dzdx = 0.1;
        float dzdy = 0.2;
        float z0 = 10000.0;

        for (int i = 0; i < nX; ++i) {
            float x = static_cast<float>(i);
            for ( int j = 0; j < nY; ++j) {
                float y = static_cast<float>(j);
                *rampimg.xy_at(i, j) = dzdx*x + dzdy*y + z0;
            }
        }

        // check corner, edge, and center pixels
        math::BackgroundControl bctrl = math::BackgroundControl(math::Interpolate::AKIMA_SPLINE);
        bctrl.setNxSample(6);
        bctrl.setNySample(6);
        bctrl.getStatisticsControl()->setNumSigmaClip(20.0); //something large enough to avoid clipping entirely
        bctrl.getStatisticsControl()->setNumIter(1);
        PTR(math::BackgroundMI) backobj =
            boost::dynamic_pointer_cast<math::BackgroundMI>(math::makeBackground(rampimg, bctrl));

        // test the values at the corners and in the middle
        int ntest = 3;
        for (int i = 0; i < ntest; ++i) {
            int xpix = i*(nX - 1)/(ntest - 1);
            for (int j = 0; j < ntest; ++j) {
                int ypix = j*(nY - 1)/(ntest - 1);
                double testval = backobj->getPixel(xpix, ypix);
                double realval = *rampimg.xy_at(xpix, ypix);
                BOOST_CHECK_CLOSE(testval/realval, 1.0, 2.5e-5);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(BackgroundParabola) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */

    {
        // make an image which varies parabolicly (spline should be exact for 2rd order polynomial)
        int const nX = 512;
        int const nY = 512;
        image::Image<float> parabimg = image::Image<float>(geom::Extent2I(nX, nY));
        float d2zdx2 = -1.0e-4;
        float d2zdy2 = -1.0e-4;
        float dzdx   = 0.1;
        float dzdy   = 0.2;
        float z0 = 10000.0;  // no cross-terms

        for ( int i = 0; i < nX; ++i ) {
            for ( int j = 0; j < nY; ++j ) {
                *parabimg.xy_at(i, j) = d2zdx2*i*i + d2zdy2*j*j + dzdx*i + dzdy*j + z0;
            }
        }

        // check corner, edge, and center pixels
        math::BackgroundControl bctrl = math::BackgroundControl(math::Interpolate::CUBIC_SPLINE);
        bctrl.setNxSample(24);
        bctrl.setNySample(24);
        bctrl.getStatisticsControl()->setNumSigmaClip(10.0);
        bctrl.getStatisticsControl()->setNumIter(1);
        PTR(math::BackgroundMI) backobj =
            boost::dynamic_pointer_cast<math::BackgroundMI>(math::makeBackground(parabimg, bctrl));

        // check the values at the corners and in the middle
        int const ntest = 3;
        for (int i = 0; i < ntest; ++i) {
            int xpix = i*(nX - 1)/(ntest - 1);
            for (int j = 0; j < ntest; ++j) {
                int ypix = j*(nY - 1)/(ntest - 1);
                double testval = backobj->getPixel(xpix, ypix);
                double realval = *parabimg.xy_at(xpix, ypix);
                //print xpix, ypix, testval, realval
                // quadratic terms skew the averages of the subimages and the clipped mean for
                // a subimage != value of center pixel.  1/20 counts on a 10000 count sky
                //  is a fair (if arbitrary) test.
                BOOST_CHECK_CLOSE( testval, realval, 0.05 );
            }
        }
    }
}
