#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Background

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Background.h"

using namespace std;
namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;
typedef image::DecoratedImage<float> DecoratedImageT;

BOOST_AUTO_TEST_CASE(BackgroundBasic) {

    int nx = 40;
    int ny = 40;
    ImageT img(nx, ny);
    ImageT::Pixel const pixval = 10000;
    img = pixval;

    {
        int xcen = nx/2;
        int ycen = ny/2;
        math::BackgroundControl bgCtrl(math::NATURAL_SPLINE);
        // test methods native BackgroundControl
        bgCtrl.setNxSample(3);
        bgCtrl.setNySample(3);
        // test methods for public stats objects in bgCtrl
        bgCtrl.sctrl.setNumSigmaClip(3);
        bgCtrl.sctrl.setNumIter(3);
        math::Background back = math::make_Background(img, bgCtrl);
        double const testval = back.getPixel(xcen, ycen);
        
        image::Image<float>::Ptr bImage = back.getImage<float>();
        ImageT::Pixel const testFromImage = *(bImage->xy_at(xcen, ycen));
        
        BOOST_CHECK_EQUAL(testval, pixval);
        BOOST_CHECK_EQUAL(testval, testFromImage);

    }

}

BOOST_AUTO_TEST_CASE(BackgroundTestImages) {

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
        
        string afwdata_dir = getenv("AFWDATA_DIR");
        for (vector<string>::iterator imgfile = imgfiles.begin(); imgfile != imgfiles.end(); ++imgfile) {
            
            string img_path = afwdata_dir + "/Statistics/" + *imgfile;

            // get the image and header
            DecoratedImageT dimg(img_path);
            ImageT::Ptr img = dimg.getImage();
            lsst::daf::base::PropertySet::Ptr fitsHdr = dimg.getMetadata(); // the FITS header

            // get the true values of the mean and stdev
            float const req_mean = static_cast<float>(fitsHdr->getAsDouble("MEANREQ"));
            float const req_stdev = static_cast<float>(fitsHdr->getAsDouble("SIGREQ"));

            int const naxis1 = img->getWidth();
            int const naxis2 = img->getHeight();
            
            // create a background control object
            math::BackgroundControl bctrl(math::NATURAL_SPLINE);
            bctrl.setNxSample(3);
            bctrl.setNySample(3);
            float stdev_subimg = req_stdev / sqrt(naxis1*naxis2/(bctrl.getNxSample()*bctrl.getNySample()));

            // run the background constructor and call the getPixel() and getImage() functions.
            math::Background backobj = math::make_Background(*img, bctrl);

            // test getPixel()
            float const testval = static_cast<float>(backobj.getPixel(naxis1/2,naxis2/2));
            BOOST_REQUIRE( fabs(testval - req_mean) < 2.0*stdev_subimg );

            // test getImage() by checking the center pixel
            image::Image<float>::Ptr bimg = backobj.getImage<float>();
            float const testImgval = static_cast<float>(*(bimg->xy_at(naxis1/2,naxis2/2)));
            BOOST_REQUIRE( fabs(testImgval - req_mean) < 2.0*stdev_subimg );
            
        }
    }
        
}



BOOST_AUTO_TEST_CASE(BackgroundRamp) {

    {
        
        // make a ramping image (spline should be exact for linear increasing image
	int const nx = 512;
	int const ny = 512;
        image::Image<double> rampimg = image::Image<double>(nx,ny);
	double dzdx = 0.1;
        double dzdy = 0.2;
        double z0 = 10000.0;

	for (int i = 0; i < nx; ++i) {
            double x = static_cast<double>(i);
	    for ( int j = 0; j < ny; ++j) {
                double y = static_cast<double>(j);
		*rampimg.xy_at(i, j) = dzdx*x + dzdy*y + z0;
            }
        }
	
	// check corner, edge, and center pixels
        math::BackgroundControl bctrl = math::BackgroundControl(math::NATURAL_SPLINE);
	bctrl.setNxSample(6);
	bctrl.setNySample(6);
	bctrl.sctrl.setNumSigmaClip(20.0);  // something large enough to avoid clipping entirely
	bctrl.sctrl.setNumIter(1);
        math::Background backobj = math::Background(rampimg,bctrl);

        // test the values at the corners and in the middle
        int const ntest = 3;
	for (int i = 0; i < ntest; ++i) {
            int xpix = i*(nx - 1)/(ntest - 1);
	    for(int j = 0; j < ntest; ++j) {
                int ypix = j*(ny - 1)/(ntest - 1);
		double testval = backobj.getPixel(xpix, ypix);
                double realval = *rampimg.xy_at(xpix, ypix);
		BOOST_CHECK_CLOSE( testval, realval, 1.0e-10 );
            }
        }
                    
    }

}
BOOST_AUTO_TEST_CASE(BackgroundParabola) {

    {
        
	// make an image which varies parabolicly (spline should be exact for 2rd order polynomial)
	int const nx = 512;
	int const ny = 512;
        image::Image<double> parabimg = image::Image<double>(nx,ny);
	double d2zdx2 = -1.0e-4;
        double d2zdy2 = -1.0e-4;
        double dzdx   = 0.1;
        double dzdy   = 0.2;
        double z0 = 10000.0;  // no cross-terms

	for ( int i = 0; i < nx; ++i ) {
	    for ( int j = 0; j < ny; ++j ) {
		*parabimg.xy_at(i, j) = d2zdx2*i*i + d2zdy2*j*j + dzdx*i + dzdy*j + z0;
            }
        }
	
	// check corner, edge, and center pixels
        math::BackgroundControl bctrl = math::BackgroundControl(math::NATURAL_SPLINE);
	bctrl.setNxSample(16);
	bctrl.setNySample(16);
	bctrl.sctrl.setNumSigmaClip(10.0);
	bctrl.sctrl.setNumIter(1);
        math::Background backobj = math::Background(parabimg,bctrl);

	// debug
	//bimg = backobj.getImageD()
	//ds9.mtv(parabimg)
	//ds9.mtv(bimg, frame=1)
	//parabimg.writeFits("a.fits")
	//bimg.writeFits("b.fits")

        // check the values at the corners and int he middle
        int const ntest = 3;
	for (int i = 0; i < ntest; ++i) {
            int xpix = i*(nx - 1)/(ntest - 1);
	    for(int j = 0; j < ntest; ++j) {
                int ypix = j*(ny - 1)/(ntest - 1);
		double testval = backobj.getPixel(xpix, ypix);
		double realval = *parabimg.xy_at(xpix,ypix);
		//print xpix, ypix, testval, realval
		// quadratic terms skew the averages of the subimages and the clipped mean for
		// a subimage != value of center pixel.  1/20 counts on a 10000 count sky
		//  is a fair (if arbitrary) test.
		BOOST_CHECK_CLOSE( testval, realval, 0.05 );
            }
        }
    }
}
