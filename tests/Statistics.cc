#include <iostream>
#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Statistics

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

using namespace std;

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef image::Image<float> ImageT;
typedef image::DecoratedImage<float> DecoratedImageT;

BOOST_AUTO_TEST_CASE(StatisticsBasic) {
    ImageT img(10,40);
    ImageT::Pixel const pixval = 10000;
    img = pixval;

    {
        math::Statistics stats = math::make_Statistics(img, math::NPOINT | math::STDEV | math::MEAN);
        double const mean = stats.getValue(math::MEAN);
        double const dmean = stats.getError(math::MEAN);
        double const sd = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(stats.getValue(math::NPOINT), img.getWidth()*img.getHeight());
        BOOST_CHECK_EQUAL(mean, img(0,0));
        BOOST_CHECK(std::isnan(dmean)); // we didn't ask for the error, so it's a NaN
        BOOST_CHECK_EQUAL(sd, 0);
    }

    {
        math::Statistics stats = math::make_Statistics(img, math::STDEV | math::MEAN | math::ERRORS);
        std::pair<double, double> const mean = stats.getResult(math::MEAN);
        double const sd = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(mean.first,  img(0,0));
        BOOST_CHECK_EQUAL(mean.second, sd/sqrt(img.getWidth()*img.getHeight()));
    }

    {
        math::Statistics stats = math::make_Statistics(img, math::NPOINT);
        BOOST_CHECK_THROW(stats.getValue(math::MEAN), lsst::pex::exceptions::InvalidParameter);
    }

    // ===============================================================================
    // sjb code for percentiles and clipped stats
    {
        math::Statistics stats = math::make_Statistics(img, math::MEDIAN);
        BOOST_CHECK_EQUAL(pixval, stats.getValue(math::MEDIAN));
    }
    {
        math::Statistics stats = math::make_Statistics(img, math::IQRANGE);
        BOOST_CHECK_EQUAL(0.0, stats.getValue(math::IQRANGE));
    }
    {
        math::Statistics stats = math::make_Statistics(img, math::MEANCLIP);
        BOOST_CHECK_EQUAL(pixval, stats.getValue(math::MEANCLIP));
    }
    {
        math::Statistics stats = math::make_Statistics(img, math::VARIANCECLIP);
        BOOST_CHECK_EQUAL(0.0, stats.getValue(math::VARIANCECLIP));
    }

    
    
    {
        ImageT img2(img);
        //
        // Add 1 to every other row, so the variance is 1/4
        //
        BOOST_REQUIRE(img.getHeight()%2 == 0);
        for (int y = 1; y < img.getHeight(); y += 2) {
            for (ImageT::x_iterator ptr = img.row_begin(y), end = ptr + img.getWidth(); ptr != end; ++ptr) {
                *ptr += 1;
            }
        }

        math::Statistics stats = math::make_Statistics(img2,
                                                               math::NPOINT | math::STDEV | math::MEAN | math::ERRORS);
        std::pair<double, double> const mean = stats.getResult(math::MEAN);
        double const n = stats.getValue(math::NPOINT);
        double const sd = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(mean.first,  img(0,0) + 0.5);
        BOOST_CHECK_EQUAL(sd, 1/sqrt(4.0)*sqrt(n/(n - 1)));
        BOOST_CHECK_CLOSE(mean.second, sd/sqrt(img.getWidth()*img.getHeight()), 1e-10);
    }

}




BOOST_AUTO_TEST_CASE(StatisticsRamp) {

    int nx = 101;
    int ny = 64;
    ImageT img(nx,ny);
    
    double z0 = 10.0;
    double dzdx = 1.0;
    double mean = z0 + (nx/2)*dzdx;
    double stdev = 0.0;
    for (int i_y=0; i_y < ny; ++i_y) {
        double x = 0;
        for (ImageT::x_iterator ptr=img.row_begin(i_y); ptr != img.row_end(i_y); ++ptr) {
            *ptr = z0 + dzdx*x;
            x += 1.0;
            stdev += (*ptr - mean)*(*ptr - mean);
        }
    }
    stdev = sqrt(stdev/(nx*ny-1));
    
    {
        math::Statistics stats = math::make_Statistics(img, math::NPOINT | math::STDEV | math::MEAN);
        double const testmean = stats.getValue(math::MEAN);
        double const teststdev = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(stats.getValue(math::NPOINT), nx*ny);
        BOOST_CHECK_EQUAL(testmean, mean);
        BOOST_CHECK_EQUAL(teststdev, stdev );
    }

    {
        math::Statistics stats = math::make_Statistics(img, math::STDEV | math::MEAN | math::ERRORS);
        std::pair<double, double> const mean = stats.getResult(math::MEAN);
        double const sd = stats.getValue(math::STDEV);
        
        BOOST_CHECK_EQUAL(mean.first,  img(nx/2,ny/2));
        BOOST_CHECK_EQUAL(mean.second, sd/sqrt(img.getWidth()*img.getHeight()));
    }

    // ===============================================================================
    // sjb code for percentiles and clipped stats
    {
        math::Statistics stats = math::make_Statistics(img, math::MEDIAN);
        BOOST_CHECK_EQUAL(z0+dzdx*(nx-1)/2.0, stats.getValue(math::MEDIAN));
    }
    {
        math::Statistics stats = math::make_Statistics(img, math::IQRANGE);
        BOOST_CHECK_EQUAL(dzdx*(nx-1)/2.0, stats.getValue(math::IQRANGE));
    }
    {
        math::Statistics stats = math::make_Statistics(img, math::MEANCLIP);
        BOOST_CHECK_EQUAL(z0+dzdx*(nx-1)/2.0, stats.getValue(math::MEANCLIP));
    }
    //{
    //    math::Statistics stats = math::make_Statistics(img, math::VARIANCECLIP);
    //    BOOST_CHECK_EQUAL(0.0, stats.getValue(math::VARIANCECLIP));
    //}

    
    
}


BOOST_AUTO_TEST_CASE(StatisticsTestImages) {
    
    /* =============================================================================
     * Tests of mean and standard deviation for Russ Laher's noise images.
     * - only one for now (time consuming)
     */
    {
        vector<string> imgfiles;
        imgfiles.push_back("v1_i1_g_m400_s20_f.fits");
        imgfiles.push_back("v1_i1_g_m400_s20_u16.fits");
        imgfiles.push_back("v1_i2_g_m400_s20_f.fits");
        imgfiles.push_back("v1_i2_g_m400_s20_u16.fits");
        imgfiles.push_back("v2_i1_p_m9_f.fits");
        imgfiles.push_back("v2_i1_p_m9_u16.fits");
        imgfiles.push_back("v2_i2_p_m9_f.fits");
        imgfiles.push_back("v2_i2_p_m9_u16.fits");

        string afwdata_dir = getenv("AFWDATA_DIR");
        for (vector<string>::iterator imgfile = imgfiles.begin(); imgfile != imgfiles.end(); ++imgfile) {
            
            string img_path = afwdata_dir + "/Statistics/" + *imgfile;

            // get the image and header
            DecoratedImageT dimg(img_path);
            lsst::daf::base::DataProperty::PtrType fitsHdr = dimg.getMetadata(); // the FITS header

            // get the true values of the mean and stdev
            double const true_mean = boost::any_cast<double>(fitsHdr->findUnique("MEANCOMP", true)->getValue());
            double const true_stdev = boost::any_cast<double>(fitsHdr->findUnique("SIGCOMP", true)->getValue());

            // measure the mean and stdev with the Statistics class
            ImageT::Ptr img = dimg.getImage();
            math::Statistics statobj = math::make_Statistics(*img,math::MEAN | math::STDEV);
            //int n = img->getWidth() * img->getHeight();
            //double sampleToPop = 1.0; //sqrt( n/static_cast<double>(n - 1) );
            double const mean  = statobj.getValue(math::MEAN);
            double const stdev = statobj.getValue(math::STDEV);
            

            BOOST_CHECK_CLOSE(mean, true_mean, 1e-8);
            BOOST_CHECK_CLOSE(stdev, true_stdev, 1e-8);
        }
    }
}
