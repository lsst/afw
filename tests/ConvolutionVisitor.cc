// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include <vector>
#include <exception>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ConvolutionVisitor

#include "boost/make_shared.hpp"
#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/FourierCutout.h"
#include "lsst/afw/math/ConvolutionVisitor.h"

#include "lsst/pex/exceptions/Runtime.h"

namespace math = lsst::afw::math;

typedef math::FourierCutout FourierCutout;
typedef math::FourierCutoutStack FourierCutoutStack;
typedef math::ConvolutionVisitor ConvolutionVisitor;
typedef math::ImageConvolutionVisitor ImageConvolutionVisitor;
typedef ImageConvolutionVisitor::Image Image;
typedef ImageConvolutionVisitor::ImagePtrList ImagePtrList;
typedef math::FourierConvolutionVisitor FourierConvolutionVisitor;
typedef std::vector<FourierCutout::Ptr> FourierCutoutVector;
typedef FourierCutout::Complex Complex;




//the following test arrays where constructed using numpy
//the fft was computed using numpy.fft.rfft2

int const STACK_DEPTH = 3;
//stack of 3 image, each with width = 5, height = 4
int const IMG_HEIGHT = 4;
int const IMG_WIDTH = 5;
FourierCutout::Real IMG_STACK[] = {
    //img 0
    0.33513162,  0.34152215,  0.84575211,  0.25487178,  0.84134722,
    0.31479221,  0.47327061,  0.36582024,  0.70971434,  0.43028471,
    0.23375864,  0.40928687,  0.78796993,  0.08044837,  0.14428223,
    0.33723794,  0.74822495,  0.88635509,  0.74912522,  0.5357934,

    //img 1
    0.09665001,  0.20036448,  0.0930482 ,  0.00527651,  0.15051378,
    0.3089627 ,  0.31239953,  0.46431586,  0.48297487,  0.1780558,
    0.39905868,  0.85173424,  0.34705442,  0.23215277,  0.35078164,
    0.78489037,  0.53779432,  0.64608295,  0.18282591,  0.917828,

    //img 3
    0.75029482,  0.10442891,  0.9591722 ,  0.51431895,  0.25177158,
    0.45397202,  0.75372244,  0.82175259,  0.85670104,  0.5519374,
    0.82939743,  0.38143594,  0.24266148,  0.80346663,  0.1956888,
    0.17255558,  0.20760615,  0.43392076,  0.80113605,  0.19277729
};

//stack of 3 fourier images, each with width = 3, height = 4 
//not that fourier width == image width/2 +1, and fourier height = image height
int const FOURIER_WIDTH = 3;
Complex FOURIER_STACK[] = {
    //fourier img 0
    Complex( 9.82498961, +0.00000000e+00), Complex(-1.35273888, -6.61296240e-01),
    Complex(-0.50745493, +1.02619760e+00),
    Complex( 0.96287885, +9.62854493e-01), Complex( 0.55192916, +4.82965925e-01),
    Complex(-0.69396459, +2.26300637e-01),
    Complex(-1.27624780, +0.00000000e+00), Complex( 0.37772021, -4.18411731e-01),
    Complex( 0.05255398, +1.71955753e+00),
    Complex( 0.96287885, -9.62854493e-01), Complex(-0.33597078, +1.10894659e+00),
    Complex( 0.02199227, +4.50945815e-01),

    //fourier img 1
    Complex( 7.54276503, +0.00000000e+00), Complex( 0.68584754, -6.70636606e-01),
    Complex(-0.48332566, +4.36250535e-01),
    Complex(-1.63492878, +1.32271279e+00), Complex(-0.38247592, +1.31492361e+00),
    Complex(-0.52269309, -1.02141911e-01),
    Complex(-2.08949558, +0.00000000e+00), Complex( 0.16935056, -6.15310334e-01),
    Complex(-0.61996373, -6.98251036e-01),
    Complex(-1.63492878, -1.32271279e+00), Complex( 0.02940176, -4.24983611e-01),
    Complex( 0.99865262, +5.80839726e-01),

    //fourier img 2
    Complex(10.27871806, +0.00000000e+00), Complex(-1.37366470, +6.20038488e-02),
    Complex(1.74985529, -6.42653287e-01),
    Complex( 0.12733617, -1.63008966e+00), Complex(-0.86622337, -4.76770617e-01),
    Complex(0.43780577, +1.46617365e+00),
    Complex(-0.21344458, +0.00000000e+00), Complex( 1.03304629, +1.25614065e-03),
    Complex(1.45658760, +3.76952530e-01),
    Complex( 0.12733617, +1.63008966e+00), Complex(-0.12000849, -7.18772862e-02),
    Complex(0.02557684, +8.38272853e-01)
};

BOOST_AUTO_TEST_CASE(ImageConvolutionTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    Image::Ptr imgA = boost::make_shared<Image>(19,19, 1.0); 
    Image::Ptr imgB = boost::make_shared<Image>(19,19, 0.0);
    std::pair<int, int> center(9, 9);

    ImagePtrList derivativeA;       
    ImagePtrList derivativeB(3);
    for(int i = 0; i < 3; ++i) {    
        derivativeB[i] = boost::make_shared<Image>(19, 19, i);
    }

    ImageConvolutionVisitor a(center, std::vector<double>(), imgA);
    ImageConvolutionVisitor b(center, std::vector<double>(3, 0), imgB, derivativeB);
    
    Image::Ptr imgFromA = a.getImage(), imgFromB = b.getImage();

    BOOST_CHECK_EQUAL(imgFromA, imgA);
    BOOST_CHECK_EQUAL(imgFromB, imgB);

    ImagePtrList listFromA = a.getDerivativeImageList(), listFromB = b.getDerivativeImageList();

    BOOST_CHECK_EQUAL_COLLECTIONS(listFromA.begin(), listFromA.end(), derivativeA.begin(), derivativeA.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(listFromB.begin(), listFromB.end(), derivativeB.begin(), derivativeB.end());

    BOOST_CHECK_EQUAL(a.getNParameters(), derivativeA.size());
    BOOST_CHECK_EQUAL(b.getNParameters(), derivativeB.size());
    
    BOOST_CHECK(a.getCovariance().get() == 0);
    BOOST_CHECK(b.getCovariance().get() == 0);

    ConvolutionVisitor::CovariancePtr covarB =
        boost::make_shared<Eigen::MatrixXd>(b.getNParameters(), b.getNParameters() - 1);

    BOOST_CHECK_THROW(a.setCovariance(covarB), lsst::pex::exceptions::InvalidParameterException);
    BOOST_CHECK_NO_THROW(b.setCovariance(covarB));

    derivativeB.clear();

}

BOOST_AUTO_TEST_CASE(FourierConvolutionTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int width = IMG_WIDTH;
    int height = IMG_HEIGHT;
    int fourierWidth = FOURIER_WIDTH;

    std::pair<int, int> center(IMG_WIDTH/2, IMG_HEIGHT/2);

    FourierCutout::Real * testRealItr = IMG_STACK;
    Image::Ptr img = boost::make_shared<Image>(width, height); 

    for(int y = 0; y < height; ++y) {
        Image::x_iterator pixel = img->row_begin(y);
        Image::x_iterator end(img->row_end(y));
        for( ; pixel != end; ++pixel, ++testRealItr) {
            *pixel = *testRealItr;    
        }
    }
    
    ImagePtrList derivative(2);
    unsigned int nDerivatives = derivative.size();
    for(int i = 0; i < nDerivatives; ++i) {    
        derivative[i] = boost::make_shared<Image>(width, height);
        
        for(int y = 0; y < height; ++y) {
            Image::x_iterator pixel = derivative[i]->row_begin(y);
            Image::x_iterator end(derivative[i]->row_end(y));
            for( ; pixel != end; ++pixel, ++testRealItr) {
                *pixel = *testRealItr;    
            }
        }
    }
    std::vector<double> parameterList(nDerivatives, 1);
    ImageConvolutionVisitor imgVisitor(center, parameterList, img, derivative);
    FourierConvolutionVisitor fourierVisitor(imgVisitor);
 
    BOOST_CHECK_EQUAL(imgVisitor.getNParameters(), nDerivatives);
    BOOST_CHECK_EQUAL(fourierVisitor.getNParameters(), nDerivatives);
    std::vector<double> paramsFromVisitor = fourierVisitor.getParameterList();
    BOOST_CHECK_EQUAL_COLLECTIONS(
            parameterList.begin(), parameterList.end(),
            paramsFromVisitor.begin(), paramsFromVisitor.end()
    );

    //check that getX() , before fft throws    
    BOOST_REQUIRE_THROW(fourierVisitor.getFourierImage(), lsst::pex::exceptions::RuntimeErrorException);
    BOOST_REQUIRE_THROW(fourierVisitor.getFourierDerivativeImageList(),
                        lsst::pex::exceptions::RuntimeErrorException);

    //try to fft to too-small dimnesions
    BOOST_CHECK_THROW(fourierVisitor.fft(1,4), lsst::pex::exceptions::InvalidParameterException);

    BOOST_CHECK_NO_THROW(fourierVisitor.fft(width, height));

    FourierCutout::Ptr cutoutPtr;
    FourierCutoutVector cutoutVector;
    
    BOOST_CHECK_NO_THROW(cutoutPtr = fourierVisitor.getFourierImage());
    BOOST_CHECK_NO_THROW(cutoutVector = fourierVisitor.getFourierDerivativeImageList());

    BOOST_REQUIRE(cutoutPtr);
    BOOST_CHECK_EQUAL(cutoutPtr->getFourierWidth(), fourierWidth);
    BOOST_CHECK_EQUAL(cutoutPtr->getFourierHeight(), height);
    //shift the cutout, for comparison down below
    cutoutPtr->shift(center.first, center.second);

    BOOST_CHECK_EQUAL(cutoutVector.size(), nDerivatives);
    FourierCutoutVector::iterator i = cutoutVector.begin();
    FourierCutoutVector::iterator end(cutoutVector.end());
    for(;i != end; ++i) {
        cutoutPtr = (*i);
        BOOST_REQUIRE(cutoutPtr.get() != 0);
        BOOST_CHECK_EQUAL(cutoutPtr->getFourierWidth(), fourierWidth);
        BOOST_CHECK_EQUAL(cutoutPtr->getFourierHeight(), height);            
        
        //shift the cutout, for comparison down below
        cutoutPtr->shift(center.first, center.second);    
    }

    BOOST_CHECK_NO_THROW(cutoutPtr = fourierVisitor.getFourierImage());

    BOOST_REQUIRE(cutoutPtr);
    BOOST_CHECK_EQUAL(cutoutPtr->getFourierWidth(), fourierWidth);
    BOOST_CHECK_EQUAL(cutoutPtr->getFourierHeight(), height);
    BOOST_CHECK_EQUAL(cutoutPtr->getOwner(), cutoutVector[0]->getOwner());


    //check fft output values
    //fft shifts the zero frequency element to the center,
    

    Complex * testFourierIter = FOURIER_STACK;
    Complex * stackIter = cutoutPtr->begin();
    Complex * stackEnd = cutoutVector[nDerivatives - 1]->end();
    for(; stackIter != stackEnd; ++stackIter, ++testFourierIter) {
        double real = static_cast<double>(stackIter->real() - testFourierIter->real()), 
               imag = static_cast<double>(stackIter->imag() - testFourierIter->imag());
        BOOST_CHECK_SMALL(real, 0.00001);
        BOOST_CHECK_SMALL(imag, 0.00001);
    }


    
    cutoutVector.clear();
    cutoutPtr.reset(); 

    //fft to new image dimensions
    height = 15;
    width = 16;
    fourierWidth = width/2 +1;
    
    fourierVisitor.fft(width, height);

    BOOST_CHECK_NO_THROW(cutoutPtr = fourierVisitor.getFourierImage());
    BOOST_CHECK_NO_THROW(cutoutVector = fourierVisitor.getFourierDerivativeImageList());

    BOOST_REQUIRE(cutoutPtr);
    BOOST_CHECK_EQUAL(cutoutPtr->getFourierWidth(), fourierWidth);
    BOOST_CHECK_EQUAL(cutoutPtr->getFourierHeight(), height);

    BOOST_CHECK_EQUAL(cutoutVector.size(), nDerivatives);
    i = cutoutVector.begin();
    end = cutoutVector.end();
    for(;i != end; ++i) {
        cutoutPtr = (*i);
        BOOST_REQUIRE(cutoutPtr.get() != 0);
        BOOST_CHECK_EQUAL(cutoutPtr->getFourierWidth(), fourierWidth);
        BOOST_CHECK_EQUAL(cutoutPtr->getFourierHeight(), height);
    }
    cutoutVector.clear();
    cutoutPtr.reset();



}
