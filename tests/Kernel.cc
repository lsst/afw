#include <iostream>
#include <cmath>
#include <vector>
#include <exception>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Kernel

#include "boost/make_shared.hpp"
#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/ConvolutionVisitor.h"


typedef lsst::afw::math::Kernel Kernel;
typedef lsst::afw::math::FixedKernel FixedKernel;
typedef lsst::afw::math::LinearCombinationKernel LinearCombinationKernel;
typedef lsst::afw::math::KernelList KernelList;
typedef lsst::afw::image::Image<Kernel::Pixel> Image;
typedef lsst::afw::math::ConvolutionVisitor ConvolutionVisitor;
typedef lsst::afw::math::ImageConvolutionVisitor ImageConvolutionVisitor;
typedef lsst::afw::math::FourierConvolutionVisitor FourierConvolutionVisitor;

BOOST_AUTO_TEST_CASE(ConvolutionVisitorTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int width = 7, height = 7;
    Image img(width,height, 0);
    img(width/2 + 1, height/2 + 1) = 1;

    
    KernelList basisList;
    for(int y = 0, i = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x, ++i) {
            Image tmp(width, height, 0);
            tmp(x,y) = 1;
            basisList.push_back(boost::make_shared<FixedKernel>(tmp));
        }
    }

    FixedKernel fixedKernel(img);
    LinearCombinationKernel linearCombinationKernel(basisList, std::vector<double>(basisList.size()));

    FourierConvolutionVisitor::Ptr fourierVisitor;
    ImageConvolutionVisitor::Ptr imgVisitor;
    ConvolutionVisitor::Ptr visitorPtr;

    BOOST_CHECK_NO_THROW(imgVisitor = fixedKernel.computeImageConvolutionVisitor(
            lsst::afw::image::PointD(3.4, 0.8886))
    );
    BOOST_CHECK(imgVisitor.get() != 0);
    BOOST_CHECK_NO_THROW(fourierVisitor = fixedKernel.computeFourierConvolutionVisitor(
            lsst::afw::image::PointD(0, 1))
    );
    BOOST_CHECK(fourierVisitor.get() != 0);

    Image::Ptr imgFromVisitor = imgVisitor->getImage();

    BOOST_CHECK_EQUAL(imgFromVisitor->getHeight(), height);
    BOOST_CHECK_EQUAL(imgFromVisitor->getWidth(), width);
    
    for(int y = 0; y < height; ++y) {
        Image::x_iterator vIter = imgFromVisitor->row_begin(y);
        Image::x_iterator vEnd = imgFromVisitor->row_end(y);
        Image::x_iterator iIter = img.row_begin(y);
        for(; vIter != vEnd; ++vIter, ++iIter) {
            BOOST_CHECK_CLOSE(static_cast<double>(*vIter), static_cast<double>(*iIter), 0.00001);
        }
    }


    BOOST_CHECK_NO_THROW(imgVisitor = linearCombinationKernel.computeImageConvolutionVisitor(
            lsst::afw::image::PointD(3.4, 0.8886))
    );

    BOOST_CHECK(imgVisitor.get() != 0);
    imgFromVisitor = imgVisitor->getImage();

    BOOST_CHECK_EQUAL(imgFromVisitor->getHeight(), height);
    BOOST_CHECK_EQUAL(imgFromVisitor->getWidth(), width);
    BOOST_CHECK_EQUAL(imgVisitor->getNParameters(), width*height);

    BOOST_CHECK_NO_THROW(fourierVisitor = linearCombinationKernel.computeFourierConvolutionVisitor(
            lsst::afw::image::PointD(4.0, 33.2))
    );
    BOOST_CHECK(fourierVisitor.get() !=0);
    BOOST_CHECK_EQUAL(fourierVisitor->getNParameters(), width*height);

}
