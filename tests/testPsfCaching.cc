#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE PsfCaching
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include "boost/filesystem.hpp"

#include "ndarray/eigen.h"
#include "lsst/utils/ieee.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/detection/DoubleGaussianPsf.h"

BOOST_AUTO_TEST_CASE(FixedPsfCaching) {
    using namespace lsst::afw::detection;
    using namespace lsst::afw::geom;
    DoubleGaussianPsf psf(7, 7, 1.5, 3.0, 0.2);
    PTR(Psf::Image) im1 = psf.computeKernelImage(Point2D(0, 0), Psf::INTERNAL);
    PTR(Psf::Image) im2 = psf.computeImage(Point2D(0, 0), Psf::INTERNAL);
    BOOST_CHECK_CLOSE(im1->getArray().asEigen().sum(), 1.0, 1E-8);
    BOOST_CHECK_EQUAL(im1->getArray().asEigen(), im2->getArray().asEigen());
    PTR(Psf::Image) im3 = psf.computeKernelImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im1 == im3);
    PTR(Psf::Image) im4 = psf.computeKernelImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im3 == im4);
    PTR(Psf::Image) im5 = psf.computeImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im2 != im5);
    PTR(Psf::Image) im6 = psf.computeImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im5 == im6);
}

BOOST_AUTO_TEST_CASE(VariablePsfCaching) {
    using namespace lsst::afw::detection;
    using namespace lsst::afw::geom;
    using namespace lsst::afw::math;
    std::vector<PTR(Kernel::SpatialFunction)> spatialFuncs;
    spatialFuncs.push_back(boost::make_shared< PolynomialFunction2<double> >(1));
    spatialFuncs.push_back(boost::make_shared< PolynomialFunction2<double> >(1));
    spatialFuncs.push_back(boost::make_shared< PolynomialFunction2<double> >(0));
    spatialFuncs[0]->setParameter(0, 1.0);
    spatialFuncs[0]->setParameter(1, 0.5);
    spatialFuncs[0]->setParameter(2, 0.5);
    spatialFuncs[1]->setParameter(0, 1.0);
    spatialFuncs[1]->setParameter(1, 0.5);
    spatialFuncs[1]->setParameter(2, 0.5);
    GaussianFunction2<double> kernelFunc(1.0, 1.0);
    AnalyticKernel kernel(7, 7, kernelFunc, spatialFuncs);
    KernelPsf psf(kernel);
    PTR(Psf::Image) im1 = psf.computeKernelImage(Point2D(0, 0), Psf::INTERNAL);
    PTR(Psf::Image) im2 = psf.computeImage(Point2D(0, 0), Psf::INTERNAL);
    BOOST_CHECK_CLOSE(im1->getArray().asEigen().sum(), 1.0, 1E-8);
    BOOST_CHECK_EQUAL(im1->getArray().asEigen(), im2->getArray().asEigen());
    PTR(Psf::Image) im3 = psf.computeKernelImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im1 != im3);
    PTR(Psf::Image) im4 = psf.computeKernelImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im3 == im4);
    PTR(Psf::Image) im5 = psf.computeImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im2 != im5);
    PTR(Psf::Image) im6 = psf.computeImage(Point2D(5, 6), Psf::INTERNAL);
    BOOST_CHECK(im5 == im6);
}
