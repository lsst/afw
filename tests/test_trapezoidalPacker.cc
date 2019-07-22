#define BOOST_TEST_MODULE TrapezoidalPacker
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/format.hpp"

#include "ndarray/eigen.h"
#include "lsst/afw/math/detail/TrapezoidalPacker.h"

typedef lsst::afw::math::detail::TrapezoidalPacker Packer;
typedef lsst::afw::math::ChebyshevBoundedFieldControl Control;

ndarray::Array<double, 1, 1> makeRandomArray(int n) {
    ndarray::Array<double, 1, 1> result = ndarray::allocate(n);
    ndarray::asEigenArray(result).setRandom();
    return result;
}

void compareArrays(int n, double const* a, double const* b) {
    for (int i = 0; i < n; ++i) {
        BOOST_CHECK_EQUAL(a[i], b[i]);
    }
}

void checkUnpacked(ndarray::Array<double const, 2, 2> const& unpacked,
                   ndarray::Array<double const, 1, 1> const& tx, ndarray::Array<double const, 1, 1> const& ty,
                   Control const& ctrl) {
    for (int i = 0; i <= ctrl.orderY; ++i) {
        for (int j = 0; j <= ctrl.orderX; ++j) {
            if (ctrl.triangular && ((i + j) > std::max(ctrl.orderX, ctrl.orderY))) {
                BOOST_CHECK_EQUAL(unpacked[i][j], 0.0);
            } else {
                BOOST_CHECK_EQUAL(unpacked[i][j], tx[j] * ty[i]);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(WideTrapezoidal) {
    Control ctrl;
    ctrl.orderX = 4;
    ctrl.orderY = 3;
    ctrl.triangular = true;
    Packer packer(ctrl);
    BOOST_CHECK_EQUAL(packer.nx, 5);
    BOOST_CHECK_EQUAL(packer.ny, 4);
    BOOST_CHECK_EQUAL(packer.m, 0);
    BOOST_CHECK_EQUAL(packer.size, 14);
    ndarray::Array<double, 1, 1> out1 = ndarray::allocate(packer.size);
    ndarray::Array<double, 1, 1> tx = makeRandomArray(packer.nx);
    ndarray::Array<double, 1, 1> ty = makeRandomArray(packer.ny);
    packer.pack(out1, tx, ty);
    double const check1[] = {tx[0] * ty[0], tx[1] * ty[0], tx[2] * ty[0], tx[3] * ty[0], tx[4] * ty[0],
                             tx[0] * ty[1], tx[1] * ty[1], tx[2] * ty[1], tx[3] * ty[1], tx[0] * ty[2],
                             tx[1] * ty[2], tx[2] * ty[2], tx[0] * ty[3], tx[1] * ty[3]};
    compareArrays(packer.size, check1, out1.begin());
    checkUnpacked(packer.unpack(out1), tx, ty, ctrl);
}

BOOST_AUTO_TEST_CASE(TallTrapezoidal) {
    Control ctrl;
    ctrl.orderX = 2;
    ctrl.orderY = 4;
    ctrl.triangular = true;
    Packer packer(ctrl);
    BOOST_CHECK_EQUAL(packer.nx, 3);
    BOOST_CHECK_EQUAL(packer.ny, 5);
    BOOST_CHECK_EQUAL(packer.m, 2);
    BOOST_CHECK_EQUAL(packer.size, 12);
    ndarray::Array<double, 1, 1> out1 = ndarray::allocate(packer.size);
    ndarray::Array<double, 1, 1> tx = makeRandomArray(packer.nx);
    ndarray::Array<double, 1, 1> ty = makeRandomArray(packer.ny);
    packer.pack(out1, tx, ty);
    double const check1[] = {tx[0] * ty[0], tx[1] * ty[0], tx[2] * ty[0], tx[0] * ty[1],
                             tx[1] * ty[1], tx[2] * ty[1], tx[0] * ty[2], tx[1] * ty[2],
                             tx[2] * ty[2], tx[0] * ty[3], tx[1] * ty[3], tx[0] * ty[4]};
    compareArrays(packer.size, check1, out1.begin());
    checkUnpacked(packer.unpack(out1), tx, ty, ctrl);
}

BOOST_AUTO_TEST_CASE(Triangular) {
    Control ctrl;
    ctrl.orderX = 3;
    ctrl.orderY = 3;
    ctrl.triangular = true;
    Packer packer(ctrl);
    BOOST_CHECK_EQUAL(packer.nx, 4);
    BOOST_CHECK_EQUAL(packer.ny, 4);
    BOOST_CHECK_EQUAL(packer.m, 0);
    BOOST_CHECK_EQUAL(packer.size, 10);
    ndarray::Array<double, 1, 1> out1 = ndarray::allocate(packer.size);
    ndarray::Array<double, 1, 1> tx = makeRandomArray(packer.nx);
    ndarray::Array<double, 1, 1> ty = makeRandomArray(packer.ny);
    packer.pack(out1, tx, ty);
    double const check1[] = {tx[0] * ty[0], tx[1] * ty[0], tx[2] * ty[0], tx[3] * ty[0], tx[0] * ty[1],
                             tx[1] * ty[1], tx[2] * ty[1], tx[0] * ty[2], tx[1] * ty[2], tx[0] * ty[3]};
    compareArrays(packer.size, check1, out1.begin());
    checkUnpacked(packer.unpack(out1), tx, ty, ctrl);
}

BOOST_AUTO_TEST_CASE(WideRectangular) {
    Control ctrl;
    ctrl.orderX = 3;
    ctrl.orderY = 2;
    ctrl.triangular = false;
    Packer packer(ctrl);
    BOOST_CHECK_EQUAL(packer.nx, 4);
    BOOST_CHECK_EQUAL(packer.ny, 3);
    BOOST_CHECK_EQUAL(packer.m, 3);
    BOOST_CHECK_EQUAL(packer.size, 12);
    ndarray::Array<double, 1, 1> out1 = ndarray::allocate(packer.size);
    ndarray::Array<double, 1, 1> tx = makeRandomArray(packer.nx);
    ndarray::Array<double, 1, 1> ty = makeRandomArray(packer.ny);
    packer.pack(out1, tx, ty);
    double const check1[] = {tx[0] * ty[0], tx[1] * ty[0], tx[2] * ty[0], tx[3] * ty[0],
                             tx[0] * ty[1], tx[1] * ty[1], tx[2] * ty[1], tx[3] * ty[1],
                             tx[0] * ty[2], tx[1] * ty[2], tx[2] * ty[2], tx[3] * ty[2]};
    compareArrays(packer.size, check1, out1.begin());
    ndarray::Array<double, 2, 2> out2 = packer.unpack(out1);
}

BOOST_AUTO_TEST_CASE(TallRectangular) {
    Control ctrl;
    ctrl.orderX = 2;
    ctrl.orderY = 3;
    ctrl.triangular = false;
    Packer packer(ctrl);
    BOOST_CHECK_EQUAL(packer.nx, 3);
    BOOST_CHECK_EQUAL(packer.ny, 4);
    BOOST_CHECK_EQUAL(packer.m, 4);
    BOOST_CHECK_EQUAL(packer.size, 12);
    ndarray::Array<double, 1, 1> out1 = ndarray::allocate(packer.size);
    ndarray::Array<double, 1, 1> tx = makeRandomArray(packer.nx);
    ndarray::Array<double, 1, 1> ty = makeRandomArray(packer.ny);
    packer.pack(out1, tx, ty);
    double const check1[] = {tx[0] * ty[0], tx[1] * ty[0], tx[2] * ty[0], tx[0] * ty[1],
                             tx[1] * ty[1], tx[2] * ty[1], tx[0] * ty[2], tx[1] * ty[2],
                             tx[2] * ty[2], tx[0] * ty[3], tx[1] * ty[3], tx[2] * ty[3]};
    compareArrays(packer.size, check1, out1.begin());
    ndarray::Array<double, 2, 2> out2 = packer.unpack(out1);
}
