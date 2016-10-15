

#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SpanSet

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/log/Log.h"
#include "lsst/afw/geom/SpanSet.h"
#include "lsst/afw/image.h"
#include "ndarray.h"
#include "Eigen/Core"

namespace geom = lsst::afw::geom;

void testNullSpanSet() {
    // Create null SpanSet
    geom::SpanSet nullSS;
    BOOST_CHECK(nullSS.getArea() == 0);
    BOOST_CHECK(nullSS.size() == 0);
    BOOST_CHECK(nullSS.getBBox().getDimensions().getX() == 0);
    BOOST_CHECK(nullSS.getBBox().getDimensions().getY() == 0);
}

void testBBoxSpanSet() {
    // Create a SpanSet from a box
    geom::Point2I min(2, 2);
    geom::Point2I max(6, 6);
    geom::Box2I BBox(min, max);

    geom::SpanSet boxSS(BBox);
    BOOST_CHECK(boxSS.getArea() == 25);
    BOOST_CHECK(boxSS.getBBox().getMinX() == 2);
    BOOST_CHECK(boxSS.getBBox().getMinY() == 2);
    BOOST_CHECK(boxSS.getBBox().getMaxX() == 6);
    BOOST_CHECK(boxSS.getBBox().getMaxY() == 6);

    int beginY = 2;
    for (auto const & spn : boxSS) {
        BOOST_CHECK(spn.getY() == beginY);
        BOOST_CHECK(spn.getMinX() == 2);
        BOOST_CHECK(spn.getMaxX() == 6);
        ++beginY;
    }
}

void testVectorSpanSetConstructors() {
    // Test SpanSet can be created from a vector of Spans, with an iterator, copying the vector
    // and moving the vector
    geom::SpanSet SSFromVec;
    std::vector<geom::Span> SpanSetInput = {geom::Span(0, 2, 4),
                                            geom::Span(1, 2, 4),
                                            geom::Span(2, 2, 4)};
    for (int i = 0; i < 3; ++i) {
        if (i == 0)
            SSFromVec = geom::SpanSet(SpanSetInput.begin(), SpanSetInput.end());
        if (i == 1)
            SSFromVec = geom::SpanSet(SpanSetInput);
        if (i == 2 )
            SSFromVec = geom::SpanSet(std::move(SpanSetInput));

        BOOST_CHECK(SSFromVec.getArea() == 9);
        BOOST_CHECK(SSFromVec.getBBox().getMinX() == 2);
        BOOST_CHECK(SSFromVec.getBBox().getMinY() == 0);
        BOOST_CHECK(SSFromVec.getBBox().getMaxX() == 4);
        BOOST_CHECK(SSFromVec.getBBox().getMaxY() == 2);

        int beginY = 0;
        for (auto const & spn : SSFromVec) {
            BOOST_CHECK(spn.getY() == beginY);
            BOOST_CHECK(spn.getMinX() == 2);
            BOOST_CHECK(spn.getMaxX() == 4);
            ++beginY;
        }
    }
}

void testContiguous() {
    // Contiguous SpanSet
    std::vector<geom::Span> SpanSetConVec = {geom::Span(0, 2, 5), geom::Span(1, 5, 8)};
    geom::SpanSet SSCont(SpanSetConVec);

    BOOST_CHECK(SSCont.isContiguous() == true);

    // Almost Contiguous SpanSet (tested to verify overlap function logic)
    std::vector<geom::Span> SpanSetAlmostConVec = {geom::Span(0, 2, 5), geom::Span(1, 6, 9)};
    geom::SpanSet SSAlmostCont(SpanSetAlmostConVec);
    BOOST_CHECK(SSAlmostCont.isContiguous() == false);

    // Not Contiguous SpanSet
    std::vector<geom::Span> SpanSetNotConVec = {geom::Span(0, 2, 5), geom::Span(1, 20, 25)};
    geom::SpanSet SSNotCont(SpanSetNotConVec);
    BOOST_CHECK(SSNotCont.isContiguous() == false);

    // Test has hole in one row
    std::vector<geom::Span> SpanSetWithHoleVec = {geom::Span(0, 1, 3),
                                                  geom::Span(0, 5, 7),
                                                  geom::Span(1, 0, 6)};
    geom::SpanSet SSWithHole(SpanSetWithHoleVec);
    BOOST_CHECK(SSWithHole.isContiguous() == true);
}

void testShiftedBy() {
    geom::Extent2I SSextent(geom::Point2I(2, 2));
    // BBox lower corner should be at -2,-2
    auto SpanSetNoShift = geom::SpanSet::spanSetFromShape(2);
    auto shiftedSpanSet = SpanSetNoShift.shiftedBy(SSextent);

    BOOST_CHECK(shiftedSpanSet.getBBox().getMinX() == 0);
    BOOST_CHECK(shiftedSpanSet.getBBox().getMinY() == 0);
}

void testClippedTo() {
    geom::Box2I clipBox(geom::Box2I(geom::Point2I(-2, -2), geom::Point2I(2, 2)));
    // BBox lower corner shouuld be at -4,-4
    auto SpanSetNoClip = geom::SpanSet::spanSetFromShape(4);
    auto SpanSetClip = SpanSetNoClip.clippedTo(clipBox);

    BOOST_CHECK(SpanSetClip.getBBox().getMinX() == -2);
    BOOST_CHECK(SpanSetClip.getBBox().getMinY() == -2);
}

void testTransformed() {
    // This linear transform matrix should scale all the points by a factor of 2
    Eigen::Matrix2d scaleMatrix;
    scaleMatrix.row(0) << 2, 0;
    scaleMatrix.row(1) << 0, 2;
    geom::LinearTransform transform(scaleMatrix);
    auto SpanSetPreScale = geom::SpanSet::spanSetFromShape(2);
    // transformedBy chains from LinearTransform->AffineTransform->XYTransform
    // so testing LinearTransform tests all function overloads
    auto SpanSetPostScale = SpanSetPreScale.transformedBy(transform);

    BOOST_CHECK(SpanSetPostScale.getBBox().getMinX() == -4);
    BOOST_CHECK(SpanSetPostScale.getBBox().getMinY() == -4);
}

void testOverlaps() {
    auto SpanSetNoShift = geom::SpanSet::spanSetFromShape(4);
    auto SpanSetShift = SpanSetNoShift.shiftedBy(2, 2);
    auto SpanSetShiftFar = SpanSetNoShift.shiftedBy(10, 10);

    BOOST_CHECK(SpanSetNoShift.overlaps(SpanSetShift) == true);
    BOOST_CHECK(SpanSetNoShift.overlaps(SpanSetShiftFar) == false);
}

void testContains() {
    auto SpanSetLarge = geom::SpanSet::spanSetFromShape(4);
    auto SpanSetSmall = geom::SpanSet::spanSetFromShape(1).shiftedBy(1, 1);
    auto SpanSetSmallFar = geom::SpanSet::spanSetFromShape(1).shiftedBy(8, 8);
    geom::Point2I pointIn(1, 1);
    geom::Point2I pointOut(20, 20);

    BOOST_CHECK(SpanSetLarge.contains(SpanSetSmall) == true);
    BOOST_CHECK(SpanSetLarge.contains(SpanSetSmallFar) == false);
    BOOST_CHECK(SpanSetLarge.contains(pointIn) == true);
    BOOST_CHECK(SpanSetLarge.contains(pointOut) == false);
}

void testComputeCentroid() {
    auto SpanSetShape = geom::SpanSet::spanSetFromShape(4).shiftedBy(2, 2);
    auto center = SpanSetShape.computeCentroid();

    BOOST_CHECK(center.getX() == 2);
    BOOST_CHECK(center.getY() == 2);
}

void testComputeShape() {
    auto SpanSetShape = geom::SpanSet::spanSetFromShape(1);
    auto quad = SpanSetShape.computeShape();

    BOOST_CHECK(quad.getIxx() == 0.4);
    BOOST_CHECK(quad.getIyy() == 0.4);
    BOOST_CHECK(quad.getIxy() == 0);
}

void testDilate() {
    // Only need to test use of a stencil as it delegates to the other function overload
    auto SpanSetPreDilate = geom::SpanSet::spanSetFromShape(1);
    // This should dilate the shape by 1
    auto SpanSetPostDilate = SpanSetPreDilate.dilate(1);

    BOOST_CHECK(SpanSetPostDilate.getBBox().getMinX() == -2);
    BOOST_CHECK(SpanSetPostDilate.getBBox().getMinY() == -2);
}

void testErode() {
    // Only need to test use of a stencil as it delegates to the other function overload
    auto SpanSetPreErode = geom::SpanSet::spanSetFromShape(2);
    // This should erode the shape by 1
    auto SpanSetPostErode = SpanSetPreErode.erode(1);

    BOOST_CHECK(SpanSetPostErode.getBBox().getMinX() == -1);
    BOOST_CHECK(SpanSetPostErode.getBBox().getMinY() == -1);
}

void testFlatten() {
    // Test version without output array, as this simply delegates
    // Create an array and initialize it to 9
    ndarray::Array<int, 2, 2> input = ndarray::allocate(ndarray::makeVector(6, 6));
    input.deep() = 9;

    // overload a few pixels with the value 1
    input[1][1] = 1;
    input[1][2] = 1;
    input[2][1] = 1;
    input[2][2] = 1;
    // Create a spanSet for those points
    std::vector<geom::Span> spanVector = {geom::Span(0, 0, 1), geom::Span(1, 0, 1)};
    geom::SpanSet SpnSt(spanVector);
    auto flatArr = SpnSt.flatten(input, geom::Point2I(-1, -1));
    for (auto const &  val : flatArr) {
        BOOST_CHECK(val == 1);
    }
}

void testUnflatten() {
    // Test version without output array, as this simply delegates
    ndarray::Array<int, 1, 1> input = ndarray::allocate(6);
    input.deep() = 9;

    // Create a SpanSet which has six points int it
    std::vector<geom::Span> spanVector = {geom::Span(9, 2, 3), geom::Span(10, 3, 4), geom::Span(11, 2, 3)};
    geom::SpanSet SpnSt(spanVector);
    auto output = SpnSt.unflatten(input);

    // Loop over each point in the array correspointing to the spanset, minux xy0 (the min of the SpanSet)
    // verify the array contains the value 9
    for (auto const & spn : SpnSt) {
        for (auto const & pt : spn) {
            BOOST_CHECK(output[pt.getY()-9][pt.getX()-2] == 9);
        }
    }
    // Verify that a point outside the vector has the value zero
    BOOST_CHECK(output[0][2] == 0);
}

std::pair<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, geom::SpanSet> populateMask() {
    // Create a mask and populate it with the value 2
    lsst::afw::image::Mask<lsst::afw::image::MaskPixel> msk(10, 10);
    auto spnSt = geom::SpanSet::spanSetFromShape(3).shiftedBy(5, 5);
    spnSt.setMask(msk, static_cast<lsst::afw::image::MaskPixel>(2));
    return std::pair<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, geom::SpanSet>(msk, spnSt);
}

void testSetMask() {
    auto ret = populateMask();
    auto msk = ret.first;
    auto spnSt = ret.second;
    // Verify the mask was populated with the value 2
    auto mskArray = msk.getArray();
    for (auto const & spn : spnSt) {
        for (auto const & pt : spn) {
            BOOST_CHECK(mskArray[pt.getY()][pt.getX()] == static_cast<lsst::afw::image::MaskPixel>(2));
        }
    }
}

void testClearMask() {
    auto ret = populateMask();
    auto msk = ret.first;
    auto spnSt = ret.second;
    spnSt.clearMask(msk, static_cast<lsst::afw::image::MaskPixel>(2));
    // Verify the mask is now zero
    auto mskArray = msk.getArray();
    for (auto const & spn : spnSt) {
        for (auto const & pt : spn) {
            BOOST_CHECK(mskArray[pt.getY()][pt.getX()] == static_cast<lsst::afw::image::MaskPixel>(0));
        }
    }
}

std::pair<geom::SpanSet, geom::SpanSet> makeOverlapSpanSets() {
    using SS = geom::SpanSet;
    auto firstSpanSet = SS::spanSetFromShape(2, geom::Stencil::BOX).shiftedBy(2, 4);
    auto secondSpanSet = SS::spanSetFromShape(2, geom::Stencil::BOX).shiftedBy(2, 2);
    return std::pair<geom::SpanSet, geom::SpanSet>(firstSpanSet, secondSpanSet);
}

void testIntersect() {
    auto ret = makeOverlapSpanSets();
    auto firstSS = ret.first;
    auto secondSS = ret.second;

    auto overlap = firstSS.intersect(secondSS);

    // Verify the two boxes overlap in y range 2->4
    int yStart = 2;
    for (auto const & spn : overlap) {
        BOOST_CHECK(spn.getY() == yStart);
        BOOST_CHECK(spn.getMinX() == 0);
        BOOST_CHECK(spn.getMaxX() == 4);
        ++yStart;
    }
}

void testIntersectNot() {
    auto ret = makeOverlapSpanSets();
    auto firstSS = ret.first;
    auto secondSS = ret.second;

    auto overlap = firstSS.intersectNot(secondSS);

    // Verify the result is in the y range 4->6
    int yStart = 5;
    for (auto const & spn : overlap) {
        BOOST_CHECK(spn.getY() == yStart);
        BOOST_CHECK(spn.getMinX() == 0);
        BOOST_CHECK(spn.getMaxX() == 4);
        ++yStart;
    }
}

void testUnion() {
    auto ret = makeOverlapSpanSets();
    auto firstSS = ret.first;
    auto secondSS = ret.second;

    auto overlap = firstSS.union_(secondSS);

    // Verify both SpanSets are combined together
    int yStart = 0;
    for (auto const & spn : overlap) {
        BOOST_CHECK(spn.getY() == yStart);
        BOOST_CHECK(spn.getMinX() == 0);
        BOOST_CHECK(spn.getMaxX() == 4);
        ++yStart;
    }
}

void testEquality() {
    auto ret = makeOverlapSpanSets();
    auto firstSS = ret.first;
    auto secondSS = ret.second;
    auto secondSSShift = secondSS.shiftedBy(0, 2);

    BOOST_CHECK((firstSS == secondSS) == false);
    BOOST_CHECK((firstSS != secondSS) == true);
    BOOST_CHECK((firstSS == secondSSShift) == true);
    BOOST_CHECK((firstSS != secondSSShift) == false);
}

void testFunctor() {
    // Test the remaining types of functors. Above code has tested ndarray functors
    // need to test, constants, iterators, Images
    lsst::afw::image::Image<int> imageObject(5, 5, 0);
    std::vector<int> vecObject(25, 0);

    auto SSShape = geom::SpanSet::spanSetFromShape(2, geom::Stencil::BOX).shiftedBy(2, 2);
    // use a constant as a test of constantGetter
    namespace afwImage = lsst::afw::image;
    std::cout << "before apply" << std::endl;
    for (auto & val : SSShape) {
        std::cout << "SS is " << val << std::endl;
    }
    SSShape.applyFunctor([](geom::Point2I pt, int n, afwImage::Image<int>::Pixel & out, int in){ out = in; },
                            imageObject, 6);
    SSShape.applyFunctor([](geom::Point2I, int n, int & out, int in){out = in;}, vecObject.begin(), 6);
    auto imageIter = imageObject.begin();
    auto vectorIter = vecObject.begin();
    for (int i = 0; i < 25; ++i, ++imageIter, ++vectorIter) {
        BOOST_CHECK(*imageIter == 6);
        BOOST_CHECK(*vectorIter == 6);
    }
}


BOOST_AUTO_TEST_CASE(SpanSet_test) {
    testNullSpanSet();
    testBBoxSpanSet();
    testVectorSpanSetConstructors();
    // getArea, and getBBox tested above in the tests of the constructors
    testContiguous();
    testShiftedBy();
    testClippedTo();
    testTransformed();
    testOverlaps();
    testContains();
    testComputeCentroid();
    testComputeShape();
    testDilate();
    testErode();
    testFlatten();
    testUnflatten();
    testSetMask();
    testClearMask();
    testIntersect();
    testIntersectNot();
    testUnion();
    testEquality();
    testFunctor();
}
