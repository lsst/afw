/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */


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
#include "lsst/afw/fits.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "ndarray.h"
#include "Eigen/Core"

namespace afwGeom = lsst::afw::geom;

BOOST_AUTO_TEST_CASE(SpanSet_testNullSpanSet) {
    // Create null SpanSet
    afwGeom::SpanSet nullSS;
    BOOST_CHECK(nullSS.getArea() == 0);
    BOOST_CHECK(nullSS.size() == 0);
    BOOST_CHECK(nullSS.getBBox().getDimensions().getX() == 0);
    BOOST_CHECK(nullSS.getBBox().getDimensions().getY() == 0);
}

BOOST_AUTO_TEST_CASE(SpanSet_testBBoxSpanSet) {
    // Create a SpanSet from a box
    afwGeom::Point2I min(2, 2);
    afwGeom::Point2I max(6, 6);
    afwGeom::Box2I BBox(min, max);

    afwGeom::SpanSet boxSS(BBox);
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

BOOST_AUTO_TEST_CASE(SpanSet_testVectorSpanSetConstructors)
{
    // Test SpanSet can be created from a vector of Spans, with an iterator, copying the vector
    // and moving the vector
    std::shared_ptr<afwGeom::SpanSet> SSFromVec;
    std::vector<afwGeom::Span> SpanSetInput = {afwGeom::Span(0, 2, 4),
                                               afwGeom::Span(1, 2, 4),
                                               afwGeom::Span(2, 2, 4)};
    for (int i = 0; i < 3; ++i) {
        if (i == 0)
            SSFromVec = std::make_shared<afwGeom::SpanSet>(SpanSetInput.begin(), SpanSetInput.end());
        if (i == 1)
            SSFromVec = std::make_shared<afwGeom::SpanSet>(SpanSetInput);
        if (i == 2 )
            SSFromVec = std::make_shared<afwGeom::SpanSet>(std::move(SpanSetInput));

        BOOST_CHECK(SSFromVec->getArea() == 9);
        BOOST_CHECK(SSFromVec->getBBox().getMinX() == 2);
        BOOST_CHECK(SSFromVec->getBBox().getMinY() == 0);
        BOOST_CHECK(SSFromVec->getBBox().getMaxX() == 4);
        BOOST_CHECK(SSFromVec->getBBox().getMaxY() == 2);

        int beginY = 0;
        for (auto const & spn : *SSFromVec) {
            BOOST_CHECK(spn.getY() == beginY);
            BOOST_CHECK(spn.getMinX() == 2);
            BOOST_CHECK(spn.getMaxX() == 4);
            ++beginY;
        }
    }
}

BOOST_AUTO_TEST_CASE(SpanSet_testContiguous) {
    // Contiguous SpanSet
    std::vector<afwGeom::Span> SpanSetConVec = {afwGeom::Span(0, 2, 5), afwGeom::Span(1, 5, 8)};
    afwGeom::SpanSet SSCont(SpanSetConVec);

    BOOST_CHECK(SSCont.isContiguous() == true);

    // Almost Contiguous SpanSet (tested to verify overlap function logic)
    std::vector<afwGeom::Span> SpanSetAlmostConVec = {afwGeom::Span(0, 2, 5), afwGeom::Span(1, 6, 9)};
    afwGeom::SpanSet SSAlmostCont(SpanSetAlmostConVec);
    BOOST_CHECK(SSAlmostCont.isContiguous() == false);

    // Not Contiguous SpanSet
    std::vector<afwGeom::Span> SpanSetNotConVec = {afwGeom::Span(0, 2, 5), afwGeom::Span(1, 20, 25)};
    afwGeom::SpanSet SSNotCont(SpanSetNotConVec);
    BOOST_CHECK(SSNotCont.isContiguous() == false);

    // Test has hole in one row
    std::vector<afwGeom::Span> SpanSetWithHoleVec = {afwGeom::Span(0, 1, 3),
                                                     afwGeom::Span(0, 5, 7),
                                                     afwGeom::Span(1, 0, 6)};
    afwGeom::SpanSet SSWithHole(SpanSetWithHoleVec);
    BOOST_CHECK(SSWithHole.isContiguous() == true);

    // Test a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    BOOST_CHECK(nullSpanSet.isContiguous() == true);
}

BOOST_AUTO_TEST_CASE(SpanSet_testSplit) {
    std::vector<afwGeom::Span> SpanSetConVec = {afwGeom::Span(0, 2, 5), afwGeom::Span(1, 5, 8)};
    afwGeom::SpanSet SSCont(SpanSetConVec);

    //This should return only 1 entry
    auto result1 = SSCont.split();
    BOOST_CHECK(result1.size() == 1);
    int i = 0;
    for (auto const & value : *(result1[0])) {
        BOOST_CHECK(value == SpanSetConVec[i]);
        ++i;
    }
    std::vector<afwGeom::Span> SpanSetNotConVec = {afwGeom::Span(0, 2, 5),
                                                   afwGeom::Span(1, 20, 25),
                                                   afwGeom::Span(2, 19,23)};
    afwGeom::SpanSet SSNotCont(SpanSetNotConVec.begin(), SpanSetNotConVec.end());

    // This should return two entries
    auto result2 = SSNotCont.split();
    BOOST_CHECK(result2.size() == 2);

    // Check that the first one has the same entries as the first entry in the list
    afwGeom::SpanSet const & tempSpanSet1 = *(result2[0]);
    auto vectorIterator = SpanSetNotConVec.begin();
    for (auto const & value : tempSpanSet1) {
        BOOST_CHECK(value == *vectorIterator);
        ++vectorIterator;
    }

    // Check that the second group contains the last two entries from the list
    auto const & tempSpanSet2 = *(result2[1]);
    for (auto const & value : tempSpanSet2) {
        BOOST_CHECK(value == *vectorIterator);
        ++vectorIterator;
    }
    // Check for a nullSpanSet
    afwGeom::SpanSet nullSpanSet;
    auto result3 = nullSpanSet.split();
    BOOST_CHECK(*(result3[0]) == nullSpanSet);
}

BOOST_AUTO_TEST_CASE(SpanSet_testShiftedBy) {
    afwGeom::Extent2I SSextent(afwGeom::Point2I(2, 2));
    // BBox lower corner should be at -2,-2
    auto SpanSetNoShift = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::CIRCLE);
    auto shiftedSpanSet = SpanSetNoShift->shiftedBy(SSextent);

    BOOST_CHECK(shiftedSpanSet->getBBox().getMinX() == 0);
    BOOST_CHECK(shiftedSpanSet->getBBox().getMinY() == 0);

    // Try to shift a nullSpanSet
    afwGeom::SpanSet nullSpanSet;
    auto shiftedNullSpanSet = nullSpanSet.shiftedBy(SSextent);
    BOOST_CHECK(shiftedNullSpanSet->size() == 0);
}

BOOST_AUTO_TEST_CASE(SpanSet_testClippedTo) {
    afwGeom::Box2I clipBox(afwGeom::Box2I(afwGeom::Point2I(-2, -2), afwGeom::Point2I(2, 2)));
    // BBox lower corner shouuld be at -4,-4
    auto SpanSetNoClip = afwGeom::SpanSet::fromShape(4, afwGeom::Stencil::CIRCLE);
    auto SpanSetClip = SpanSetNoClip->clippedTo(clipBox);

    BOOST_CHECK(SpanSetClip->getBBox().getMinX() == -2);
    BOOST_CHECK(SpanSetClip->getBBox().getMinY() == -2);

    // Try clipping a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto clippedNullSpanSet = nullSpanSet.clippedTo(clipBox);
    BOOST_CHECK(clippedNullSpanSet->size() == 0);
}

BOOST_AUTO_TEST_CASE(SpanSet_testTransformed) {
    // This linear transform matrix should scale all the points by a factor of 2
    Eigen::Matrix2d scaleMatrix;
    scaleMatrix.row(0) << 2, 0;
    scaleMatrix.row(1) << 0, 2;
    afwGeom::LinearTransform transform(scaleMatrix);
    auto SpanSetPreScale = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::CIRCLE);
    // transformedBy chains from LinearTransform->AffineTransform->XYTransform
    // so testing LinearTransform tests all function overloads
    auto SpanSetPostScale = SpanSetPreScale->transformedBy(transform);

    BOOST_CHECK(SpanSetPostScale->getBBox().getMinX() == -4);
    BOOST_CHECK(SpanSetPostScale->getBBox().getMinY() == -4);

    // Try transforming a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto transformedNullSpanSet = nullSpanSet.transformedBy(transform);
    BOOST_CHECK(transformedNullSpanSet->size() == 0);
}

BOOST_AUTO_TEST_CASE(SpanSet_testOverlaps) {
    auto SpanSetNoShift = afwGeom::SpanSet::fromShape(4, afwGeom::Stencil::CIRCLE);
    auto SpanSetShift = SpanSetNoShift->shiftedBy(2, 2);
    auto SpanSetShiftFar = SpanSetNoShift->shiftedBy(10, 10);

    BOOST_CHECK(SpanSetNoShift->overlaps(*SpanSetShift) == true);
    BOOST_CHECK(SpanSetNoShift->overlaps(*SpanSetShiftFar) == false);

    // Try with a nullSpanSet, null spansets should not overlap anything
    afwGeom::SpanSet nullSpanSet;
    BOOST_CHECK(SpanSetNoShift->overlaps(nullSpanSet) == false);
    BOOST_CHECK(nullSpanSet.overlaps(*SpanSetNoShift) == false);
}

BOOST_AUTO_TEST_CASE(SpanSet_testContains) {
    auto SpanSetLarge = afwGeom::SpanSet::fromShape(4, afwGeom::Stencil::CIRCLE);
    auto SpanSetSmall = afwGeom::SpanSet::fromShape(1, afwGeom::Stencil::CIRCLE)->shiftedBy(1, 1);
    auto SpanSetSmallFar = afwGeom::SpanSet::fromShape(1, afwGeom::Stencil::CIRCLE)->shiftedBy(8, 8);
    afwGeom::Point2I pointIn(1, 1);
    afwGeom::Point2I pointOut(20, 20);

    BOOST_CHECK(SpanSetLarge->contains(*SpanSetSmall) == true);
    BOOST_CHECK(SpanSetLarge->contains(*SpanSetSmallFar) == false);
    BOOST_CHECK(SpanSetLarge->contains(pointIn) == true);
    BOOST_CHECK(SpanSetLarge->contains(pointOut) == false);

    // Test with a nullSpanSet
    afwGeom::SpanSet nullSpanSet;
    BOOST_CHECK(SpanSetLarge->contains(nullSpanSet) == false);
    BOOST_CHECK(nullSpanSet.contains(*SpanSetSmall) == false);
}

BOOST_AUTO_TEST_CASE(SpanSet_testComputeCentroid) {
    auto SpanSetShape = afwGeom::SpanSet::fromShape(4, afwGeom::Stencil::CIRCLE)->shiftedBy(2, 2);
    auto center = SpanSetShape->computeCentroid();

    BOOST_CHECK(center.getX() == 2);
    BOOST_CHECK(center.getY() == 2);

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullCenter = nullSpanSet.computeCentroid();
    BOOST_CHECK(std::isnan(nullCenter.getX()));
    BOOST_CHECK(std::isnan(nullCenter.getY()));

}

BOOST_AUTO_TEST_CASE(SpanSet_testComputeShape) {
    auto SpanSetShape = afwGeom::SpanSet::fromShape(1, afwGeom::Stencil::CIRCLE);
    auto quad = SpanSetShape->computeShape();

    BOOST_CHECK(quad.getIxx() == 0.4);
    BOOST_CHECK(quad.getIyy() == 0.4);
    BOOST_CHECK(quad.getIxy() == 0);

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullShape = nullSpanSet.computeShape();
    BOOST_CHECK(std::isnan(nullShape.getIxx()));
    BOOST_CHECK(std::isnan(nullShape.getIyy()));
    BOOST_CHECK(std::isnan(nullShape.getIxy()));
}

BOOST_AUTO_TEST_CASE(SpanSet_testdilated) {
    // Only need to test use of a stencil as it delegates to the other function overload
    auto SpanSetPredilated = afwGeom::SpanSet::fromShape(1, afwGeom::Stencil::CIRCLE);
    // This should dilated the shape by 1
    auto SpanSetPostdilated = SpanSetPredilated->dilated(1);

    BOOST_CHECK(SpanSetPostdilated->getBBox().getMinX() == -2);
    BOOST_CHECK(SpanSetPostdilated->getBBox().getMinY() == -2);

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullSpanSetDialte = nullSpanSet.dilated(1);

    BOOST_CHECK(nullSpanSetDialte->getBBox().getMinX() == 0);
    BOOST_CHECK(nullSpanSetDialte->getBBox().getMinY() == 0);

    auto SpanSetNulldilated = SpanSetPredilated->dilated(nullSpanSet);
    BOOST_CHECK(SpanSetNulldilated->getBBox().getMinX() == -1);
    BOOST_CHECK(SpanSetNulldilated->getBBox().getMinY() == -1);

}

BOOST_AUTO_TEST_CASE(SpanSet_testeroded) {
    // Only need to test use of a stencil as it delegates to the other function overload
    auto SpanSetPreeroded = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::CIRCLE);
    // This should eroded the shape by 1
    auto SpanSetPosteroded = SpanSetPreeroded->eroded(1);

    BOOST_CHECK(SpanSetPosteroded->getBBox().getMinX() == -1);
    BOOST_CHECK(SpanSetPosteroded->getBBox().getMinY() == -1);

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullSpanSeteroded = nullSpanSet.eroded(1);

    BOOST_CHECK(nullSpanSeteroded->getBBox().getMinX() == 0);
    BOOST_CHECK(nullSpanSeteroded->getBBox().getMinY() == 0);

    auto SpanSetNulleroded = SpanSetPreeroded->eroded(nullSpanSet);
    BOOST_CHECK(SpanSetNulleroded->getBBox().getMinX() == -2);
    BOOST_CHECK(SpanSetNulleroded->getBBox().getMinY() == -2);
}

BOOST_AUTO_TEST_CASE(SpanSet_testFlatten) {
    // Test version without output array, as this simply delegates
    // Create an array and initialize it to 9
    ndarray::Array<int, 2, 2> input = ndarray::allocate(ndarray::makeVector(6, 6));
    int dummyArrayValue = 9;
    input.deep() = dummyArrayValue;

    // overload a few pixels with the different values
    input[1][1] = 1;
    input[1][2] = 2;
    input[2][1] = 3;
    input[2][2] = 4;
    // Create a SpanSet for those points
    std::vector<afwGeom::Span> spanVector = {afwGeom::Span(0, 0, 1), afwGeom::Span(1, 0, 1)};
    afwGeom::SpanSet spnSt(spanVector);
    auto flatArr = spnSt.flatten(input, afwGeom::Point2I(-1, -1));
    // Verify the size of the flattened array is the same as the number of pixels in the SpanSet
    BOOST_CHECK(flatArr.size() == spnSt.getArea());
    // Verify that the output values are the same, and in the same order, as the input array
    auto inputValue = 1;
    for (auto const &  val : flatArr) {
        BOOST_CHECK(val == inputValue);
        ++inputValue;
    }

    // Test with an explicit ouput array
    ndarray::Array<int, 1, 1> flatOutput = ndarray::allocate(ndarray::makeVector(4));
    flatOutput.deep() = 0;

    std::vector<afwGeom::Span> spanVectorOriginAlign = {afwGeom::Span(1,1,2), afwGeom::Span(2,1,2)};
    afwGeom::SpanSet spnStOriginAlign(spanVectorOriginAlign);


    spnStOriginAlign.flatten(flatOutput, input);

    auto nextInputValue = 1;
    for (auto const & val : flatOutput) {
        BOOST_CHECK(val == nextInputValue);
        ++nextInputValue;
    }

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullFlatArray = nullSpanSet.flatten(input, afwGeom::Point2I(-1, -1));
    BOOST_CHECK(nullFlatArray.size() == 0);
}

BOOST_AUTO_TEST_CASE(SpanSet_testUnflatten) {
    // Test version without output array, as this simply delegates
    ndarray::Array<int, 1, 1> input = ndarray::allocate(6);
    int dummyArrayValue = 4;
    input.deep() = dummyArrayValue;

    // Create a SpanSet which has six points int it
    std::vector<afwGeom::Span> spanVector = {afwGeom::Span(9, 2, 3), afwGeom::Span(10, 3, 4), afwGeom::Span(11, 2, 3)};
    afwGeom::SpanSet spnSt(spanVector);
    auto output = spnSt.unflatten(input);

    // Verify that the output has the correct dimensions
    auto arrayShape = output.getShape();
    BOOST_CHECK(static_cast<int>(arrayShape[0]) == spnSt.getBBox().getHeight());
    BOOST_CHECK(static_cast<int>(arrayShape[1]) == spnSt.getBBox().getWidth());

    // Loop over each point in the array corresponding to the spanset, minus xy0 (the min of the SpanSet)
    // verify the array contains the value dummpArrayValue
    for (auto const & spn : spnSt) {
        for (auto const & pt : spn) {
            BOOST_CHECK(output[pt.getY()-9][pt.getX()-2] == dummyArrayValue);
        }
    }
    // Verify that points outside the vector have the value zero
    BOOST_CHECK(output[0][2] == 0);
    BOOST_CHECK(output[1][0] == 0);
    BOOST_CHECK(output[2][2] == 0);

    // Test with an array for an output
    int outputSize = 6;
    ndarray::Array<int, 2, 2> unflatOutput = ndarray::allocate(ndarray::makeVector(outputSize, outputSize));
    unflatOutput.deep() = 0;
    afwGeom::Point2I outputXY0(spnSt.getBBox().getMinX(), spnSt.getBBox().getMinY());
    spnSt.unflatten(unflatOutput, input, outputXY0);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            if (spnSt.contains(afwGeom::Point2I(j + outputXY0.getX(), i + outputXY0.getY()))) {
                BOOST_CHECK(unflatOutput[i][j] == dummyArrayValue);
            } else {
                BOOST_CHECK(unflatOutput[i][j] == 0);
            }
        }
    }

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullUnflatArray = nullSpanSet.unflatten(input);
    BOOST_CHECK(nullUnflatArray.size() == 0);
}

std::pair<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, std::shared_ptr<afwGeom::SpanSet>> populateMask() {
    // Create a mask and populate it with the value 2
    lsst::afw::image::Mask<lsst::afw::image::MaskPixel> msk(10, 10);
    msk.getArray().deep() = 1;
    auto spnSt = afwGeom::SpanSet::fromShape(3, afwGeom::Stencil::CIRCLE)->shiftedBy(5, 5);
    spnSt->setMask(msk, static_cast<lsst::afw::image::MaskPixel>(2));
    return std::make_pair(msk, spnSt);
}

BOOST_AUTO_TEST_CASE(SpanSet_testSetMask) {
    auto result = populateMask();
    auto mask = result.first;
    auto spnSt = result.second;
    // Verify the mask was populated with the value 2
    auto mskArray = mask.getArray();
    auto mskShape = mskArray.getShape();
    for (std::size_t i=0; i < mskShape[0]; ++i) {
        for (std::size_t j=0; j<mskShape[1]; ++j){
            if (spnSt->contains(afwGeom::Point2I(i,j))) {
                BOOST_CHECK(mskArray[i][j] == static_cast<lsst::afw::image::MaskPixel>(3));
            } else {
                BOOST_CHECK(mskArray[i][j] == static_cast<lsst::afw::image::MaskPixel>(1));
            }
        }
    }

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    lsst::afw::image::Mask<lsst::afw::image::MaskPixel> nullMask(10, 10);
    nullSpanSet.setMask(nullMask, static_cast<lsst::afw::image::MaskPixel>(2));
    for (auto iter = nullMask.begin(); iter != nullMask.end(); ++iter){
        BOOST_CHECK(*iter == 0);
    }
}

BOOST_AUTO_TEST_CASE(SpanSet_testClearMask) {
    auto result = populateMask();
    auto mask = result.first;
    auto spnSt = result.second;
    spnSt->clearMask(mask, static_cast<lsst::afw::image::MaskPixel>(2));
    // Verify the mask is now zero
    auto mskArray = mask.getArray();
    auto mskShape = mskArray.getShape();
    for (size_t i=0; i < mskShape[0]; ++i) {
        for (size_t j=0; j < mskShape[1]; ++j) {
            BOOST_CHECK(mskArray[i][j] == static_cast<lsst::afw::image::MaskPixel>(1));
        }
    }

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    lsst::afw::image::Mask<lsst::afw::image::MaskPixel> nullMask(10, 10);
    int maskValue = 9;
    for (auto iter = nullMask.begin(); iter != nullMask.end(); ++iter){
        *iter = static_cast<lsst::afw::image::MaskPixel>(maskValue);
    }
    nullSpanSet.clearMask(nullMask, static_cast<lsst::afw::image::MaskPixel>(maskValue));

    for (auto iter = nullMask.begin(); iter != nullMask.end(); ++iter){
        BOOST_CHECK(*iter == static_cast<lsst::afw::image::MaskPixel>(maskValue));
    }
}

std::pair<std::shared_ptr<afwGeom::SpanSet>, std::shared_ptr<afwGeom::SpanSet>> makeOverlapSpanSets() {
    using SS = afwGeom::SpanSet;
    auto firstSpanSet = SS::fromShape(2, afwGeom::Stencil::BOX)->shiftedBy(2, 4);
    auto secondSpanSet = SS::fromShape(2, afwGeom::Stencil::BOX)->shiftedBy(2, 2);
    return std::pair<std::shared_ptr<afwGeom::SpanSet>,
                     std::shared_ptr<afwGeom::SpanSet>>(firstSpanSet, secondSpanSet);
}

std::pair<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, std::shared_ptr<afwGeom::SpanSet>>
 makeMaskAndSpanSetForOperationTests() {
    // Create three overlapping regions in a mask, and a SpanSet to use in set operation tests
    // This box will range from 0 to 4 in y and 0 to 4 in x
    auto firstMaskPart = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::BOX)->shiftedBy(2, 2);
    // This box will range from 6 to 10 in y and 0 to 4 in x
    auto secondMaskPart = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::BOX)->shiftedBy(2, 8);
    // This box will range from 3 to 7 in y and 0 to 4 in x
    auto spanSetMaskOperation = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::BOX)->shiftedBy(2,5);

    lsst::afw::image::Mask<lsst::afw::image::MaskPixel> mask(20,20);
    firstMaskPart->setMask(mask, static_cast<lsst::afw::image::MaskPixel>(3));
    secondMaskPart->setMask(mask, static_cast<lsst::afw::image::MaskPixel>(3));
    // This statement just sets the 4th bit to ensure that it is not included in the following tests
    spanSetMaskOperation->setMask(mask, static_cast<lsst::afw::image::MaskPixel>(4));

    return std::make_pair(mask, spanSetMaskOperation);
}

BOOST_AUTO_TEST_CASE(SpanSet_testIntersect) {
    auto result = makeOverlapSpanSets();
    auto firstSS = result.first;
    auto secondSS = result.second;

    auto overlap = firstSS->intersect(*secondSS);

    // Verify the two boxes overlap in y range 2->4
    int yStart = 2;
    for (auto const & spn : *overlap) {
        BOOST_CHECK(spn.getY() == yStart);
        BOOST_CHECK(spn.getMinX() == 0);
        BOOST_CHECK(spn.getMaxX() == 4);
        ++yStart;
    }
    // Test intersecting with a mask
    auto maskAndSet = makeMaskAndSpanSetForOperationTests();
    auto mask = maskAndSet.first;
    auto spanSetMaskOperation = maskAndSet.second;

    // intersect where the mask has the second bit set;
    auto spanSetIntersectMask = spanSetMaskOperation->intersect(mask,
                                                                static_cast<lsst::afw::image::MaskPixel>(2));
    std::vector<int> expectedYRange{3,4,6,7};
    auto yRangeIter = expectedYRange.begin();
    for (auto const & val : *spanSetIntersectMask ){
        BOOST_CHECK(val.getY() == *yRangeIter);
        ++yRangeIter;
    }

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullAsOther = firstSS->intersect(nullSpanSet);
    BOOST_CHECK(nullAsOther->size() == 0);

    auto spanSetAsOther = nullSpanSet.intersect(*firstSS);
    BOOST_CHECK(spanSetAsOther->size() == 0);
}

BOOST_AUTO_TEST_CASE(SpanSet_testIntersectNot) {
    auto ret = makeOverlapSpanSets();
    auto firstSS = ret.first;
    auto secondSS = ret.second;

    auto overlap = firstSS->intersectNot(*secondSS);

    // Verify the result is in the y range 4->6
    int yStart = 5;
    for (auto const & spn : *overlap) {
        BOOST_CHECK(spn.getY() == yStart);
        BOOST_CHECK(spn.getMinX() == 0);
        BOOST_CHECK(spn.getMaxX() == 4);
        ++yStart;
    }

    // Test intersectingNot with a mask
    auto maskAndSet = makeMaskAndSpanSetForOperationTests();
    auto mask = maskAndSet.first;
    auto spanSetMaskOperation = maskAndSet.second;

    lsst::afw::image::MaskPixel bitmask(2);
    // intersect where the mask has the second bit set
    auto spanSetIntersectNotMask = spanSetMaskOperation->intersectNot(mask, bitmask);

    BOOST_CHECK(spanSetIntersectNotMask->size() == 1);
    BOOST_CHECK((*(spanSetIntersectNotMask->begin())).getY() == 5);

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullAsOther = firstSS->intersectNot(nullSpanSet);
    // A SpanSet intersectedNot with a null SpanSet should just return itself
    BOOST_CHECK(*firstSS == *nullAsOther);

    // A null SpanSet intersectedNot with anything should return a null SpanSet
    auto spanSetAsOther = nullSpanSet.intersectNot(*firstSS);
    BOOST_CHECK(nullSpanSet == *spanSetAsOther);
}

BOOST_AUTO_TEST_CASE(SpanSet_testUnion) {
    auto ret = makeOverlapSpanSets();
    auto firstSS = ret.first;
    auto secondSS = ret.second;

    auto overlap = firstSS->union_(*secondSS);

    // Verify both SpanSets are combined together
    int yStart = 0;
    for (auto const & spn : *overlap) {
        BOOST_CHECK(spn.getY() == yStart);
        BOOST_CHECK(spn.getMinX() == 0);
        BOOST_CHECK(spn.getMaxX() == 4);
        ++yStart;
    }

    // Test union with a mask
    auto maskAndSet = makeMaskAndSpanSetForOperationTests();
    auto mask = maskAndSet.first;
    auto spanSetMaskOperation = maskAndSet.second;

    lsst::afw::image::MaskPixel bitmask(2);
    // union where the mask has the second bit set
    auto spanSetUnion = spanSetMaskOperation->union_(mask, bitmask);

    int yCoord = 0;
    for (auto const & val : *spanSetUnion) {
        BOOST_CHECK(val.getY() == yCoord);
        ++yCoord;
    }

    // Assert that the yCoord ends at the right place (one past the end)
    BOOST_CHECK(yCoord == 11);

    // Test with a null SpanSet
    afwGeom::SpanSet nullSpanSet;
    auto nullAsOther = firstSS->union_(nullSpanSet);
    // A null SpanSet unioned with another SpanSet should return the other SpanSet
    BOOST_CHECK(*nullAsOther == *firstSS);

    auto spanSetAsOther = nullSpanSet.union_(*firstSS);
    BOOST_CHECK(*spanSetAsOther == *firstSS);

}

BOOST_AUTO_TEST_CASE(SpanSet_MaskToSpanSet) {
    // This is to test the free function that turns Masks to SpanSets
    auto maskAndSet = makeMaskAndSpanSetForOperationTests();
    auto mask = maskAndSet.first;
    auto spanSetFromMask = afwGeom::SpanSet::fromMask(mask);

    int yCoord = 0;
    for (auto const & val : *spanSetFromMask) {
        BOOST_CHECK(val == afwGeom::Span(yCoord, 0, 4));
        ++yCoord;
    }

    // Check that the yCoord ends at the right place (one past the end)
    BOOST_CHECK(yCoord == 11);
}

BOOST_AUTO_TEST_CASE(SpanSet_testEquality) {
    auto ret = makeOverlapSpanSets();
    auto firstSS = ret.first;
    auto secondSS = ret.second;
    auto secondSSShift = secondSS->shiftedBy(0, 2);

    BOOST_CHECK((*firstSS == *secondSS) == false);
    BOOST_CHECK((*firstSS != *secondSS) == true);
    BOOST_CHECK((*firstSS == *secondSSShift) == true);
    BOOST_CHECK((*firstSS != *secondSSShift) == false);
}

BOOST_AUTO_TEST_CASE(SpanSet_testFunctor) {
    // Test the remaining types of functors. Above code has tested ndarray functors
    // need to test, constants, iterators, Images
    namespace afwImage = lsst::afw::image;
    int imageDim = 6;
    int initialValue = 0;
    int dataValue = 6;
    afwImage::Image<int> imageObject(imageDim, imageDim, initialValue);
    std::vector<int> vecObject(imageDim*imageDim, initialValue);
    afwImage::Image<int> const constImageObject(imageDim, imageDim, dataValue);
    afwImage::Mask<lsst::afw::image::MaskPixel> const constMaskObject(imageDim, imageDim, dataValue);
    ndarray::Array<int, 2, 2> targetForConstImage = ndarray::allocate(ndarray::makeVector(imageDim,
                                                                                          imageDim));
    targetForConstImage.deep() = initialValue;
    ndarray::Array<int, 2, 2> targetForConstMask = ndarray::allocate(ndarray::makeVector(imageDim,
                                                                                         imageDim));
    targetForConstMask.deep() = initialValue;

    int spanRadius = 2;
    auto SSShape = afwGeom::SpanSet::fromShape(spanRadius,
                                                      afwGeom::Stencil::BOX)->shiftedBy(spanRadius,
                                                                                       spanRadius);
    // use a constant as a test of constantGetter
    SSShape->applyFunctor([](afwGeom::Point2I pt, afwImage::Image<int>::Pixel & out, int in){ out = in; },
                            imageObject, dataValue);
    SSShape->applyFunctor([](afwGeom::Point2I pt, int & out, const afwImage::Image<int>::Pixel & in){ out = in; },
                            ndarray::ndImage(targetForConstImage), constImageObject);
    SSShape->applyFunctor([](afwGeom::Point2I pt, int & out, const lsst::afw::image::MaskPixel & in)
                            { out = static_cast<int>(in); },
                            ndarray::ndImage(targetForConstMask), constMaskObject);
    SSShape->applyFunctor([](afwGeom::Point2I, int & out, int in){out = in;},
                             vecObject.begin(), dataValue);

    // Check the Image values
    auto bounds = SSShape->getBBox();
    for (int i = bounds.getMinY(); i <= bounds.getMaxY(); ++i) {
        for (int j = bounds.getMinX(); j <= bounds.getMaxX(); ++j) {
            if (!SSShape->contains(afwGeom::Point2I(j,i))) {
                BOOST_CHECK(imageObject.getArray()[i][j] == initialValue);
                BOOST_CHECK(targetForConstImage[i][j] == initialValue);
                BOOST_CHECK(targetForConstMask[i][j] == initialValue);
            } else {
                BOOST_CHECK(imageObject.getArray()[i][j] == dataValue);
                BOOST_CHECK(targetForConstImage[i][j] == dataValue);
                BOOST_CHECK(targetForConstMask[i][j] == dataValue);
            }
        }
    }

    // Check the vector values
    for (int i = (2*spanRadius + 1)*(2*spanRadius + 1); i < imageDim*imageDim; ++i) {
        if (i < (2*spanRadius + 1)*(2*spanRadius + 1)) {
            BOOST_CHECK(vecObject[i] == dataValue);
        } else {
            BOOST_CHECK(vecObject[i] == initialValue);
        }
    }

    // Check on the point input of a functor
    std::vector<afwGeom::Point2I> capturedPoints;
    capturedPoints.reserve(SSShape->getArea());
    SSShape->applyFunctor([](afwGeom::Point2I point, afwGeom::Point2I & out){out = point;},
                          capturedPoints.begin());

    auto capturedPointsIter = capturedPoints.begin();
    for (auto const & spn : *SSShape) {
        for (auto const & pnt : spn) {
            BOOST_CHECK(pnt == *capturedPointsIter);
            ++capturedPointsIter;
        }
    }

    // Test null SpanSets
    lsst::afw::image::Image<int> nullImageObject(imageDim, imageDim, initialValue);
    std::vector<int> nullVecObject(imageDim*imageDim, initialValue);
    afwGeom::SpanSet nullSpanSet;

    nullSpanSet.applyFunctor([]
                             (afwGeom::Point2I pt, afwImage::Image<int>::Pixel & out, int in)
                             {out = in;},
                            imageObject, dataValue);
    nullSpanSet.applyFunctor([]
                             (afwGeom::Point2I, int & out, int in)
                             {out = in;},
                             vecObject.begin(), dataValue);

    // nullSpanSet should not have changed any values
    for (auto const & nullImageValue: nullImageObject) {
        BOOST_CHECK(nullImageValue == initialValue);
    }

    for (auto const & nullVecValue : nullVecObject) {
        BOOST_CHECK(nullVecValue == initialValue);
    }
}

BOOST_AUTO_TEST_CASE(SpanSet_testPersistence) {
    namespace tableIo = lsst::afw::table::io;
    // Create a SpanSet to persist
    auto spanSetPreArchive = afwGeom::SpanSet::fromShape(2, afwGeom::Stencil::BOX);
    // Create an output object to save the SpanSet to
    tableIo::OutputArchive outArchive;
    auto id = outArchive.put(spanSetPreArchive);
    // Create a in memory file manager, and save the fits file to that
    lsst::afw::fits::MemFileManager manager;
    lsst::afw::fits::Fits outFits(manager, "w", lsst::afw::fits::Fits::AUTO_CHECK);
    outArchive.writeFits(outFits);
    outFits.closeFile();
    // Read back in the fits file and un-persist the SpanSet
    lsst::afw::fits::Fits inFits(manager, "r", lsst::afw::fits::Fits::AUTO_CHECK);
    inFits.setHdu(INT_MIN);
    lsst::afw::table::io::InputArchive inArchive = tableIo::InputArchive::readFits(inFits);
    inFits.closeFile();
    std::shared_ptr<afwGeom::SpanSet> spanSetPostArchive = std::dynamic_pointer_cast<afwGeom::SpanSet>(inArchive.get(id));

    // Check that the two SpanSets are the same size
    BOOST_CHECK(spanSetPreArchive->size() == spanSetPostArchive->size());

    // Verify the values are the same for the two SpanSets
    auto preArchiveIterator = spanSetPreArchive->begin();
    for (auto const & val : *spanSetPostArchive) {
        BOOST_CHECK(val == *preArchiveIterator);
        ++preArchiveIterator;
    }
}
