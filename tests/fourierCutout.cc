/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
#include <iostream>
#include <cmath>
#include <vector>
#include <exception>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE FourierCutout

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/FourierCutout.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

typedef math::FourierCutout FourierCutout;
typedef math::FourierCutoutStack FourierCutoutStack;

BOOST_AUTO_TEST_CASE(CutoutPixelAccess) {     /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int width = 38;
    int height = 24;
  
    FourierCutout cutout;

    BOOST_REQUIRE_NO_THROW(
        do {
            FourierCutout tmp(width, height);
            cutout = tmp;
        } while(false)
    );   
    
    BOOST_REQUIRE_MESSAGE(cutout.begin() != NULL, "cutout deleted");

    int cutoutSize = (width/2 + 1)*height;
    BOOST_REQUIRE_EQUAL(cutoutSize, cutout.getFourierSize());

    FourierCutout::Real val = 0.2;
    FourierCutout::Complex testComplex(val);
    
    //scalar setter
    cutout = val;

    //test pixel access operator    
    for(int y = 0; y < cutout.getFourierHeight(); ++y) {
        for(int x = 0; x < cutout.getFourierWidth(); ++x) {
            BOOST_CHECK_EQUAL(cutout(x, y), testComplex);
        }
    }

    //test simple pixel iteration
    FourierCutout::const_iterator iter(cutout.begin());
    FourierCutout::const_iterator end(cutout.end());
    int nPix = 0;
    for( ; iter != end; ++iter, ++nPix) {
        BOOST_CHECK_EQUAL((*iter), testComplex);
    }
    BOOST_CHECK_EQUAL(nPix, cutoutSize);

    //test scalar multiplication
    cutout *= 2.5;

    testComplex *= 2.5;

    //test row-wise iterations)
    nPix = 0;
    int nRowPix = 0;
    for (int y = 0; y < cutout.getFourierHeight(); ++y) {
        end =  cutout.row_end(y);
        iter = cutout.row_begin(y);
        for(nRowPix = 0 ; iter != end; iter++, ++nRowPix, ++nPix) {
            BOOST_CHECK_EQUAL(*iter, testComplex);
        }
        BOOST_CHECK_EQUAL(nRowPix,  cutout.getFourierWidth());
                 
    }
    BOOST_CHECK_EQUAL(nPix, cutoutSize);    
}

BOOST_AUTO_TEST_CASE(CutoutOperatorTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int width = 4, height = 4;
    FourierCutout a(width, height);

    FourierCutout::Real val = 2.0;
    //test scalar operators
    BOOST_CHECK_NO_THROW(a = val);
    BOOST_CHECK_EQUAL(a(0,0), val);
    BOOST_CHECK_NO_THROW(a -= val);
    BOOST_CHECK_EQUAL(a(1,1), static_cast<FourierCutout::Real>(0));
    BOOST_CHECK_NO_THROW(a += val);
    BOOST_CHECK_EQUAL(a(2,2), val);

    //test deep assignment
    FourierCutout b(width, height);

    //test FourierCutout pixel-wise operators;
    BOOST_CHECK_NO_THROW(b <<= a);
    BOOST_CHECK_EQUAL(b(0,1), val);
    BOOST_CHECK_NO_THROW(b -= a);
    BOOST_CHECK_EQUAL(b(3,2), static_cast<FourierCutout::Real>(0));
    BOOST_CHECK_NO_THROW(b += a);
    BOOST_CHECK_EQUAL(b(1,3), val);
    BOOST_CHECK_NO_THROW(b *= a);  

    FourierCutout::iterator k(b.begin());
    for(FourierCutout::iterator i = a.begin(); i != a.end(); ++i, ++k) {
        BOOST_CHECK_EQUAL(*i, val);    
        BOOST_CHECK_EQUAL(*k, val*val);
    }

    //test bad dimensions exceptions    
    FourierCutout c = FourierCutout(5, 5);

    BOOST_CHECK_THROW(c *= a, lsst::pex::exceptions::InvalidParameterException);
    BOOST_CHECK_THROW(c <<= a, lsst::pex::exceptions::InvalidParameterException);
    BOOST_CHECK_THROW(c -= a, lsst::pex::exceptions::InvalidParameterException);
    BOOST_CHECK_THROW(c += a, lsst::pex::exceptions::InvalidParameterException);
    
    FourierCutout tmp(a);
    c.swap(tmp);

    BOOST_CHECK_EQUAL(a.getOwner(), c.getOwner());
    BOOST_CHECK_EQUAL(a.getImageWidth(), c.getImageWidth());
    BOOST_CHECK_EQUAL(a.getImageHeight(), c.getImageHeight());
    BOOST_CHECK_EQUAL(a.begin(), c.begin());
    BOOST_CHECK_EQUAL(a.end(), c.end());
    BOOST_CHECK_EQUAL(a.getFourierWidth(), c.getFourierWidth());
}

BOOST_AUTO_TEST_CASE(CutoutStackTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int width = 8;
    int height = 7;
    int depth = 3;
    FourierCutoutStack stack(width, height, depth);

    BOOST_CHECK_EQUAL(stack.getStackDepth(), depth);

    int cutoutSize = (width/ 2 +1 )*height;
    BOOST_CHECK_EQUAL(stack.getCutoutSize(), cutoutSize);

    FourierCutout::Ptr cutoutPtr;

    boost::shared_ptr<FourierCutout::Complex> dataPtr = stack.getData();
    FourierCutout::Complex * data = dataPtr.get();
    
    for(int i = 0; i < depth; ++i, data += cutoutSize) {
        cutoutPtr = stack.getCutout(i);
        *cutoutPtr = static_cast<FourierCutout::Real>(i);
        BOOST_CHECK_EQUAL(cutoutPtr->getFourierSize(), stack.getCutoutSize());
        BOOST_CHECK_EQUAL(cutoutPtr->begin(), data);
    }

    int begin = 1;
    FourierCutoutStack::FourierCutoutVector vector = stack.getCutoutVector(begin);
   
    int vectorLength = static_cast<int>(vector.size()); 
    BOOST_CHECK_EQUAL(vectorLength, stack.getStackDepth() - begin);
     
    for(int i = 0; i < vectorLength; ++i) {
        cutoutPtr = vector[i];
        BOOST_CHECK_EQUAL(dataPtr, cutoutPtr->getOwner()); 
        FourierCutout::iterator j = cutoutPtr->begin();
        FourierCutout::iterator end = cutoutPtr->end();

        for( ; j != end; ++j) {
            BOOST_CHECK_EQUAL((*j), static_cast<FourierCutout::Real>(i + begin));            
        }    
    }
    

    FourierCutoutStack::Ptr stackPtr = boost::make_shared<FourierCutoutStack>(stack);
    FourierCutoutStack swapStack(4, 3, 5);
    swapStack.swap(*stackPtr);

    stackPtr.reset();

    BOOST_CHECK_EQUAL(swapStack.getData(), stack.getData());
    BOOST_CHECK_EQUAL(swapStack.getCutoutSize(), stack.getCutoutSize());
    BOOST_CHECK_EQUAL(swapStack.getStackDepth(), stack.getStackDepth());
}
