// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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

/**
 * @file
 *
 * @brief Example for usage of AffineTransform and LinearTransform classes
 *
 * @author Martin Dubcovsky
 * @date 3.16.2010
 */

#include <iostream>
#include <Eigen/Core>

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/LinearTransform.h"
#include "lsst/afw/geom/AffineTransform.h"

#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Utils.h"

using namespace std;
namespace afwGeom = lsst::afw::geom;

void linearTransformExample(){
    //Default Construction
    afwGeom::LinearTransform def;

    //inspect the transform
    cout << "default LinearTransform matrix: " << def.getMatrix() <<endl;
    cout << "default LinearTransform parameters: " << def.getParameterVector() << endl;

    //Copy construct a scaling transform
    afwGeom::LinearTransform s = afwGeom::LinearTransform::makeScaling(1.5);
    cout << "scaling LinearTransform matrix: "<< s.getMatrix() << endl;
    cout << "scaling LinearTransform parameters: " << s.getParameterVector() << endl;

    //copy construt a rotation transform
    afwGeom::LinearTransform r = afwGeom::LinearTransform::makeRotation(1.0*afwGeom::radians);
    cout << "rotation LinearTransform matrix: "<< r.getMatrix() << endl;
    cout << "rotation LinearTransform parameters: " << r.getParameterVector() << endl;

    //concactenate the scaling and rotation transform
    afwGeom::LinearTransform c = s*r;
    cout << "rotation+scaling LinearTransform matrix: "<< c.getMatrix() << endl;
    cout << "rotation+scaling LinearTransform parameters: " << c.getParameterVector() << endl;

    //create a point, and duplicate it as an extent
    afwGeom::Point2D point = afwGeom::Point2D(3.0, 4.5);
    afwGeom::Extent2D extent(point);

    //apply the LinearTransforms to points and extents. Because there is no
    //translataion component to a LinearTransform, this operation is equivalent
    //on points and extents
    cout << "original point: " << point << "\tTransformed point: "<< c(point) << endl;
    cout << "original extent: " << extent << "\tTransformed extent: "<< c(extent) << endl;

    //The Affine transform can also compute the derivative of the transformation
    //with respect to the transform parameters
    cout << "Transformation derivative: " << c.dTransform(point) << endl;
}

void affineTransformExample() {
    //Default Construction
    afwGeom::AffineTransform def;

    //inspect the transform
    cout << "default AffineTransform matrix: " << def.getMatrix() <<endl;
    cout << "default AffineTransform parameters: " << def.getParameterVector() << endl;

    //Copy construct a scaling transform
    afwGeom::AffineTransform s = afwGeom::AffineTransform::makeScaling(1.5);
    cout << "scaling AffineTransform matrix: "<< s.getMatrix() << endl;
    cout << "scaling AffineTransform parameters: " << s.getParameterVector() << endl;

    //copy construt a rotation transform
    afwGeom::AffineTransform r = afwGeom::AffineTransform::makeRotation(1.0*afwGeom::radians);
    cout << "rotation AffineTransform matrix: "<< r.getMatrix() << endl;
    cout << "rotation AffineTransform parameters: " << r.getParameterVector() << endl;

    //copy construct a translation transform
    afwGeom::AffineTransform t = afwGeom::AffineTransform::makeTranslation(afwGeom::Extent2D(15.0, 10.3));
    cout << "translation AffineTransform matrix: "<< t.getMatrix() << endl;
    cout << "translation AffineTransform parameters: " << t.getParameterVector() << endl;

    //concactenate the scaling and rotation transform
    afwGeom::AffineTransform c = s*r*t;
    cout << "translation+rotation+scaling AffineTransform matrix: "<< c.getMatrix() << endl;
    cout << "translation+rotation+scaling AffineTransform parameters: " << c.getParameterVector() << endl;

    //We can grab just the Linear part of the AffineTransform
    cout << "linear part of affine: " << c.getLinear() <<endl;

    //or we cna grab the translation
    cout << "translation part of affine: " << c.getTranslation() << endl;


    //create a point, and duplicate it as an extent
    afwGeom::Point2D point = afwGeom::Point2D(3.0, 4.5);
    afwGeom::Extent2D extent(point);

    //apply the LinearTransforms to points and extents. Because there a
    //translataion component to a AffineTransform, this operation is not
    //equivalent on points and extents
    cout << "original point: " << point << "\tTransformed point: "<< c(point) << endl;
    cout << "original extent: " << extent << "\tTransformed extent: "<< c(extent) << endl;

    //The Affine transform can also compute the derivative of the transformation
    //with respect to the transform parameters
    cout << "point transformation derivative: " << c.dTransform(point) << endl;
    cout << "extent transformation derivative: " << c.dTransform(extent) << endl;




}

int main() {
    linearTransformExample();
    affineTransformExample();

    return 0;
}
