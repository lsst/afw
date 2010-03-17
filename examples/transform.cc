// -*- LSST-C++ -*-
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
    cout << "default LinearTransform parameters: " << def.getVector() << endl;
    
    //Copy construct a scaling transform
    afwGeom::LinearTransform s = afwGeom::LinearTransform::makeScaling(1.5);
    cout << "scaling LinearTransform matrix: "<< s.getMatrix() << endl;
    cout << "scaling LinearTransform parameters: " << s.getVector() << endl;

    //copy construt a rotation transform
    afwGeom::LinearTransform r = afwGeom::LinearTransform::makeRotation(1.0);
    cout << "rotation LinearTransform matrix: "<< r.getMatrix() << endl;
    cout << "rotation LinearTransform parameters: " << r.getVector() << endl;
   
    //concactenate the scaling and rotation transform
    afwGeom::LinearTransform c = s*r;
    cout << "rotation+scaling LinearTransform matrix: "<< c.getMatrix() << endl;
    cout << "rotation+scaling LinearTransform parameters: " << c.getVector() << endl;
    
    //create a point, and duplicate it as an extent
    afwGeom::Point2D point = afwGeom::makePointD(3.0, 4.5);
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
    cout << "default AffineTransform parameters: " << def.getVector() << endl;
    
    //Copy construct a scaling transform
    afwGeom::AffineTransform s = afwGeom::AffineTransform::makeScaling(1.5);
    cout << "scaling AffineTransform matrix: "<< s.getMatrix() << endl;
    cout << "scaling AffineTransform parameters: " << s.getVector() << endl;

    //copy construt a rotation transform
    afwGeom::AffineTransform r = afwGeom::AffineTransform::makeRotation(1.0);
    cout << "rotation AffineTransform matrix: "<< r.getMatrix() << endl;
    cout << "rotation AffineTransform parameters: " << r.getVector() << endl;
  
    //copy construct a translation transform
    afwGeom::AffineTransform t = afwGeom::AffineTransform::makeTranslation(afwGeom::makeExtentD(15.0, 10.3));
    cout << "translation AffineTransform matrix: "<< t.getMatrix() << endl;
    cout << "translation AffineTransform parameters: " << t.getVector() << endl;

    //concactenate the scaling and rotation transform
    afwGeom::AffineTransform c = s*r*t;
    cout << "translation+rotation+scaling AffineTransform matrix: "<< c.getMatrix() << endl;
    cout << "translation+rotation+scaling AffineTransform parameters: " << c.getVector() << endl;
    
    //We can grab just the Linear part of the AffineTransform
    cout << "linear part of affine: " << c.getLinear() <<endl;

    //or we cna grab the translation
    cout << "translation part of affine: " << c.getTranslation() << endl;


    //create a point, and duplicate it as an extent
    afwGeom::Point2D point = afwGeom::makePointD(3.0, 4.5);
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

void wcsExample() {
    //initialize a trivial WCS for examples
    lsst::afw::image::Wcs wcs(
        lsst::afw::image::PointD(35, 45), 
        lsst::afw::image::PointD(0.0,0.0),
        Eigen::Matrix2d::Identity()
    );
    afwGeom::PointD point = afwGeom::makePointD(35, 45);

    //We can obtain the linear approximation of a WCS at a point as an
    //AffineTransform
    afwGeom::AffineTransform approx = wcs.linearizeAt(point);
    cout << "Linear Approximation of Wcs object at point "<<point<< ": " << approx << endl;    
    
    //alternatively we can obtain the CD matrix of the wcs as a LinearTransform
    cout << "CD matrix of Wcs object: " << wcs.getLinearTransform() << endl;
    

}
int main() {
    linearTransformExample();
    affineTransformExample();
    wcsExample();

    return 0;
}
