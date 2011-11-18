// -*- LSST-C++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010m 2011 LSST Corporation.
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
/*!
 * \brief Implementation of ShapeletLocalPsf
 */
#include "lsst/afw/detection/LocalPsf.h"
#include "lsst/afw/detection/FootprintArray.cc"
#include "lsst/ndarray/eigen.h"
#include "Eigen/SVD"

/************************************************************************************************************/

namespace afwDet = lsst::afw::detection;
namespace afwGeom = lsst::afw::geom;

void afwDet::ShapeletLocalPsf::evaluatePointSource(
    Footprint const & fp, 
    ndarray::Array<Pixel,1,0> const & array,
    geom::Extent2D const & offset
) const {
    MultiShapelet::Evaluator evaluator = _shapelet.evaluate();
    ndarray::Array<Pixel,1,0>::Iterator pixIter = array.begin();
    for (
        Footprint::SpanList::const_iterator spanIter = fp.getSpans().begin();
        spanIter != fp.getSpans().end();
        ++spanIter
    ) {
        Span const & span = **spanIter;
        for (int x = span.getX0(); x <= span.getX1(); ++x, ++pixIter) {
            *pixIter = evaluator(x - offset.getX(), span.getY() - offset.getY());
        }
    }
}

afwDet::LocalPsf::Shapelet afwDet::ImageLocalPsf::computeShapelet(
    math::shapelets::BasisTypeEnum basisType, 
    int order,
    geom::ellipses::Ellipse const & ellipse
) const {
    ndarray::Array<Pixel,2,2> matrix = ndarray::allocate(
        _image.getWidth() * _image.getHeight(),
        math::shapelets::computeSize(order)
    );
    ndarray::Array<Pixel,1,1> array = ndarray::flatten<1>(
        ndarray::copy(_image.getArray()).shallow()
    );
    math::shapelets::BasisEvaluator b(order, basisType);
    geom::AffineTransform g = ellipse.getGridTransform();
    for (int y = 0; y < _image.getHeight(); ++y) {
        for (int x = 0; x < _image.getWidth(); ++x) {
            b.fillEvaluation(
                matrix[x + y * _image.getWidth()],
                g(geom::Point2D(x + _image.getX0(), y + _image.getY0()))
            );
        }
    }
    Eigen::JacobiSVD< ndarray::EigenView<Pixel,2,2>::PlainEigenType > solver(
        matrix.asEigen(), Eigen::ComputeThinU | Eigen::ComputeThinV
    );
    ndarray::Array<Pixel,1,1> shapeletArray = ndarray::allocate(matrix.getSize<1>());
    shapeletArray.asEigen() = solver.solve(array.asEigen());
    Shapelet result(order, basisType, ellipse, shapeletArray);
    result.getCoefficients().deep() /= result.evaluate().integrate();
    return result;
}

afwGeom::ellipses::Ellipse afwDet::ImageLocalPsf::computeMoments() const {
    // Note that the constructor guarantees that the sum of pixels is one, so we can assume that here.
    double ix=0.0, iy=0.0, ixx=0.0, iyy=0.0, ixy=0.0;
    Image::ConstArray array = _image.getArray();
    double y = _image.getY0() - getPoint().getY();
    for (Image::ConstArray::Iterator rowIter = array.begin(); rowIter != array.end(); ++y, ++rowIter) {
        double x = _image.getX0() - getPoint().getX();
        for (
            Image::ConstArray::Reference::Iterator pixIter = rowIter->begin(); 
            pixIter != rowIter->end();
            ++x, ++pixIter
        ) {
            ix += x * (*pixIter);
            iy += y * (*pixIter);
            ixx += x * x * (*pixIter);
            iyy += y * y * (*pixIter);
            ixy += x * y * (*pixIter);
        }
    }
    ixx -= ix * ix;
    iyy -= iy * iy;
    ixy -= ix * iy;
    if (ixx < 0.0 || iyy < 0.0 || ixx*iyy < ixy*ixy) {
	throw LSST_EXCEPT(
	    lsst::pex::exceptions::RuntimeErrorException,
	    "PSF Quadrupole moments do not define a valid ellipse"
	);
    }
    return geom::ellipses::Ellipse(
        geom::ellipses::Quadrupole(ixx, iyy, ixy),
        getPoint() + geom::Extent2D(ix, iy)
    );
}

void afwDet::ImageLocalPsf::evaluatePointSource(
    Footprint const & fp, 
    ndarray::Array<Pixel, 1, 0> const & array,
    geom::Extent2D const & offset
) const {
    if (fp.getArea() != array.getSize<0>()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Footprint area does not match 1d array size."
        );
    }
    image::Image<Pixel>::Ptr shiftedImage = math::offsetImage(_image, offset.getX(), offset.getY());
    geom::Box2I bbox = shiftedImage->getBBox(image::PARENT);
    if (!bbox.contains(fp.getBBox())) {
        geom::Box2I tmpBBox = fp.getBBox();
        tmpBBox.include(bbox);
        image::Image<Pixel> tmp(tmpBBox);
        image::Image<Pixel> sub(tmp, bbox, image::PARENT, false);
        sub <<= *shiftedImage;
        shiftedImage->swap(tmp);
    }
    detection::flattenArray(fp, shiftedImage->getArray(), array, shiftedImage->getXY0());
}
