// -*- LSST-C++ -*-
/*!
 * \brief Implementation of ImageLocalPsf
 */
#include <limits>
#include <typeinfo>
#include <cmath>
#include "lsst/afw/detection/LocalPsf.h"
#include "lsst/afw/detection/FootprintArray.cc"
#include "lsst/ndarray/eigen.h"
#include "lsst/afw/geom/ellipses.h"
#include <Eigen/Cholesky>

/************************************************************************************************************/

namespace pexExcept = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;
namespace afwMath = lsst::afw::math;

#if 0

namespace lsst {
namespace afw {
namespace detection {

CONST_PTR(LocalPsf::Shapelet)
ImageLocalPsf::asShapelet(math::shapelets::BasisTypeEnum basisType) const {
    ndarray::Array<Pixel,2,2> matrix = ndarray::allocate(
        _image.getWidth() * _image.getHeight(),
        math::shapelets::computeSize(_order)
    );
    math::shapelets::BasisEvaluator b(_order, basisType);
    geom::ellipses::Ellipse e(
        this->asGaussian(), 
        geom::Point2D(0.5 * (_image.getWidth() - 1), 0.5 * (_image.getHeight() - 1))
    );
    geom::AffineTransform g = e.getGridTransform();
    ndarray::Array<Pixel,2,2> array2 = ndarray::copy(_image.getArray());
    ndarray::Array<Pixel,1,1> array1 = ndarray::flatten<1>(array2);
    for (int y = 0; y < _image.getHeight(); ++y) {
        for (int x = 0; x < _image.getWidth(); ++x) {
            b.fillEvaluation(
                matrix[x + y * _image.getWidth()],
                g(geom::Point2D(x, y))
            );
        }
    }
    Eigen::MatrixXd h(matrix.getSize<1>(), matrix.getSize<1>());
    h.part<Eigen::SelfAdjoint>() = ndarray::viewAsTransposedEigen(matrix) * ndarray::viewAsEigen(matrix);
    Eigen::VectorXd rhs = ndarray::viewAsTransposedEigen(matrix) * ndarray::viewAsEigen(array1);
    Eigen::LDLT<Eigen::MatrixXd> cholesky(h);
    ndarray::Array<Pixel,1,1> shapeletArray = ndarray::allocate(matrix.getSize<1>());
    ndarray::EigenView<Pixel,1,1> shapeletVector(shapeletArray);
    if (!cholesky.solve(rhs, &shapeletVector)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::RuntimeErrorException,
            "Singular matrix encounter converting LocalPsf to shapelets."
        );
    }
    Shapelet::Ptr shapelet = boost::make_shared<Shapelet>(_order, basisType, e.getCore(), shapeletArray);
    shapelet->getCoefficients().deep() /= shapelet->evaluate().integrate();
    return shapelet;
}

CONST_PTR(LocalPsf::MultiShapelet)
ImageLocalPsf::asMultiShapelet(math::shapelets::BasisTypeEnum basisType) const {
    Shapelet::ConstPtr shapelet = asShapelet(basisType);
    std::list<Shapelet> elements;
    elements.push_back(*shapelet);
    return boost::make_shared<MultiShapelet>(elements);
}

void ImageLocalPsf::evaluatePointSource(
    Footprint const & fp, 
    geom::Point2D const & point, 
    ndarray::Array<Pixel, 1, 0> const & array
) const {
    if (fp.getArea() != array.getSize<0>()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Footprint area does not match 1d array size."
        );
    }
    geom::Point2D imageCenter(0.5 * (_image.getWidth() - 1), 0.5 * (_image.getHeight() - 1));
    geom::Extent2D shift = point - imageCenter;
    Image::Ptr shiftedImage = math::offsetImage(_image, shift.getX(), shift.getY());
    geom::Box2I bbox = shiftedImage->getBBox(image::PARENT);
    if (!bbox.contains(fp.getBBox())) {
        Image tmp(fp.getBBox());
        Image sub(tmp, bbox, image::PARENT, false);
        sub <<= *shiftedImage;
        shiftedImage->swap(tmp);
        bbox = fp.getBBox();
    }
    detection::flattenArray(fp, shiftedImage->getArray(), array, shiftedImage->getXY0());
}

}}}

#endif
