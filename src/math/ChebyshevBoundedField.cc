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

#include "boost/make_shared.hpp"

#include "ndarray/eigen.h"
#include "lsst/afw/math/LeastSquares.h"
#include "lsst/afw/math/ChebyshevBoundedField.h"
#include "lsst/afw/math/detail/TrapezoidalPacker.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst { namespace afw { namespace math {

int ChebyshevBoundedFieldControl::computeSize() const {
    return detail::TrapezoidalPacker(*this).size;
}

// ------------------ Constructors and helpers ---------------------------------------------------------------

namespace {

// Compute an affine transform that maps an arbitrary box to [-1,1]x[-1,1]
geom::AffineTransform makeChebyshevRangeTransform(afw::geom::Box2D const bbox) {
    return geom::AffineTransform(
        geom::LinearTransform::makeScaling(2.0 / bbox.getWidth(), 2.0 / bbox.getHeight()),
        geom::Extent2D(
            -(2.0*bbox.getCenterX()) / bbox.getWidth(),
            -(2.0*bbox.getCenterY()) / bbox.getHeight()
        )
    );
}

} // anonyomous

ChebyshevBoundedField::ChebyshevBoundedField(
    afw::geom::Box2I const & bbox,
    ndarray::Array<double const,2,2> const & coefficients
) : BoundedField(bbox),
    _toChebyshevRange(makeChebyshevRangeTransform(geom::Box2D(bbox))),
    _coefficients(coefficients)
{}

ChebyshevBoundedField::ChebyshevBoundedField(
    afw::geom::Box2I const & bbox
) : BoundedField(bbox),
    _toChebyshevRange(makeChebyshevRangeTransform(geom::Box2D(bbox)))
{}

// ------------------ fit() and helpers ---------------------------------------------------------------------

namespace {

typedef ChebyshevBoundedField::Control Control;
typedef detail::TrapezoidalPacker Packer;

// fill an array with 1-d Chebyshev functions of the 1st kind T(x), evaluated at the given point x
void evaluateBasis1d(
    ndarray::Array<double,1,1> const & t,
    double x
) {
    int const n = t.getSize<0>();
    if (n > 0) {
        t[0] = 1.0;
    }
    if (n > 1) {
        t[1] = x;
    }
    for (int i = 2; i < n; ++i) {
        t[i] = 2.0*x*t[i-1] - t[i-2];
    }
}

// Create a matrix of 2-d Chebyshev functions evaluated at a set of positions, with
// Chebyshev order along columns and evaluation positions along rows.  We pack the
// 2-d functions using the TrapezoidalPacker class, because we don't want any columns
// that correspond to coefficients that should be set to zero.
ndarray::Array<double,2,2> makeMatrix(
    ndarray::Array<double const,1> const & x,
    ndarray::Array<double const,1> const & y,
    geom::AffineTransform const & toChebyshevRange,
    Packer const & packer,
    Control const & ctrl
) {
    int const nPoints = x.getSize<0>();
    ndarray::Array<double,1,1> tx = ndarray::allocate(packer.nx);
    ndarray::Array<double,1,1> ty = ndarray::allocate(packer.ny);
    ndarray::Array<double,2,2> out = ndarray::allocate(nPoints, packer.size);
    // Loop over x and y together, computing T_i(x) and T_j(y) arrays for each point,
    // then packing them together.
    for (int p = 0; p < nPoints; ++p) {
        geom::Point2D sxy = toChebyshevRange(geom::Point2D(x[p], y[p]));
        evaluateBasis1d(tx, sxy.getX());
        evaluateBasis1d(ty, sxy.getY());
        packer.pack(out[p], tx, ty); // this sets a row of out to the packed outer product of tx and ty
    }
    return out;
}

// Create a matrix of 2-d Chebyshev functions evaluated on a grid of positions, with
// Chebyshev order along columns and evaluation positions along rows.  We pack the
// 2-d functions using the TrapezoidalPacker class, because we don't want any columns
// that correspond to coefficients that should be set to zero.
ndarray::Array<double,2,2> makeMatrix(
    geom::Box2I const & bbox,
    geom::AffineTransform const & toChebyshevRange,
    Packer const & packer,
    Control const & ctrl
) {
    // Create a 2-d array that contains T_j(x) for each x value, with x values in rows and j in columns
    ndarray::Array<double,2,2> tx = ndarray::allocate(bbox.getWidth(), packer.nx);
    for (int x = bbox.getBeginX(), p = 0; p < bbox.getWidth(); ++p, ++x) {
        evaluateBasis1d(
            tx[p],
            toChebyshevRange[geom::AffineTransform::XX]*x + toChebyshevRange[geom::AffineTransform::X]
        );
    }

    // Loop over y values, and at each point, compute T_i(y), then loop over x and multiply by the T_j(x)
    // we already computed and stored above.
    ndarray::Array<double,2,2> out = ndarray::allocate(bbox.getArea(),packer.size);
    ndarray::Array<double,2,2>::Iterator outIter = out.begin();
    ndarray::Array<double,1,1> ty = ndarray::allocate(ctrl.orderY + 1);
    for (int y = bbox.getBeginY(), i = 0; i < bbox.getHeight(); ++i, ++y) {
        evaluateBasis1d(
            ty,
            toChebyshevRange[geom::AffineTransform::YY]*y + toChebyshevRange[geom::AffineTransform::Y]
        );
        for (int j = 0; j < bbox.getWidth(); ++j, ++outIter) {
            // this sets a row of out to the packed outer product of tx and ty
            packer.pack(*outIter, tx[j], ty);
        }
    }
    return out;
}

} // anonymous

PTR(ChebyshevBoundedField) ChebyshevBoundedField::fit(
    afw::geom::Box2I const & bbox,
    ndarray::Array<double const,1> const & x,
    ndarray::Array<double const,1> const & y,
    ndarray::Array<double const,1> const & z,
    Control const & ctrl
) {
    // Initialize the result object, so we can make use of the AffineTransform it builds
    PTR(ChebyshevBoundedField) result(new ChebyshevBoundedField(bbox));
    // This packer object knows how to map the 2-d Chebyshev functions onto a 1-d array,
    // using only those that the control says should have nonzero coefficients.
    Packer const packer(ctrl);
    // Create a "design matrix" for the linear least squares problem (A in min||Ax-b||)
    ndarray::Array<double,2,2> matrix = makeMatrix(x, y, result->_toChebyshevRange, packer, ctrl);
    // Solve the linear least squares problem.
    LeastSquares lstsq = LeastSquares::fromDesignMatrix(matrix, z, LeastSquares::NORMAL_EIGENSYSTEM);
    // Unpack the solution into a 2-d matrix, with zeros for values we didn't fit.
    result->_coefficients = packer.unpack(lstsq.getSolution());
    return result;
}

PTR(ChebyshevBoundedField) ChebyshevBoundedField::fit(
    afw::geom::Box2I const & bbox,
    ndarray::Array<double const,1> const & x,
    ndarray::Array<double const,1> const & y,
    ndarray::Array<double const,1> const & z,
    ndarray::Array<double const,1> const & w,
    Control const & ctrl
) {
    // Initialize the result object, so we can make use of the AffineTransform it builds
    PTR(ChebyshevBoundedField) result(new ChebyshevBoundedField(bbox));
    // This packer object knows how to map the 2-d Chebyshev functions onto a 1-d array,
    // using only those that the control says should have nonzero coefficients.
    Packer const packer(ctrl);
    // Create a "design matrix" for the linear least squares problem ('A' in min||Ax-b||)
    ndarray::Array<double,2,2> matrix = makeMatrix(x, y, result->_toChebyshevRange, packer, ctrl);
    // We want to do weighted least squares, so we multiply both the data vector 'b' and the
    // matrix 'A' by the weights.
    matrix.asEigen<Eigen::ArrayXpr>().colwise() *= w.asEigen<Eigen::ArrayXpr>();
    ndarray::Array<double,1,1> wz = ndarray::copy(z);
    wz.asEigen<Eigen::ArrayXpr>() *= w.asEigen<Eigen::ArrayXpr>();
    // Solve the linear least squares problem.
    LeastSquares lstsq = LeastSquares::fromDesignMatrix(matrix, wz, LeastSquares::NORMAL_EIGENSYSTEM);
    // Unpack the solution into a 2-d matrix, with zeros for values we didn't fit.
    result->_coefficients = packer.unpack(lstsq.getSolution());
    return result;
}

template <typename T>
PTR(ChebyshevBoundedField) ChebyshevBoundedField::fit(
    image::Image<T> const & img,
    Control const & ctrl
) {
    // Initialize the result object, so we can make use of the AffineTransform it builds
    geom::Box2I bbox = img.getBBox(image::PARENT);
    PTR(ChebyshevBoundedField) result(new ChebyshevBoundedField(bbox));
    // This packer object knows how to map the 2-d Chebyshev functions onto a 1-d array,
    // using only those that the control says should have nonzero coefficients.
    Packer const packer(ctrl);
    ndarray::Array<double,2,2> matrix = makeMatrix(bbox, result->_toChebyshevRange, packer, ctrl);
    // Flatten the data image into a 1-d vector.
    ndarray::Array<double,2,2> imgCopy = ndarray::allocate(img.getArray().getShape());
    imgCopy.deep() = img.getArray();
    ndarray::Array<double const,1,1> z = ndarray::flatten<1>(imgCopy);
    // Solve the linear least squares problem.
    LeastSquares lstsq = LeastSquares::fromDesignMatrix(matrix, z, LeastSquares::NORMAL_EIGENSYSTEM);
    // Unpack the solution into a 2-d matrix, with zeros for values we didn't fit.
    result->_coefficients = packer.unpack(lstsq.getSolution());
    return result;
}

// ------------------ modifier factories ---------------------------------------------------------------

PTR(ChebyshevBoundedField) ChebyshevBoundedField::truncate(Control const & ctrl) const {
    if (ctrl.orderX >= _coefficients.getSize<1>()) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("New x order (%d) exceeds old x order (%d)")
             % ctrl.orderX % (_coefficients.getSize<1>() - 1)).str()
        );
    }
    if (ctrl.orderY >= _coefficients.getSize<0>()) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("New y order (%d) exceeds old y order (%d)")
             % ctrl.orderY % (_coefficients.getSize<0>() - 1)).str()
        );
    }
    ndarray::Array<double,2,2> coefficients = ndarray::allocate(ctrl.orderY + 1, ctrl.orderX + 1);
    coefficients.deep()
        = _coefficients[ndarray::view(0, ctrl.orderY + 1)(0, ctrl.orderX + 1)];
    if (ctrl.triangular) {
        Packer packer(ctrl);
        ndarray::Array<double,1,1> packed = ndarray::allocate(packer.size);
        packer.pack(packed, coefficients);
        packer.unpack(coefficients, packed);
    }
    return boost::make_shared<ChebyshevBoundedField>(getBBox(), coefficients);
}

PTR(ChebyshevBoundedField) ChebyshevBoundedField::relocate(geom::Box2I const & bbox) const {
    return boost::make_shared<ChebyshevBoundedField>(bbox, _coefficients);
}


// ------------------ evaluate() and helpers ---------------------------------------------------------------

namespace {

// To evaluate a 1-d Chebyshev function without needing to have workspace, we use the
// Clenshaw algorith, which is like going through the recurrence relation in reverse.
// The CoeffGetter argument g is something that behaves like an array, providing access
// to the coefficients.
template <typename CoeffGetter>
double evaluateFunction1d(
    CoeffGetter g,
    double x,
    int size
) {
    double b_kp2=0.0, b_kp1=0.0;
    for (int k = (size - 1); k > 0; --k) {
        double b_k = g[k] + 2*x*b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }
    return g[0] + x*b_kp1 - b_kp2;
}

// This class imitates a 1-d array, by running evaluateFunction1d on a nested dimension;
// this lets us reuse the logic in evaluateFunction1d for both dimensions.  Essentially,
// we run evaluateFunction1d on a column of coefficients to evaluate T_j(x), then pass
// the result of that to evaluateFunction1d with the results as the "coefficients" associated
// with the T_j(y) functions.
struct RecursionArrayImitator {

    double operator[](int i) const {
        return evaluateFunction1d(coefficients[i], x, coefficients.getSize<1>());
    }

    RecursionArrayImitator(ndarray::Array<double const,2,2> const & coefficients_, double x_) :
        coefficients(coefficients_), x(x_)
    {}

    ndarray::Array<double const,2,2> coefficients;
    double x;
};

} // anonymous

double ChebyshevBoundedField::evaluate(geom::Point2D const & position) const {
    geom::Point2D p = _toChebyshevRange(position);
    return evaluateFunction1d(RecursionArrayImitator(_coefficients, p.getX()),
                              p.getY(), _coefficients.getSize<0>());
}

// ------------------ persistence ---------------------------------------------------------------------------

namespace {

struct PersistenceHelper {
    table::Schema schema;
    table::Key<int> orderX;
    table::Key< table::Point<int> > bboxMin;
    table::Key< table::Point<int> > bboxMax;
    table::Key< table::Array<double> > coefficients;

    PersistenceHelper(int nx, int ny) :
        schema(),
        orderX(schema.addField<int>("order.x", "maximum Chebyshev function order in x")),
        bboxMin(schema.addField< table::Point<int> >("bbox.min", "lower-left corner of bounding box")),
        bboxMax(schema.addField< table::Point<int> >("bbox.max", "upper-right corner of bounding box")),
        coefficients(
            schema.addField< table::Array<double> >(
                "coefficients", "Chebyshev function coefficients, ordered by y then x", nx*ny
            )
        )
     {}

    PersistenceHelper(table::Schema const & s) :
        schema(s),
        orderX(s["order.x"]),
        bboxMin(s["bbox.min"]),
        bboxMax(s["bbox.max"]),
        coefficients(s["coefficients"])
    {}

};

class ChebyshevBoundedFieldFactory : public table::io::PersistableFactory {
public:

    virtual PTR(table::io::Persistable)
    read(InputArchive const & archive, CatalogVector const & catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        table::BaseRecord const & record = catalogs.front().front();
        PersistenceHelper const keys(record.getSchema());
        geom::Box2I bbox(record.get(keys.bboxMin), record.get(keys.bboxMax));
        int nx = record.get(keys.orderX) + 1;
        int ny = keys.coefficients.getSize() / nx;
        LSST_ARCHIVE_ASSERT(nx * ny == keys.coefficients.getSize());
        ndarray::Array<double,2,2> coefficients = ndarray::allocate(ny, nx);
        ndarray::flatten<1>(coefficients) = record.get(keys.coefficients);
        return boost::make_shared<ChebyshevBoundedField>(bbox, coefficients);
    }

    ChebyshevBoundedFieldFactory(std::string const & name) : afw::table::io::PersistableFactory(name) {}

};

std::string getChebyshevBoundedFieldPersistenceName() { return "ChebyshevBoundedField"; }

ChebyshevBoundedFieldFactory registration(getChebyshevBoundedFieldPersistenceName());

} // anonymous

std::string ChebyshevBoundedField::getPersistenceName() const {
    return getChebyshevBoundedFieldPersistenceName();
}

std::string ChebyshevBoundedField::getPythonModule() const {
    return "lsst.afw.math";
}

void ChebyshevBoundedField::write(OutputArchiveHandle & handle) const {
    PersistenceHelper const keys(_coefficients.getSize<1>(), _coefficients.getSize<0>());
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    PTR(table::BaseRecord) record = catalog.addNew();
    record->set(keys.orderX, _coefficients.getSize<1>() - 1);
    record->set(keys.bboxMin, getBBox().getMin());
    record->set(keys.bboxMax, getBBox().getMax());
    (*record)[keys.coefficients].deep() = ndarray::flatten<1>(_coefficients);
    handle.saveCatalog(catalog);
}

// ------------------ explicit instantiation ----------------------------------------------------------------

#ifndef DOXYGEN

#define INSTANTIATE(T)                                                 \
    template PTR(ChebyshevBoundedField) ChebyshevBoundedField::fit(    \
        image::Image<T> const & image,                                 \
        Control const & ctrl                                           \
    )

INSTANTIATE(float);
INSTANTIATE(double);

#endif

}}} // namespace lsst::afw::math
