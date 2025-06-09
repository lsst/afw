// -*- lsst-c++ -*-
/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <vector>

#include "Eigen/QR"
#include "lsst/afw/geom/SipApproximation.h"
#include "lsst/geom/polynomials/PolynomialFunction2d.h"

namespace lsst { namespace afw { namespace geom {

namespace poly = lsst::geom::polynomials;

namespace {

std::pair<poly::PolynomialFunction2dYX, poly::PolynomialFunction2dYX> fitPolynomial(
    int order,
    double svdThreshold,
    std::vector<lsst::geom::Point2D> const & input,
    std::vector<lsst::geom::Point2D> const & output
) {
    auto basis = poly::PolynomialBasis2dYX(order);
    auto workspace = basis.makeWorkspace();
    // Since we want to null the constant terms in the polynomial, we chop off
    // the first column from the matrix we take the SVD of, and then set those
    // coefficients to zero in the result.
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(input.size(), basis.size() - 1);
    Eigen::VectorXd xRhs(input.size());
    Eigen::VectorXd yRhs(input.size());
    Eigen::VectorXd tmp(basis.size());
    for (int i = 0; i < matrix.rows(); ++i) {
        tmp.setZero();
        basis.fill(input[i], tmp, workspace);
        matrix.row(i) = tmp.tail(basis.size() - 1);
        auto rhs = output[i];
        xRhs[i] = rhs.getX();
        yRhs[i] = rhs.getY();
    }
    auto decomp = matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svdThreshold >= 0) {
        decomp.setThreshold(svdThreshold);
    }
    Eigen::VectorXd xCoeff = Eigen::VectorXd::Zero(basis.size());
    if (Eigen::isnan(xCoeff.array()).any()) {
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeError,
            "Polynomial fit failed due to numerical instability; try decreasing the order."
        );
    }
    xCoeff.tail(basis.size() - 1) = decomp.solve(xRhs);
    auto polyX = makeFunction2d(basis, xCoeff);
    Eigen::VectorXd yCoeff = Eigen::VectorXd::Zero(basis.size());
    if (Eigen::isnan(yCoeff.array()).any()) {
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeError,
            "Polynomial fit failed due to numerical instability; try decreasing the order."
        );
    }
    yCoeff.tail(basis.size() - 1) = decomp.solve(yRhs);
    auto polyY = makeFunction2d(basis, yCoeff);
    return std::make_pair(polyX, polyY);
}

// Return a vector of points on a grid, covering the given bounding box.
std::vector<lsst::geom::Point2D> makeGrid(lsst::geom::Box2D const & bbox,
                                          lsst::geom::Extent2I const & shape) {
    if (shape.getX() <= 1 || shape.getY() <= 1) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            "Grid shape values must be two or greater."
        );
    }
    std::vector<lsst::geom::Point2D> points;
    points.reserve(shape.getX()*shape.getY());
    double const dx = bbox.getWidth()/(shape.getX() - 1);
    double const dy = bbox.getHeight()/(shape.getY() - 1);
    for (int iy = 0; iy < shape.getY(); ++iy) {
        double const y = bbox.getMinY() + iy*dy;
        for (int ix = 0; ix < shape.getX(); ++ix) {
            points.emplace_back(bbox.getMinX() + ix*dx, y);
        }
    }
    return points;
}

} // anonymous

// Private implementation object for SipApproximation that manages the grid of points on which
// we evaluate the exact transform when fitting.
struct SipApproximation::FitGrid {

    FitGrid(
        lsst::geom::Box2D const & bbox_,
        lsst::geom::Extent2I const & shape_,
        lsst::geom::Point2D const & crpix,
        TransformPoint2ToPoint2 const & pixelToIwc
    ) :
        bbox(bbox_),
        shape(shape_)
    {
        // Make the pixel grid and transform it to get the intermediate world
        // coordinates grid.
        auto pix = makeGrid(bbox, shape);
        iwc = pixelToIwc.applyForward(pix);

        lsst::geom::Extent2D offset(crpix);

        // Apply the CRPIX offset to make pix into dpix.
        dpix = std::move(pix);
        std::for_each(
            dpix.begin(), dpix.end(),
            [&offset](lsst::geom::Point2D & p){ p -= offset; }
        );
    }

    lsst::geom::Extent2D getStep() const noexcept {
        return lsst::geom::Extent2D(
            bbox.getWidth()/(shape.getX() - 1),
            bbox.getHeight()/(shape.getY() - 1)
        );
    }

    lsst::geom::Box2D bbox;
    lsst::geom::Extent2I shape;
    std::vector<lsst::geom::Point2D> dpix; //   [pixel coords] - CRPIX
    std::vector<lsst::geom::Point2D> iwc;  //   [intermediate world coords]
};

// Private implementation object for SipApproximation that manages the grid of points on which
// we evaluate the exact transform when validating.
struct SipApproximation::ValidationGrid {

    ValidationGrid(FitGrid const & fit_grid, SkyWcs const & target) :
        // Shrink the grid bbox by half of the step size, and shrink the shape
        // by one, to get a grid that is offset from the fit grid.
        bbox(fit_grid.bbox.erodedBy(0.5*fit_grid.getStep())),
        shape(fit_grid.shape - lsst::geom::Extent2I(1, 1)),
        pix(makeGrid(bbox, shape)),
        sky(target.pixelToSky(pix))
    {}

    std::size_t size() const noexcept {
        return pix.size();
    }

    lsst::geom::Box2D bbox;
    lsst::geom::Extent2I shape;
    std::vector<lsst::geom::Point2D> pix;
    std::vector<lsst::geom::SpherePoint> sky;
};


// Private implementation object for SipApproximation that manages the solution
struct SipApproximation::Solution {

    static std::unique_ptr<Solution> fit(int order_, double svdThreshold, SipApproximation const & parent);

    Solution(
        Eigen::Matrix2d const & cd_,
        poly::PolynomialFunction2dYX const & a_,
        poly::PolynomialFunction2dYX const & b_
    ) :
        cd(cd_), a(a_), b(b_)
    {
        LSST_THROW_IF_NE(a.getBasis().getOrder(), b.getBasis().getOrder(),
                         pex::exceptions::InvalidParameterError,
                         "A and B polynomials must have the same order (%d != %d).");
    }

    using Workspace = poly::PolynomialFunction2dYX::Workspace;

    Workspace makeWorkspace() const { return a.makeWorkspace(); }

    Eigen::Matrix2d cd;
    poly::PolynomialFunction2dYX a;
    poly::PolynomialFunction2dYX b;
};

std::unique_ptr<SipApproximation::Solution> SipApproximation::Solution::fit(
    int order,
    double svdThreshold,
    SipApproximation const & parent
) {
    poly::PolynomialBasis2dYX basis(order);
    if (basis.size() > parent._fitGrid->dpix.size()) {
        throw LSST_EXCEPT(
            pex::exceptions::LogicError,
            (boost::format("Number of parameters (%d) is larger than number of data points (%d)")
             % (2*basis.size()) % (2*parent._fitGrid->dpix.size())).str()
        );
    }
    // Fit polynomials R and S that directly map (u, v) = (pixel - crpix) to
    // IWC. These are *not* the SIP A and B yet, because they don't separate
    // out the CD matrix.
    auto [R, S] = fitPolynomial(order, svdThreshold, parent._fitGrid->dpix, parent._fitGrid->iwc);
    // Since there are no linear SIP terms, the linear terms in the polynomial
    // are just the CD matrix coefficients:
    Eigen::Matrix2d cd = Eigen::Matrix2d::Zero();
    cd(0, 0) = R.getCoefficients()[basis.index(1, 0)];
    cd(0, 1) = R.getCoefficients()[basis.index(0, 1)];
    cd(1, 0) = S.getCoefficients()[basis.index(1, 0)];
    cd(1, 1) = S.getCoefficients()[basis.index(0, 1)];
    // The SIP A and B polynomial coefficients for a given order (p, q) can now
    // be computed by multiplying the (R, S) coefficients for (p, q) by the
    // inverse of the CD matrix.  There are no constant or linear (A, B) terms
    // so we leave the first three terms in each polynomial at zero.
    Eigen::Matrix2d cdInv = cd.inverse();
     if (Eigen::isnan(cdInv.array()).any()) {
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeError,
            "CD matrix inversion failed due to numerical instability; try decreasing the SIP order."
        );
    }
    Eigen::Matrix2Xd RS = Eigen::Matrix2Xd::Zero(2, basis.size());
    RS.row(0) = R.getCoefficients();
    RS.row(1) = S.getCoefficients();
    Eigen::Matrix2Xd AB = Eigen::Matrix2Xd::Zero(2, basis.size());
    AB.rightCols(basis.size() - 3) = cdInv * RS.rightCols(basis.size() - 3);
    poly::PolynomialFunction2dYX A(basis, AB.row(0));
    poly::PolynomialFunction2dYX B(basis, AB.row(1));
    return std::make_unique<Solution>(cd, A, B);
}

SipApproximation::SipApproximation(
    SkyWcs const & target,
    lsst::geom::Box2D const & bbox,
    lsst::geom::Extent2I const & gridShape,
    int order,
    std::optional<lsst::geom::Point2D> const & pixelOrigin,
    double svdThreshold
) :
    _bbox(bbox),
    _pixelOrigin(pixelOrigin.has_value() ? pixelOrigin.value() : bbox.getCenter()),
    _skyOrigin(target.pixelToSky(_pixelOrigin))
{
    if (!_bbox.contains(_pixelOrigin)) {
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterError,
            (boost::format("CRPIX value %s is outside the bounding box %s") % _pixelOrigin % _bbox).str()
        );
    }
    if (order < 1) {
        order = 1;
    }
    // Make a TAN WCS at the desired CRPIX and CRVAL with an arbitrary pixel
    // scale, just so we can extract the sky->IWC transform from it (which
    // doesn't depend on the pixel scale).
    auto tanWcs = makeSkyWcs(_pixelOrigin, _skyOrigin, makeCdMatrix(1*lsst::geom::degrees));
    auto pixelToIwc = target.getTransform()->then(
        *getIntermediateWorldCoordsToSky(*tanWcs)->inverted()
    );
    _fitGrid.reset(new FitGrid(bbox, gridShape, _pixelOrigin, *pixelToIwc));
    _validationGrid.reset(new ValidationGrid(*_fitGrid, target));
    _solution = Solution::fit(order, svdThreshold, *this);
    _wcs = makeTanSipWcs(_pixelOrigin, _skyOrigin, _solution->cd, getA(), getB());
}

SipApproximation::~SipApproximation() noexcept = default;

int SipApproximation::getOrder() const noexcept {
    return _solution->a.getBasis().getOrder();
}

double SipApproximation::getA(int p, int q) const {
    return _solution->a[_solution->a.getBasis().index(p, q)];
}

double SipApproximation::getB(int p, int q) const {
    return _solution->b[_solution->b.getBasis().index(p, q)];
}

namespace {

template <typename F>
Eigen::MatrixXd makeCoefficientMatrix(std::size_t order, F getter) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(order + 1, order + 1);
    for (std::size_t p = 0; p <= order; ++p) {
        for (std::size_t q = 0; q <= order - p; ++q) {
            result(p, q) = getter(p, q);
        }
    }
    return result;
}

} // anonymous

Eigen::MatrixXd SipApproximation::getA() const {
    return makeCoefficientMatrix(
        getOrder(),
        [this](int p, int q) { return getA(p, q); }
    );
}

Eigen::MatrixXd SipApproximation::getB() const {
    return makeCoefficientMatrix(
        getOrder(),
        [this](int p, int q) { return getB(p, q); }
    );
}

Eigen::Matrix2d SipApproximation::getCdMatrix() const noexcept {
    return _solution->cd;
}

std::shared_ptr<SkyWcs> SipApproximation::getWcs() const noexcept {
    return _wcs;
}

std::pair<lsst::geom::Angle, double> SipApproximation::computeDeltas() const {
    auto approxSky = _wcs->pixelToSky(_validationGrid->pix);
    auto approxPix = _wcs->skyToPixel(_validationGrid->sky);
    std::pair<lsst::geom::Angle, double> result(0*lsst::geom::arcseconds, 0);
    for (std::size_t i = 0; i < _validationGrid->size(); ++i) {
        auto deltaSky = approxSky[i].separation(_validationGrid->sky[i]);
        auto deltaPix = (approxPix[i] - _validationGrid->pix[i]).computeNorm();
        result.first = std::max(result.first, deltaSky);
        result.second = std::max(result.second, deltaPix);
    }
    return result;
}

}}}  // namespace lsst::afw::geom
