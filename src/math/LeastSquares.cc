// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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

#include "lsst/afw/math/LeastSquares.h"

namespace lsst { namespace afw { namespace math {

class LeastSquares::Impl {
public:

    bool const needsNormalEquations;
    double threshold;
    int dimension;
    int rank;

    Eigen::MatrixXd design;
    Eigen::VectorXd data;
    Eigen::MatrixXd hessian;
    Eigen::VectorXd rhs;

    virtual void factor() = 0;

    virtual ndarray::Array<double const,1,1> solve() = 0;
    virtual ndarray::Array<double const,2,2> computeCovariance() = 0;
    virtual ndarray::Array<double const,2,2> computeHessian() = 0;
    virtual double computeChiSq() = 0;

    Impl(bool needsNormalEquations_, double threshold_) : 
        needsNormalEquations(needsNormalEquations), threshold(threshold_), dimension(0), rank(0)
        {}

    virtual ~Impl() {}
};

namespace {

} // anonymous

void LeastSquares::setThreshold(double threshold) { _impl->threshold = threshold; }

double LeastSquares::getThreshold() const { return _impl->threshold; }

ndarray::Array<double const,1,1> LeastSquares::solve() { return _impl->solve(); }

ndarray::Array<double const,2,2> LeastSquares::computeCovariance() { return _impl->computeCovariance(); }

ndarray::Array<double const,2,2> LeastSquares::computeHessian() { return _impl->computeHessian(); }

double LeastSquares::computeChiSq() { return _impl->computeChiSq(); }

int LeastSquares::getDimension() const { return _impl->dimension; }

int LeastSquares::getRank() const { return _impl->rank; }

LeastSquares::LeastSquares(Factorization factorization) {
    // TODO
}

LeastSquares::~LeastSquares() {}

Eigen::MatrixXd & LeastSquares::_getDesignMatrix() { return _impl->design; }
Eigen::VectorXd & LeastSquares::_getDataVector() { return _impl->data; }

Eigen::MatrixXd & LeastSquares::_getHessianMatrix() { return _impl->hessian; }
Eigen::VectorXd & LeastSquares::_getRhsVector() { return _impl->rhs; }

void LeastSquares::_factor(bool haveNormalEquations) {
    if (_impl->needsNormalEquations && !haveNormalEquations) {
        _getHessianMatrix() = Eigen::MatrixXd::Zero(_getDesignMatrix().cols(), _getDesignMatrix().cols());
        _getHessianMatrix().selfadjointView<Eigen::Lower>().rankUpdate(_getDesignMatrix().transpose());
        _getRhsVector() = _getDesignMatrix().transpose() * _getDataVector();
    }
    _impl->factor();
}

}}} // namespace lsst::afw::math
