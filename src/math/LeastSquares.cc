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

#include "Eigen/Eigenvalues"
#include "Eigen/SVD"
#include "Eigen/Cholesky"
#include "boost/format.hpp"
#include "boost/make_shared.hpp"

#include "lsst/afw/math/LeastSquares.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging.h"

namespace lsst { namespace afw { namespace math {

class LeastSquares::Impl {
public:

    enum MatrixState { NO_MATRIX=0, LOWER_MATRIX=1, FULL_MATRIX=2 };

    MatrixState fisherState;
    double threshold;
    int dimension;
    int rank;

    Eigen::MatrixXd design;
    Eigen::VectorXd data;
    Eigen::MatrixXd fisher;
    Eigen::VectorXd rhs;

    ndarray::Array<double,1,1> solution;
    ndarray::Array<double,2,2> covariance;
    ndarray::Array<double,1,1> condition;

    pex::logging::Debug log;

    template <typename D>
    void setRank(Eigen::MatrixBase<D> const & values) {
        double cond = threshold * values[0];
        for (rank = dimension; (rank > 1) && (values[rank-1] < cond); --rank);
    }

    void computeFisherMatrix(MatrixState desired) {
        if (fisherState < LOWER_MATRIX && desired >= LOWER_MATRIX) {
            fisher = Eigen::MatrixXd::Zero(design.cols(), design.cols());
            fisher.selfadjointView<Eigen::Lower>().rankUpdate(design.adjoint());
            fisherState = LOWER_MATRIX;
        }
        if (fisherState < FULL_MATRIX && desired >= FULL_MATRIX) {
            fisher.triangularView<Eigen::StrictlyUpper>() = fisher.adjoint();
            fisherState = FULL_MATRIX;
        }
    }

    virtual void factor() = 0;

    virtual void updateRank() = 0;

    virtual void solve() = 0;
    virtual void computeCovariance() = 0;

    virtual void getCondition() = 0;

    Impl(int dimension_, double threshold_) : 
        fisherState(NO_MATRIX), threshold(threshold_), dimension(dimension_), rank(dimension_),
        log("afw.math.LeastSquares")
        {}

    virtual ~Impl() {}
};

namespace {

class EigensystemSolver : public LeastSquares::Impl {
public:

    explicit EigensystemSolver(int dimension) :
        Impl(dimension, std::numeric_limits<double>::epsilon()),
        _eig(dimension), _svd(), _tmp(dimension)
    {}
    
    virtual void factor() {
        if (fisherState == NO_MATRIX) {
            rhs = design.adjoint() * data;
        }
        computeFisherMatrix(LOWER_MATRIX);
        _eig.compute(fisher);
        if (_eig.info() == Eigen::Success) {
            setRank(_eig.eigenvalues().reverse());
            log.debug<5>("SelfAdjointEigenSolver succeeded: dimension=%d, rank=%d", dimension, rank);
        } else {
            // Note that the fallback is using SVD of the Fisher to compute the Eigensystem, because those
            // are the same for a symmetric matrix; this is very different from doing a direct SVD of
            // the design matrix.
            computeFisherMatrix(FULL_MATRIX);
            _svd.compute(fisher, Eigen::ComputeFullU); // Matrix is symmetric, so V == U == eigenvectors
            setRank(_svd.singularValues());
            log.debug<5>(
                "SelfAdjointEigenSolver failed; falling back to equivalent SVD: dimension=%d, rank=%d",
                dimension, rank
            );
        }
    }

    virtual void updateRank() {
        if (_eig.info() == Eigen::Success) {
            setRank(_eig.eigenvalues().reverse());
        } else {
            setRank(_svd.singularValues());
        }
    }

    virtual void getCondition() {
        if (_eig.info() == Eigen::Success) {
            condition.asEigen() = _eig.eigenvalues().reverse();
        } else {
            condition.asEigen() = _svd.singularValues();
        }
    }

    virtual void solve() {
        if (_eig.info() == Eigen::Success) {
            _tmp.head(rank) = _eig.eigenvectors().rightCols(rank).adjoint() * rhs;
            _tmp.head(rank).array() /= _eig.eigenvalues().tail(rank).array();
            solution.asEigen() = _eig.eigenvectors().rightCols(rank) * _tmp.head(rank);
        } else {
            _tmp.head(rank) = _svd.matrixU().leftCols(rank).adjoint() * rhs;
            _tmp.head(rank).array() /= _svd.singularValues().head(rank).array();
            solution.asEigen() = _svd.matrixU().leftCols(rank) * _tmp.head(rank);
        }
    }

    virtual void computeCovariance() {
        if (_eig.info() == Eigen::Success) {
            covariance.asEigen() = 
                _eig.eigenvectors().rightCols(rank)
                * _eig.eigenvalues().tail(rank).array().inverse().matrix().asDiagonal()
                * _eig.eigenvectors().rightCols(rank).adjoint();
        } else {
            covariance.asEigen() = 
                _svd.matrixU().leftCols(rank)
                * _svd.singularValues().head(rank).array().inverse().matrix().asDiagonal()
                * _svd.matrixU().leftCols(rank).adjoint();
        }
    }
        
private:
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> _eig;
    Eigen::JacobiSVD<Eigen::MatrixXd> _svd; // only used if Eigendecomposition fails, should be very rare
    Eigen::VectorXd _tmp;
};

class CholeskySolver : public LeastSquares::Impl {
public:

    explicit CholeskySolver(int dimension) : Impl(dimension, 0.0), _ldlt(dimension) {}
    
    virtual void factor() {
        if (fisherState == NO_MATRIX) {
            rhs = design.adjoint() * data;
        }
        computeFisherMatrix(LOWER_MATRIX);
        _ldlt.compute(fisher);
    }

    virtual void updateRank() {}

    virtual void getCondition() { condition.asEigen() = _ldlt.vectorD(); }

    virtual void solve() { solution.asEigen() = _ldlt.solve(rhs); }

    virtual void computeCovariance() {
        ndarray::EigenView<double,2,2> cov(covariance);
        cov.setIdentity();
        cov = _ldlt.solve(cov);
    }
        
private:
    Eigen::LDLT<Eigen::MatrixXd> _ldlt;
};

class SvdSolver : public LeastSquares::Impl {
public:

    explicit SvdSolver(int dimension) :
        Impl(dimension, std::numeric_limits<double>::epsilon()), _svd(), _tmp(dimension)
    {}
    
    virtual void factor() {
        _svd.compute(design, Eigen::ComputeThinU | Eigen::ComputeThinV);
        setRank(_svd.singularValues());
        log.debug<5>("Using direct SVD method; dimension=%d, rank=%d", dimension, rank);
    }

    virtual void updateRank() { setRank(_svd.singularValues()); }

    virtual void getCondition() { condition.asEigen() = _svd.singularValues(); }

    virtual void solve() {
        _tmp.head(rank) = _svd.matrixU().leftCols(rank).adjoint() * data;
        _tmp.head(rank).array() /= _svd.singularValues().head(rank).array();
        solution.asEigen() = _svd.matrixV().leftCols(rank) * _tmp.head(rank);
    }

    virtual void computeCovariance() {
        covariance.asEigen() = 
            _svd.matrixV().leftCols(rank)
            * _svd.singularValues().head(rank).array().inverse().square().matrix().asDiagonal()
            * _svd.matrixV().leftCols(rank).adjoint();
    }
        
private:
    Eigen::JacobiSVD<Eigen::MatrixXd> _svd;
    Eigen::VectorXd _tmp;    
};

} // anonymous

void LeastSquares::setThreshold(double threshold) { _impl->threshold = threshold; _impl->updateRank(); }

double LeastSquares::getThreshold() const { return _impl->threshold; }

ndarray::Array<double const,1,1> LeastSquares::solve() {
    if (_impl->solution.isEmpty()) {
        _impl->solution = ndarray::allocate(_impl->dimension);
    }
    _impl->solve();
    return _impl->solution;
}

ndarray::Array<double const,2,2> LeastSquares::computeCovariance() {
    if (_impl->covariance.isEmpty()) {
        _impl->covariance = ndarray::allocate(_impl->dimension, _impl->dimension);
    }
    _impl->computeCovariance();
    return _impl->covariance;
}

ndarray::Array<double const,2,2> LeastSquares::computeFisherMatrix() {
    _impl->computeFisherMatrix(Impl::FULL_MATRIX);
    // Wrap the Eigen::MatrixXd in an ndarray::Array, using _impl as the reference-counted owner.
    // Doesn't matter if we swap strides, because it's symmetric.
    return ndarray::external(
        _impl->fisher.data(),
        ndarray::makeVector(_impl->dimension, _impl->dimension),
        ndarray::makeVector(_impl->dimension, 1),
        _impl
    );
}

ndarray::Array<double const,1,1> LeastSquares::getCondition() {
    if (_impl->condition.isEmpty()) {
        _impl->condition = ndarray::allocate(_impl->dimension);
    }
    _impl->getCondition();
    return _impl->condition;
}

int LeastSquares::getDimension() const { return _impl->dimension; }

int LeastSquares::getRank() const { return _impl->rank; }

LeastSquares::LeastSquares(Factorization factorization, int dimension) {
    switch (factorization) {
    case NORMAL_EIGENSYSTEM:
        _impl = boost::make_shared<EigensystemSolver>(dimension);
        break;
    case NORMAL_CHOLESKY:
        _impl = boost::make_shared<CholeskySolver>(dimension);
        break;
    case DIRECT_SVD:
        _impl = boost::make_shared<SvdSolver>(dimension);
        break;        
    }
}

LeastSquares::~LeastSquares() {}

Eigen::MatrixXd & LeastSquares::_getDesignMatrix() { return _impl->design; }
Eigen::VectorXd & LeastSquares::_getDataVector() { return _impl->data; }

Eigen::MatrixXd & LeastSquares::_getFisherMatrix() { return _impl->fisher; }
Eigen::VectorXd & LeastSquares::_getRhsVector() { return _impl->rhs; }

void LeastSquares::_factor(bool haveNormalEquations) {
    if (haveNormalEquations) {
        if (_getFisherMatrix().rows() != _impl->dimension) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                (boost::format("Number of rows of Fisher matrix (%d) does not match"
                               " dimension of LeastSquares solver.")
                 % _getFisherMatrix().rows() % _impl->dimension).str()
            );
        }
        if (_getFisherMatrix().cols() != _impl->dimension) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                (boost::format("Number of columns of Fisher matrix (%d) does not match"
                               " dimension of LeastSquares solver.")
                 % _getFisherMatrix().cols() % _impl->dimension).str()
            );
        }
        if (_getRhsVector().size() != _impl->dimension) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                (boost::format("Number of elements in RHS vector (%d) does not match"
                               " dimension of LeastSquares solver.")
                 % _getRhsVector().size() % _impl->dimension).str()
            );
        }
        _impl->fisherState = Impl::FULL_MATRIX;
    } else {
        if (_getDesignMatrix().cols() != _impl->dimension) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                "Number of columns of design matrix does not match dimension of LeastSquares solver."
            );
        }
        if (_getDesignMatrix().rows() != _getDataVector().size()) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                (boost::format("Number of rows of design matrix (%d) does not match number of "
                               "data points (%d)") % _getDesignMatrix().rows() % _getDataVector().size()
                ).str()
            );
        }
        if (_getDesignMatrix().cols() > _getDataVector().size()) {
            throw LSST_EXCEPT(
                pex::exceptions::InvalidParameterException,
                (boost::format("Number of columns of design matrix (%d) must be smaller than number of "
                               "data points (%d)") % _getDesignMatrix().cols() % _getDataVector().size()
                ).str()
            );
        }
        _impl->fisherState = Impl::NO_MATRIX;
    }
    _impl->factor();
}

}}} // namespace lsst::afw::math
