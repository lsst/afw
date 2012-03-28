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

#ifndef LSST_AFW_MATH_LeastSquares_h_INCLUDED
#define LSST_AFW_MATH_LeastSquares_h_INCLUDED

#include "ndarray/eigen.h"

namespace lsst { namespace afw { namespace math {

/**
 *  @brief Solver for linear least-squares problems.
 *
 *  Linear least-squares problems are defined as finding the vector @f$x@f$ that minimizes 
 *  @f$ \left|\bm{A} \bm{x} -\bm{b}\right|_2 @f$, with the number of rows of @f$A@f$ generally
 *  greater than the number of columns.  We call @f$\bm{A}@f$ the design matrix, @f$\bm{b}@f$
 *  the data vector, and @f$\bm{x}@f$ the solution vector.  When the rank of @f$\bm{A}@f$ is
 *  equal to the number of columns, we can obtain using the solution using the normal equations:
 *  @f[
 *      \bm{A}^T \bm{A} \bm{x} = \bm{A}^T \bm{b}
 *  @f]
 *  If @f$\bm{A}@f$ is not full-rank, the problem is underconstrained, and we usually wish to
 *  solve the minimum-norm least-squares problem, which also minimizes @f$|\bm{x}|_2@f$.
 *  This can be done by computing a pseudo-inverse of @f$\bm{A}@f$ using an SVD or complete
 *  orthogonal factorization of @f$\bm{A}@f$, or by performing an Eigen decomposition of
 *  @f$\bm{A}^T\bm{A}@f$.
 *
 *  This class can be constructed from the design matrix and data vector, or from the two terms
 *  in the normal equations (below, we call the matrix @f$\bm{A}^T\bm{A}@f$ the Hessian, as it
 *  is the second-derivative matrix of the model w.r.t. the parameters).  If initialized with
 *  the design matrix and data vector, we can still use the normal equations to solve it.
 *  The solution via the normal equations is more susceptible to round-off error, but it is also
 *  faster, and if the normal equation terms can be computed directly it can be significantly
 *  less expensive in terms of memory.  The Hessian matrix is a symmetric matrix, and it should
 *  be exactly symmetric when provided as input, because which triangle will be used is an
 *  implementation detail that is subject to change.
 *
 *  The solver always operates in double precision, and returns all results in double precision.
 *  However, it can be initialized from single precision inputs.  It isn't usually a good idea
 *  to construct the normal equations in single precision, however, even when the data are
 *  single precision.
 */
class LeastSquares {
public:

    enum Factorization {
        NORMAL_EIGENSYSTEM,  /**<
                              *   @brief Use the normal equations with a symmetric Eigensystem decomposition.
                              *
                              *   This method is fully robust and computes the minimum-norm solution when
                              *   the problem does not have full rank.  It is affected slightly more by
                              *   round-off error than the SVD method, but because it can handle singular
                              *   matrices this usually isn't a problem.
                              */
        NORMAL_CHOLESKY,     /**<
                              *   @brief Use the normal equations with a Cholesky decomposition.
                              *
                              *   While this method uses a robust LDL^T decomposition that does not
                              *   require square roots, it is not appropriate for problems that do
                              *   not have full rank, and cannot be used to determine whether a
                              *   problem has full rank.  It is the fastest decomposition.
                              */
        DIRECT_SVD           /**<
                              *   @brief Use a thin singular value decomposition of the design matrix.
                              *
                              *   This method is the most robust and computes the minimum-norm solution
                              *   the problem does not have full rank.  However, it is also the slowest
                              *   and is not available when the solver is initialized with the
                              *   normal equations.
                              */
    };

    /// @brief Initialize from the design matrix and data vector given as ndarrays.
    template <typename T1, typename T2, int C1, int C2>
    static LeastSquares fromDesignMatrix(
        ndarray::Array<T1 const,2,C1> const & design,
        ndarray::Array<T2 const,1,C2> const & data,
        Factorization factorization
    ) {
        _initialize(factorization);
        _getDesignMatrix() = design.asEigen();
        _getDataVector() = data.asEigen();
        _factor(false);
    }

    /// @brief Initialize from the design matrix and data vector given as an Eigen objects.
    template <typename D1, typename D2>
    static LeastSquares fromDesignMatrix(
        Eigen::MatrixBase<D1> const & design,
        Eigen::MatrixBase<D2> const & data,
        Factorization factorization
    ) {
        _initialize(factorization);
        _getDesignMatrix() = design;
        _getDataVector() = data;
        _factor(false);
    }

    /// @brief Reset the design matrix and data vector given as ndarrays; dimension must not change.
    template <typename T1, typename T2, int C1, int C2>
    void setDesignMatrix(
        ndarray::Array<T1 const,2,C1> const & design,
        ndarray::Array<T2 const,1,C2> const & data
    ) {
        _getDesignMatrix() = design.asEigen();
        _getDataVector() = data.asEigen();
        _factor(false);
    }

    /// @brief Reset the design matrix and data vector given as Eigen objects; dimension must not change.
    template <typename D1, typename D2>
    void setDesignMatrix(
        Eigen::MatrixBase<D1> const & design,
        Eigen::MatrixBase<D2> const & data
    ) {
        _getDesignMatrix() = design;
        _getDataVector() = data;
        _factor(false);
    }

    /// @brief Reset the design matrix given as an ndarray; dimension and data are not changed.
    template <typename T1, typename T2, int C1, int C2>
    void setDesignMatrix(ndarray::Array<T1 const,2,C1> const & design) {
        _getDesignMatrix() = design.asEigen();
        _factor(false);
    }

    /// @brief Reset the design matrix given as an Eigen object; dimension and data are not changed.
    template <typename D1, typename D2>
    void setDesignMatrix(Eigen::MatrixBase<D1> const & design) {
        _getDesignMatrix() = design;
        _factor(false);
    }
    

    /// @brief Initialize from the terms in the normal equations, given as ndarrays.
    template <typename T1, typename T2, int C1, int C2>
    static LeastSquares fromNormalEquations(
        ndarray::Array<T1 const,2,C1> const & hessian,
        ndarray::Array<T2 const,1,C2> const & rhs,
        Factorization factorization
    ) {
        _initialize(factorization);
        if (C1 > 0 == Eigen::MatrixXd::IsRowMajor)
            _getHessianMatrix() = hessian.asEigen();
        else
            _getHessianMatrix() = hessian.asEigen().transpose();
        _getRhsVector() = rhs.asEigen();
        _factor(true);
    }

    /// @brief Initialize from the terms in the normal equations, given as Eigen objects.
    template <typename D1, typename D2>
    static LeastSquares fromNormalEquations(
        Eigen::MatrixBase<D1> const & hessian,
        Eigen::MatrixBase<D2> const & rhs,
        Factorization factorization
    ) {
        _initialize(factorization);
        if (C1 > 0 == Eigen::MatrixXd::IsRowMajor)
            _getHessianMatrix() = hessian;
        else
            _getHessianMatrix() = hessian.transpose();
        _getRhsVector() = rhs;
        _factor(true);
    }

    /// @brief Reset the terms in the normal equations given as ndarrays; dimension must not change.
    template <typename T1, typename T2, int C1, int C2>
    void setNormalEquations(
        ndarray::Array<T1 const,2,C1> const & hessian,
        ndarray::Array<T2 const,1,C2> const & rhs
    ) {
        if (C1 > 0 == Eigen::MatrixXd::IsRowMajor)
            _getHessianMatrix() = hessian.asEigen();
        else
            _getHessianMatrix() = hessian.asEigen().transpose();
        _getRhsVector() = rhs.asEigen();
        _factor(true);
    }

    /// @brief Reset the terms in the normal equations given as Eigen objects; dimension must not change.
    template <typename D1, typename D2>
    void setNormalEquations(
        Eigen::MatrixBase<D1> const & hessian,
        Eigen::MatrixBase<D2> const & rhs
    ) {
        if (C1 > 0 == Eigen::MatrixXd::IsRowMajor)
            _getHessianMatrix() = hessian;
        else
            _getHessianMatrix() = hessian.transpose();
        _getRhsVector() = rhs;
        _factor(true);        
    }

    /**
     *  @brief Set the threshold used to determine when to truncate Eigenvalues.
     *
     *  @todo need a more precise definition here
     */
    void setThreshold(double threshold);

    /// @brief Get the threshold used to determine when to truncate Eigenvalues.
    double getThreshold() const;

    /**
     *  @brief Return the vector solution to the least squares problem.
     *
     *  The returned array is owned by the LeastSquares object and may be modified in-place
     *  by future calls to LeastSquares member functions, so it's best to promptly copy the
     *  result elsewhere.
     *
     *  If you want an Eigen object instead, just use solve().asEigen().
     */
    ndarray::Array<double const,1,1> solve();

    /**
     *  @brief Return the covariance matrix of the least squares problem.
     *
     *  The returned array is owned by the LeastSquares object and may be modified in-place
     *  by future calls to LeastSquares member functions, so it's best to promptly copy the
     *  result elsewhere.
     *
     *  If you want an Eigen object instead, just use computeCovariance().asEigen().
     */
    ndarray::Array<double const,2,2> computeCovariance();

    /**
     *  @brief Return the Hessian matrix (inverse of the covariance) of the parameters.
     *
     *  Note that the Hessian matrix is exactly the same as the matrix on the lhs of the
     *  normal equations.
     *
     *  The returned array is owned by the LeastSquares object and may be modified in-place
     *  by future calls to LeastSquares member functions, so it's best to promptly copy the
     *  result elsewhere.
     *
     *  If you want an Eigen object instead, just use computeCovariance().asEigen().
     */
    ndarray::Array<double const,2,2> computeHessian();

    /// @brief Compute the unreduced chi-squared (sum of squared residuals).
    double computeChiSq();

    /// @brief Return the number of parameters.
    int getDimension() const;

    /**
     *  @brief Return the rank of the problem (number of nonzero Eigenvalues).
     *
     *  The returned value is always the same as getDimension() when the factorization is NORMAL_CHOLESKY
     *  (which may be incorrect, because a Cholesky decomposition is not rank-revealing).
     */
    int getRank() const;

    // Need to define dtor in source file so it can see Impl declaration.
    ~LeastSquares();
    
private:

    void _initialize(Factorization factorization);

    // We want a column-major design matrix so the self-adjoint product is cache-friendly, hence '-2'...
    // so we always copy a (possibly row-major) design matrix into a col-major one.  This is an
    // unnecessarily and cache-unfriendly operation when solver is DIRECT_SVD, but right now it doesn't
    // seem to be worth special-casing the design for that case.  In other cases it's a cache-unfriendly
    // op that avoids an even worse one, and it can always be avoided by using a column-major design matrix.
    Eigen::MatrixXd & _getDesignMatrix();
    Eigen::VectorXd & _getDataVector();

    // Storage order matters less here, so we just go with what Eigen is most accustomed to.
    // Because the Hessian is symmetric, we can just transpose it before copying to avoid doing
    // any expensive copies between different storage orders.
    Eigen::MatrixXd & _getHessianMatrix();
    Eigen::VectorXd & _getRhsVector();

    void _factor(bool haveNormalEquations);

    class Impl;

    explicit LeastSquares(PTR(Impl) impl) : _impl(impl) {}

    PTR(Impl) _impl;
};

}}} // namespace lsst::afw::math

#endif // !LSST_AFW_MATH_LeastSquares_h_INCLUDED
