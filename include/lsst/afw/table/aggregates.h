// -*- lsst-c++ -*-
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
#ifndef AFW_TABLE_aggregates_h_INCLUDED
#define AFW_TABLE_aggregates_h_INCLUDED

#include "lsst/afw/table/FunctorKey.h"
#include "lsst/afw/table/Schema.h"

namespace lsst { namespace afw {

namespace geom {

template <typename T, int N> class Point;

namespace ellipses {

class Quadrupole;

} // namespace ellipses
} // namespace geom

namespace table {

/**
 *  @brief A FunctorKey used to get or set a geom::Point from an (x,y) pair of int or double Keys.
 */
template <typename T>
class PointKey : public FunctorKey< lsst::afw::geom::Point<T,2> > {
public:

    /// Default constructor; instance will not be usuable unless subsequently assigned to.
    PointKey() : _x(), _y() {}

    /// Construct from a pair of Keys
    PointKey(Key<T> const & x, Key<T> const & y) : _x(x), _y(y) {}

    /**
     *  Construct from a compound Key<Point>
     *
     *  Key<Point> is now deprecated in favor of PointKey; this constructor is intended to
     *  aid in the transition.
     */
    explicit PointKey(Key< Point<T> > const & other) : _x(other.getX()), _y(other.getY()) {}

    /**
     *  @brief Construct from a subschema, assuming .x and .y subfields
     *
     *  If a schema has "a.x" and "a.y" fields, this constructor allows you to construct
     *  a PointKey via:
     *  @code
     *  PointKey<T> k(schema["a"]);
     *  @endcode
     */
    PointKey(SubSchema const & s) : _x(s["x"]), _y(s["y"]) {}

    /// Get a Point from the given record
    virtual geom::Point<T,2> get(BaseRecord const & record) const;

    /// Set a Point in the given record
    virtual void set(BaseRecord & record, geom::Point<T,2> const & value) const;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying x and y Keys
    bool operator==(PointKey<T> const & other) const { return _x == other._x && _y == other._y; }
    bool operator!=(PointKey<T> const & other) const { return !(*this == other); }
    //@}

    /// Return True if both the x and y Keys are valid.
    bool isValid() const { return _x.isValid() && _y.isValid(); }

    /// Return the underlying x Key
    Key<T> getX() const { return _x; }

    /// Return the underlying y Key
    Key<T> getY() const { return _y; }

private:
    Key<T> _x;
    Key<T> _y;
};

typedef PointKey<int> Point2IKey;
typedef PointKey<double> Point2DKey;

/**
 *  @brief A FunctorKey used to get or set a geom::ellipses::Quadrupole from an (xx,yy,xy) tuple of Keys.
 */
class QuadrupoleKey : public FunctorKey< lsst::afw::geom::ellipses::Quadrupole > {
public:

    /// Default constructor; instance will not be usuable unless subsequently assigned to.
    QuadrupoleKey() : _ixx(), _iyy(), _ixy() {}

    /// Construct from individual Keys
    QuadrupoleKey(Key<double> const & ixx, Key<double> const & iyy, Key<double> const & ixy) :
        _ixx(ixx), _iyy(iyy), _ixy(ixy)
    {}

    /**
     *  Construct from a compound Key<Moments<double>>
     *
     *  Key<Moments> is now deprecated in favor of PointKey; this constructor is intended to
     *  aid in the transition.
     */
    explicit QuadrupoleKey(Key< Moments<double> > const & other) :
        _ixx(other.getIxx()), _iyy(other.getIyy()), _ixy(other.getIxy())
    {}

    /**
     *  @brief Construct from a subschema, assuming .xx, .yy, and .xy subfields
     *
     *  If a schema has "a.xx", "a.yy", and "a.xy" fields, this constructor allows you to construct
     *  a QuadrupoleKey via:
     *  @code
     *  QuadrupoleKey k(schema["a"]);
     *  @endcode
     */
    QuadrupoleKey(SubSchema const & s) : _ixx(s["xx"]), _iyy(s["yy"]), _ixy(s["xy"]) {}

    /// Get a Quadrupole from the given record
    virtual geom::ellipses::Quadrupole get(BaseRecord const & record) const;

    /// Set a Quadrupole in the given record
    virtual void set(BaseRecord & record, geom::ellipses::Quadrupole const & value) const;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying Ixx, Iyy, Ixy Keys
    bool operator==(QuadrupoleKey const & other) const {
        return _ixx == other._ixx && _iyy == other._iyy && _ixy == other._ixy;
    }
    bool operator!=(QuadrupoleKey const & other) const { return !(*this == other); }
    //@}

    /// Return True if all the constituent Keys are valid.
    bool isValid() const { return _ixx.isValid() && _iyy.isValid() && _ixy.isValid(); }

    //@{
    /// Return a constituent Key
    Key<double> getIxx() const { return _ixx; }
    Key<double> getIyy() const { return _iyy; }
    Key<double> getIxy() const { return _ixy; }
    //@}

private:
    Key<double> _ixx;
    Key<double> _iyy;
    Key<double> _ixy;
};

template <typename T, int N>
class CovarianceMatrixKey : public FunctorKey< Eigen::Matrix<T,N,N> > {
public:

    typedef std::vector< Key<T> > SigmaKeyArray;
    typedef std::vector< Key<T> > CovarianceKeyArray;
    typedef std::vector<std::string> NameArray;

    /// Construct an invalid instance; must assign before subsequent use.
    CovarianceMatrixKey();

    /**
     *  @brief Construct a from arrays of per-element Keys
     *
     *  The sigma array Keys should point to the square root of the diagonal of the
     *  covariance matrix.  The cov array Keys should point to the off-diagonal elements
     *  of the lower-triangle, packed first in rows, then in columns (or equivalently,
     *  in the upper-triangle, packed first in columns, then in rows).  For a 4x4 matrix,
     *  the order is is:
     *  @code
     *    sigma[0]^2   cov[0]       cov[1]       cov[3]
     *    cov[0]       sigma[1]^2   cov[2]       cov[4]
     *    cov[1]       cov[2]       sigma[2]^2   cov[5]
     *    cov[3]       cov[4]       cov[5]       sigma[3]^2
     *  @endcode
     *
     *  The cov array may also be empty, to indicate that no off-diagonal elements are
     *  stored, and should be set to zero.  If not empty, the size of the cov matrix
     *  must be exactly n*(n-1)/2, where n is the size of the sigma matrix.
     */
    explicit CovarianceMatrixKey(
        SigmaKeyArray const & sigma,
        CovarianceKeyArray const & cov=CovarianceKeyArray()
    );

    /**
     *  @brief Construct from a (now-deprecated Key<Covariance<U>>)
     *
     *  This template is only instantiated for the following combinations of template parameters:
     *   - CovarianceMatrixKey<float,Eigen::Dynamic> and Key< ovariance<float> >
     *   - CovarianceMatrixKey<float,2> and Key< Covariance< Point<float> >
     *   - CovarianceMatrixKey<float,3> and Key< Covariance< Moments<float> >
     *  Calling templates other than these will result in linker errors.
     *
     *  To access this functionality in Python, please use the makeCovarianceMatrixKey free functions;
     *  Swig wasn't able to instantiate the templated constructors.
     */
    template <typename U>
    explicit CovarianceMatrixKey(Key< Covariance<U> > const & other);

    /**
     *  @brief Construct from a subschema and an array of names for each parameter of the matrix.
     *
     *  The field names should match the following convention:
     *   - diagonal elements should have names like "p1Sigma", where "p1" is the name of the parameter,
     *     and should contain the square root of the variance in that parameter.
     *   - off-diagonal elements hould have names like "p1_p2_Cov", where "p1" and "p2" are names of
     *     parameters.
     *  For example, for the covariance matrix of a position, we'd look for "xSigma", "ySigma", and
     *  "x_y_Cov".
     */
    CovarianceMatrixKey(SubSchema const & s, NameArray const & names);

    /// Get a covariance matrix from the given record
    virtual Eigen::Matrix<T,N,N> get(BaseRecord const & record) const;

    /// Set a covariance matrix in the given record (uses only the lower triangle of the given matrix)
    virtual void set(BaseRecord & record, Eigen::Matrix<T,N,N> const & value) const;

    /**
     *  @brief Return True if all the constituent sigma Keys are valid
     *
     *  Note that if the only one or more off-diagonal keys are invalid, we assume that means those terms
     *  are zero, not that the whole FunctorKey is invalid.
     */
    bool isValid() const;

    //@{
    /// Compare the FunctorKey for equality with another, using its constituent Keys
    bool operator==(CovarianceMatrixKey const & other) const;
    bool operator!=(CovarianceMatrixKey const & other) const { return !(*this == other); }
    //@}

private:
    bool _isDiagonalVariance;
    SigmaKeyArray _sigma;
    CovarianceKeyArray _cov;
};

inline CovarianceMatrixKey<float,Eigen::Dynamic>
makeCovarianceMatrixKey(Key< Covariance<float> > const & other) {
    return CovarianceMatrixKey<float,Eigen::Dynamic>(other);
}

inline CovarianceMatrixKey<float,2>
makeCovarianceMatrixKey(Key< Covariance< Point<float> > > const & other) {
    return CovarianceMatrixKey<float,2>(other);
}

inline CovarianceMatrixKey<float,3>
makeCovarianceMatrixKey(Key< Covariance< Moments<float> > > const & other) {
    return CovarianceMatrixKey<float,3>(other);
}

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_aggregates_h_INCLUDED
