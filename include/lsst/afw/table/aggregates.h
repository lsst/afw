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

#include "lsst/utils/hashCombine.h"

#include "lsst/afw/table/FunctorKey.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/geom.h"

namespace lsst {
namespace afw {
namespace geom {

namespace ellipses {

class Quadrupole;

}  // namespace ellipses
}  // namespace geom

namespace table {

/**
 *  A FunctorKey used to get or set a lsst::geom::Point from an (x,y) pair of int or double Keys.
 */
template <typename T>
class PointKey : public FunctorKey<lsst::geom::Point<T, 2> > {
public:
    /**
     *  Add a pair of _x, _y fields to a Schema, and return a PointKey that points to them.
     *
     *  @param[in,out] schema  Schema to add fields to.
     *  @param[in]     name    Name prefix for all fields; "_x", "_y", will be appended to this
     *                         to form the full field names.
     *  @param[in]     doc     String used as the documentation for the fields.
     *  @param[in]     unit    String used as the unit for all fields.
     */
    static PointKey addFields(Schema& schema, std::string const& name, std::string const& doc,
                              std::string const& unit);

    /// Default constructor; instance will not be usable unless subsequently assigned to.
    PointKey() noexcept : _x(), _y() {}

    /// Construct from a pair of Keys
    PointKey(Key<T> const& x, Key<T> const& y) noexcept : _x(x), _y(y) {}

    PointKey(PointKey const&) noexcept = default;
    PointKey(PointKey&&) noexcept = default;
    PointKey& operator=(PointKey const&) noexcept = default;
    PointKey& operator=(PointKey&&) noexcept = default;
    ~PointKey() noexcept override = default;

    /**
     *  Construct from a subschema, assuming x and y subfields
     *
     *  If a schema has "a_x" and "a_y" fields, this constructor allows you to construct
     *  a PointKey via:
     *
     *      PointKey<T> k(schema["a"]);
     */
    PointKey(SubSchema const& s) : _x(s["x"]), _y(s["y"]) {}

    /// Get a Point from the given record
    lsst::geom::Point<T, 2> get(BaseRecord const& record) const override;

    /// Set a Point in the given record
    void set(BaseRecord& record, lsst::geom::Point<T, 2> const& value) const override;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying x and y Keys
    bool operator==(PointKey<T> const& other) const noexcept { return _x == other._x && _y == other._y; }
    bool operator!=(PointKey<T> const& other) const noexcept { return !(*this == other); }
    //@}

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept {
        // Completely arbitrary seed
        return utils::hashCombine(17, _x, _y);
    }

    /// Return True if both the x and y Keys are valid.
    bool isValid() const noexcept { return _x.isValid() && _y.isValid(); }

    /// Return the underlying x Key
    Key<T> getX() const noexcept { return _x; }

    /// Return the underlying y Key
    Key<T> getY() const noexcept { return _y; }

private:
    Key<T> _x;
    Key<T> _y;
};

typedef PointKey<int> Point2IKey;
typedef PointKey<double> Point2DKey;

/**
 *  A FunctorKey used to get or set a lsst::geom::Box2I or Box2D from a (min, max) pair of PointKeys.
 *
 *  The Box2IKey and Box2DKey typedefs should be preferred to using the template name directly.
 */
template <typename Box>
class BoxKey : public FunctorKey<Box> {
public:
    /// Type of coordinate elements (i.e. int or double).
    using Element = typename Box::Element;

    /**
     *  Add _min_x, _min_y, _max_x, _max_y fields to a Schema, and return a BoxKey that points to them.
     *
     *  @param[in,out] schema  Schema to add fields to.
     *  @param[in]     name    Name prefix for all fields; suffixes above will
     *                         be appended to this to form the full field
     *                         names.  For example, if `name == "b"`, the
     *                         fields added will be "b_min_x", "b_min_y",
     *                         "b_max_x", and "b_max_y".
     *  @param[in]     doc     String used as the documentation for the fields.
     *  @param[in]     unit    String used as the unit for all fields.
     *
     */
    static BoxKey addFields(Schema& schema, std::string const& name, std::string const& doc,
                            std::string const& unit);

    /// Default constructor; instance will not be usable unless subsequently assigned to.
    BoxKey() noexcept = default;

    /// Construct from a pair of PointKeys
    BoxKey(PointKey<Element> const& min, PointKey<Element> const& max) noexcept : _min(min), _max(max) {}

    /**
     *  Construct from a subschema, assuming _min_x, _max_x, _min_y, _max_y subfields
     *
     *  If a schema has "a_min_x" and "a_min_x" (etc) fields, this constructor allows you to construct
     *  a BoxKey via:
     *
     *      BoxKey<Box> k(schema["a"]);
     */
    BoxKey(SubSchema const& s) : _min(s["min"]), _max(s["max"]) {}

    BoxKey(BoxKey const&) noexcept = default;
    BoxKey(BoxKey&&) noexcept = default;
    BoxKey& operator=(BoxKey const&) noexcept = default;
    BoxKey& operator=(BoxKey&&) noexcept = default;
    ~BoxKey() noexcept override = default;

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept {
        // Completely arbitrary seed
        return utils::hashCombine(17, _min, _max);
    }

    /// Get a Point from the given record
    Box get(BaseRecord const& record) const override;

    /// Set a Point in the given record
    void set(BaseRecord& record, Box const& value) const override;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying x and y Keys
    bool operator==(BoxKey const& other) const noexcept { return _min == other._min && _max == other._max; }
    bool operator!=(BoxKey const& other) const noexcept { return !(*this == other); }
    //@}

    /// Return True if both the min and max PointKeys are valid.
    bool isValid() const noexcept { return _min.isValid() && _max.isValid(); }

    /// Return the underlying min PointKey
    PointKey<Element> getMin() const noexcept { return _min; }

    /// Return the underlying max PointKey
    PointKey<Element> getMax() const noexcept { return _max; }

private:
    PointKey<Element> _min;
    PointKey<Element> _max;
};

using Box2IKey = BoxKey<lsst::geom::Box2I>;
using Box2DKey = BoxKey<lsst::geom::Box2D>;

/**
 *  A FunctorKey used to get or set celestial coordinates from a pair of lsst::geom::Angle keys.
 *
 *  Coords are always stored and returned in the ICRS system. Coords in other
 *  systems may be assigned, but this will result in a conversion to ICRS.
 */
class CoordKey : public FunctorKey<lsst::geom::SpherePoint> {
public:
    /**
     *  Add a pair of _ra, _dec fields to a Schema, and return a CoordKey that points to them.
     *
     *  @param[in,out] schema  Schema to add fields to.
     *  @param[in]     name    Name prefix for all fields; "_ra", "_dec", will be appended
     *                         to this to form the full field names.
     *  @param[in]     doc     String used as the documentation for the fields.
     */
    static CoordKey addFields(afw::table::Schema& schema, std::string const& name, std::string const& doc);

    /// Default constructor; instance will not be usable unless subsequently assigned to.
    CoordKey() noexcept : _ra(), _dec() {}

    /// Construct from a pair of Keys
    CoordKey(Key<lsst::geom::Angle> const& ra, Key<lsst::geom::Angle> const& dec) noexcept
            : _ra(ra), _dec(dec) {}

    /**
     *  Construct from a subschema, assuming ra and dec subfields.
     *
     *  If a schema has "a_ra" and "a_dec" fields, this constructor allows you to
     *  construct a CoordKey via:
     *
     *      CoordKey k(schema["a"]);
     */
    CoordKey(SubSchema const& s) : _ra(s["ra"]), _dec(s["dec"]) {}

    CoordKey(CoordKey const&) noexcept = default;
    CoordKey(CoordKey&&) noexcept = default;
    CoordKey& operator=(CoordKey const&) noexcept = default;
    CoordKey& operator=(CoordKey&&) noexcept = default;
    ~CoordKey() noexcept override = default;

    /// Get an lsst::geom::SpherePoint from the given record
    lsst::geom::SpherePoint get(BaseRecord const& record) const override;

    /// Set an lsst::geom::SpherePoint in the given record
    void set(BaseRecord& record, lsst::geom::SpherePoint const& value) const override;

    //@{
    /// Compare CoordKeys for equality using the constituent `ra` and `dec` Keys
    bool operator==(CoordKey const& other) const noexcept { return _ra == other._ra && _dec == other._dec; }
    bool operator!=(CoordKey const& other) const noexcept { return !(*this == other); }
    //@}

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept {
        // Completely arbitrary seed
        return utils::hashCombine(17, _ra, _dec);
    }

    bool isValid() const noexcept { return _ra.isValid() && _dec.isValid(); }

    //@{
    /// Return a constituent Key
    Key<lsst::geom::Angle> getRa() const noexcept { return _ra; }
    Key<lsst::geom::Angle> getDec() const noexcept { return _dec; }
    //@}

private:
    Key<lsst::geom::Angle> _ra;
    Key<lsst::geom::Angle> _dec;
};

/// Enum used to set units for geometric FunctorKeys
enum class CoordinateType { PIXEL, CELESTIAL };

/**
 *  A FunctorKey used to get or set a geom::ellipses::Quadrupole from a tuple of constituent Keys.
 */
class QuadrupoleKey : public FunctorKey<lsst::afw::geom::ellipses::Quadrupole> {
public:
    /**
     *
     *  Add a set of quadrupole subfields to a schema and return a QuadrupoleKey that points to them.
     *
     *  @param[in,out] schema     Schema to add fields to.
     *  @param[in]     name       Name prefix for all fields; ("_xx", "_yy", "_xy") will be appended to
     *                            this to form the full field names. In celestial coordinates, we use "x"
     *                            as a synonym for "RA" and "y" for "dec".
     *  @param[in]     doc        String used as the documentation for the fields.
     *  @param[in]     coordType  Type of coordinates in use (PIXEL or CELESTIAL).
     */
    static QuadrupoleKey addFields(Schema& schema, std::string const& name, std::string const& doc,
                                   CoordinateType coordType = CoordinateType::PIXEL);

    /// Default constructor; instance will not be usable unless subsequently assigned to.
    QuadrupoleKey() noexcept : _ixx(), _iyy(), _ixy() {}

    /// Construct from individual Keys
    QuadrupoleKey(Key<double> const& ixx, Key<double> const& iyy, Key<double> const& ixy) noexcept
            : _ixx(ixx), _iyy(iyy), _ixy(ixy) {}

    /**
     *  Construct from a subschema with appropriate subfields
     *
     *  If the schema has "a_xx", "a_yy" and "a_xy" fields this constructor enables you to
     *  construct a QuadrupoleKey via:
     *
     *      QuadrupoleKey k(schema["a"], coordType);
     */
    QuadrupoleKey(SubSchema const& s) : _ixx(s["xx"]), _iyy(s["yy"]), _ixy(s["xy"]) {}

    QuadrupoleKey(QuadrupoleKey const&) noexcept = default;
    QuadrupoleKey(QuadrupoleKey&&) noexcept = default;
    QuadrupoleKey& operator=(QuadrupoleKey const&) noexcept = default;
    QuadrupoleKey& operator=(QuadrupoleKey&&) noexcept = default;
    ~QuadrupoleKey() noexcept override = default;

    /// Get a Quadrupole from the given record
    geom::ellipses::Quadrupole get(BaseRecord const& record) const override;

    /// Set a Quadrupole in the given record
    void set(BaseRecord& record, geom::ellipses::Quadrupole const& value) const override;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying Ixx, Iyy, Ixy Keys
    bool operator==(QuadrupoleKey const& other) const noexcept {
        return _ixx == other._ixx && _iyy == other._iyy && _ixy == other._ixy;
    }
    bool operator!=(QuadrupoleKey const& other) const noexcept { return !(*this == other); }
    //@}

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept {
        // Completely arbitrary seed
        return utils::hashCombine(17, _ixx, _iyy, _ixy);
    }

    /// Return True if all the constituent Keys are valid.
    bool isValid() const noexcept { return _ixx.isValid() && _iyy.isValid() && _ixy.isValid(); }

    //@{
    /// Return a constituent Key
    Key<double> getIxx() const noexcept { return _ixx; }
    Key<double> getIyy() const noexcept { return _iyy; }
    Key<double> getIxy() const noexcept { return _ixy; }
    //@}

private:
    Key<double> _ixx;
    Key<double> _iyy;
    Key<double> _ixy;
};

/**
 *  A FunctorKey used to get or set a geom::ellipses::Ellipse from an (xx,yy,xy,x,y) tuple of Keys.
 */
class EllipseKey : public FunctorKey<lsst::afw::geom::ellipses::Ellipse> {
public:
    /**
     *  Add a set of _xx, _yy, _xy, _x, _y fields to a Schema, and return an EllipseKey that points to them.
     *
     *  @param[in,out] schema  Schema to add fields to.
     *  @param[in]     name    Name prefix for all fields; "_xx", "_yy", "_xy", "_x" ,"_y", will be
     *                         appended to this to form the full field names.
     *  @param[in]     doc     String used as the documentation for the fields.
     *  @param[in]     unit    String used as the unit for x and y fields; "<unit>^2" will be used for
     *                         xx, yy, and xy fields.
     */
    static EllipseKey addFields(Schema& schema, std::string const& name, std::string const& doc,
                                std::string const& unit);

    /// Default constructor; instance will not be usable unless subsequently assigned to.
    EllipseKey() noexcept : _qKey(), _pKey() {}

    /// Construct from individual Keys
    EllipseKey(QuadrupoleKey const& qKey, PointKey<double> const& pKey) noexcept : _qKey(qKey), _pKey(pKey) {}

    /**
     *  Construct from a subschema, assuming (xx, yy, xy, x, y) subfields
     *
     *  If a schema has "a_xx", "a_yy", "a_xy", "a_x", and "a_y" fields, this constructor allows you to
     *  construct an EllipseKey via:
     *
     *      EllipseKey k(schema["a"]);
     */
    EllipseKey(SubSchema const& s) : _qKey(s), _pKey(s) {}

    EllipseKey(EllipseKey const&) noexcept = default;
    EllipseKey(EllipseKey&&) noexcept = default;
    EllipseKey& operator=(EllipseKey const&) noexcept = default;
    EllipseKey& operator=(EllipseKey&&) noexcept = default;
    ~EllipseKey() noexcept override = default;

    /// Get an Ellipse from the given record
    geom::ellipses::Ellipse get(BaseRecord const& record) const override;

    /// Set an Ellipse in the given record
    void set(BaseRecord& record, geom::ellipses::Ellipse const& value) const override;

    //@{
    /// Compare the FunctorKey for equality with another, using the underlying Ixx, Iyy, Ixy Keys
    bool operator==(EllipseKey const& other) const noexcept {
        return _qKey == other._qKey && _pKey == other._pKey;
    }
    bool operator!=(EllipseKey const& other) const noexcept { return !(*this == other); }
    //@}

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept {
        // Completely arbitrary seed
        return utils::hashCombine(17, _qKey, _pKey);
    }

    /// Return True if all the constituent Keys are valid.
    bool isValid() const noexcept { return _qKey.isValid() && _pKey.isValid(); }

    //@{
    /// Return constituent FunctorKeys
    QuadrupoleKey getCore() const noexcept { return _qKey; }
    PointKey<double> getCenter() const noexcept { return _pKey; }
    //@}

private:
    QuadrupoleKey _qKey;
    PointKey<double> _pKey;
};

template <typename T, int N>
class CovarianceMatrixKey : public FunctorKey<Eigen::Matrix<T, N, N> > {
public:
    typedef std::vector<Key<T> > ErrKeyArray;
    typedef std::vector<Key<T> > CovarianceKeyArray;
    typedef std::vector<std::string> NameArray;

    /**
     *  Add covariance matrix fields to a Schema, and return a CovarianceMatrixKey to manage them.
     *
     *  @param[out] schema    Schema to add fields to.
     *  @param[in]  prefix    String used to form the first part of all field names.  Suffixes of
     *                        the form '_xErr' and '_x_y_Cov' will be added to form the full
     *                        field names (using names={'x', 'y'} as an example).
     *  @param[in]  unit      Unit for for error (standard deviation) values; covariance matrix
     *                        elements will be unit^2.
     *  @param[in]  names     Vector of strings containing the names of the quantities the
     *                        covariance matrix represents the uncertainty of.
     *  @param[in]  diagonalOnly   If true, only create fields for the error values.
     */
    static CovarianceMatrixKey addFields(Schema& schema, std::string const& prefix, NameArray const& names,
                                         std::string const& unit, bool diagonalOnly = false);

    /**
     *  Add covariance matrix fields to a Schema, and return a CovarianceMatrixKey to manage them.
     *
     *  @param[out] schema    Schema to add fields to.
     *  @param[in]  prefix    String used to form the first part of all field names.  Suffixes of
     *                        the form '_xErr' and '_x_y_Cov' will be added to form the full
     *                        field names (using names={'x', 'y'} as an example).
     *  @param[in]  units     Vector of units for for error (standard deviation) values; covariance
     *                        matrix elements will have "{units[i]} {units[j]}" or "{units[i]}^2",
     *                        depending on whether units[i] == units[j].
     *  @param[in]  names     Vector of strings containing the names of the quantities the
     *                        covariance matrix represents the uncertainty of.
     *  @param[in]  diagonalOnly   If true, only create fields for the error values.
     */
    static CovarianceMatrixKey addFields(Schema& schema, std::string const& prefix, NameArray const& names,
                                         NameArray const& units, bool diagonalOnly = false);

    /// Construct an invalid instance; must assign before subsequent use.
    CovarianceMatrixKey();

    /**
     *  Construct a from arrays of per-element Keys
     *
     *  The err array Keys should point to the square root of the diagonal of the
     *  covariance matrix.  The cov array Keys should point to the off-diagonal elements
     *  of the lower-triangle, packed first in rows, then in columns (or equivalently,
     *  in the upper-triangle, packed first in columns, then in rows).  For a 4x4 matrix,
     *  the order is is:
     *
     *      err[0]^2   cov[0]     cov[1]     cov[3]
     *      cov[0]     err[1]^2   cov[2]     cov[4]
     *      cov[1]     cov[2]     err[2]^2   cov[5]
     *      cov[3]     cov[4]     cov[5]     err[3]^2
     *
     *  The cov array may also be empty, to indicate that no off-diagonal elements are
     *  stored, and should be set to zero.  If not empty, the size of the cov matrix
     *  must be exactly n*(n-1)/2, where n is the size of the err matrix.
     */
    explicit CovarianceMatrixKey(ErrKeyArray const& err,
                                 CovarianceKeyArray const& cov = CovarianceKeyArray());

    /**
     *  Construct from a subschema and an array of names for each parameter of the matrix.
     *
     *  The field names should match the following convention:
     *   - diagonal elements should have names like "p1Err", where "p1" is the name of the parameter,
     *     and should contain the square root of the variance in that parameter.
     *   - off-diagonal elements hould have names like "p1_p2_Cov", where "p1" and "p2" are names of
     *     parameters.
     *  For example, for the covariance matrix of a position, we'd look for "xErr", "yErr", and
     *  "x_y_Cov".
     */
    CovarianceMatrixKey(SubSchema const& s, NameArray const& names);

    CovarianceMatrixKey(CovarianceMatrixKey const&);
    CovarianceMatrixKey(CovarianceMatrixKey&&);
    CovarianceMatrixKey& operator=(CovarianceMatrixKey const&);
    CovarianceMatrixKey& operator=(CovarianceMatrixKey&&);
    ~CovarianceMatrixKey() noexcept override;

    /// Get a covariance matrix from the given record
    Eigen::Matrix<T, N, N> get(BaseRecord const& record) const override;

    /// Set a covariance matrix in the given record (uses only the lower triangle of the given matrix)
    void set(BaseRecord& record, Eigen::Matrix<T, N, N> const& value) const override;

    /// Return the element in row i and column j
    T getElement(BaseRecord const& record, int i, int j) const;

    /// Set the element in row i and column j
    void setElement(BaseRecord& record, int i, int j, T value) const;

    /**
     *  Return True if all the constituent error Keys are valid
     *
     *  Note that if the only one or more off-diagonal keys are invalid, we assume that means those terms
     *  are zero, not that the whole FunctorKey is invalid.
     */
    bool isValid() const noexcept;

    //@{
    /// Compare the FunctorKey for equality with another, using its constituent Keys
    bool operator==(CovarianceMatrixKey const& other) const noexcept;
    bool operator!=(CovarianceMatrixKey const& other) const noexcept { return !(*this == other); }
    //@}

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept;

private:
    ErrKeyArray _err;
    CovarianceKeyArray _cov;
};
}  // namespace table
}  // namespace afw
}  // namespace lsst

namespace std {
template <typename T>
struct hash<lsst::afw::table::PointKey<T>> {
    using argument_type = lsst::afw::table::PointKey<T>;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const noexcept { return obj.hash_value(); }
};

template <typename T>
struct hash<lsst::afw::table::BoxKey<T>> {
    using argument_type = lsst::afw::table::BoxKey<T>;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const noexcept { return obj.hash_value(); }
};

template <>
struct hash<lsst::afw::table::CoordKey> {
    using argument_type = lsst::afw::table::CoordKey;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const noexcept { return obj.hash_value(); }
};

template <>
struct hash<lsst::afw::table::QuadrupoleKey> {
    using argument_type = lsst::afw::table::QuadrupoleKey;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const noexcept { return obj.hash_value(); }
};

template <>
struct hash<lsst::afw::table::EllipseKey> {
    using argument_type = lsst::afw::table::EllipseKey;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const noexcept { return obj.hash_value(); }
};

template <typename T, int N>
struct hash<lsst::afw::table::CovarianceMatrixKey<T, N>> {
    using argument_type = lsst::afw::table::CovarianceMatrixKey<T, N>;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const noexcept { return obj.hash_value(); }
};
}  // namespace std

#endif  // !AFW_TABLE_aggregates_h_INCLUDED
