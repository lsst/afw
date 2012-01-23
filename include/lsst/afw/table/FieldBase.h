// -*- lsst-c++ -*-
#ifndef AFW_TABLE_FieldBase_h_INCLUDED
#define AFW_TABLE_FieldBase_h_INCLUDED

#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/preprocessor/punctuation/paren.hpp"
#include "Eigen/Core"

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/ndarray.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/coord.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/Covariance.h"
#include "lsst/afw/table/KeyBase.h"

#define AFW_TABLE_SCALAR_FIELD_TYPE_N 5
#define AFW_TABLE_SCALAR_FIELD_TYPES                                    \
    RecordId, boost::int32_t, float, double, Angle
#define AFW_TABLE_SCALAR_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_SCALAR_FIELD_TYPES BOOST_PP_RPAREN()

#define AFW_TABLE_ARRAY_FIELD_TYPE_N 2
#define AFW_TABLE_ARRAY_FIELD_TYPES             \
    float, double
#define AFW_TABLE_ARRAY_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_ARRAY_FIELD_TYPES BOOST_PP_RPAREN()

#define AFW_TABLE_FIELD_TYPE_N 20
#define AFW_TABLE_FIELD_TYPES                                   \
    AFW_TABLE_SCALAR_FIELD_TYPES,                               \
    Flag, Coord,  \
    Array<float>, Array<double>,                                \
    Point<int>, Point<float>, Point<double>,                    \
    Moments<float>, Moments<double>,                            \
    Covariance<float>, Covariance<double>,                      \
    Covariance< Point<float> >, Covariance< Point<double> >,    \
    Covariance< Moments<float> >, Covariance< Moments<double> >
#define AFW_TABLE_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_FIELD_TYPES BOOST_PP_RPAREN()

namespace lsst { namespace afw { namespace table {

namespace detail {

class TableImpl;

 } // namespace detail

/**
 *  @brief Field base class specialization for scalars.
 */
template <typename T>
struct FieldBase {

    typedef T Value;        ///< @brief the type returned by RecordBase::get
    typedef T & Reference;  ///< @brief the type returned by RecordBase::operator[] (non-const)
    typedef T const & ConstReference;  ///< @brief the type returned by RecordBase::operator[] (const)
    typedef T Element;      ///< @brief the type of subfields (the same as the type itself for scalars)

    /// @brief Return the number of subfield elements (always one for scalars).
    int getElementCount() const { return 1; }

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

#ifndef SWIG_BUG_3465431_FIXED
    // SWIG uses this template to define the interface for the other specializations.
    // We can add other methods to full specializations using %extend, but we can't add
    // constructors that way.
    FieldBase() {}
    FieldBase(int) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    static FieldBase makeDefault() { return FieldBase(); }

    void stream(std::ostream & os) const {}

    Reference getReference(Element * p, ndarray::Manager::Ptr const &) const { return *p; }

    ConstReference getConstReference(Element const * p, ndarray::Manager::Ptr const &) const { return *p; }

    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const { return *p; }

    void setValue(Element * p, ndarray::Manager::Ptr const &, Value v) const { *p = v; }

};

/**
 *  @brief Field base class specialization for Coord.
 *
 *  Coord fields are always stored in the ICRS system.  You can assign Coords in other
 *  systems to a record, but this will result in a conversion to ICRS, and returned
 *  values will always be IcrsCoords.
 */
template <>
struct FieldBase< Coord > {

    /// @brief the type returned by RecordBase::get (coord::IcrsCoord, in this case).
    typedef lsst::afw::coord::IcrsCoord Value;

    /// @brief the type of subfields
    typedef lsst::afw::geom::Angle Element;

    /// @brief Return the number of subfield elements (always two for points).
    int getElementCount() const { return 2; }

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

#ifndef SWIG_BUG_3465431_FIXED
    // SWIG uses this template to define the interface for the other specializations.
    // We can add other methods to full specializations using %extend, but we can't add
    // constructors that way.
    FieldBase() {}
    FieldBase(int) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    static FieldBase makeDefault() { return FieldBase(); }

    void stream(std::ostream & os) const {}

    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        return Value(p[0], p[1]);
    }

    void setValue(Element * p, ndarray::Manager::Ptr const &, Value const & v) const {
        p[0] = v.getRa();
        p[1] = v.getDec();
    }

    void setValue(Element * p, ndarray::Manager::Ptr const & m, afw::coord::Coord const & v) const {
        setValue(p, m, v.toIcrs());
    }
};

/**
 *  @brief Field base class specialization for points.
 */
template <typename U>
struct FieldBase< Point<U> > {

    /**
     *  @brief the type returned by RecordBase::get
     *
     *  This will be geom::Point2I when U is an integer, and geom::Point2D when U is float or double.
     */
    typedef typename boost::mpl::if_<boost::is_same<U,int>,geom::Point2I,geom::Point2D>::type Value;

    /// @brief the type of subfields
    typedef U Element;

    /// @brief Return the number of subfield elements (always two for points).
    int getElementCount() const { return 2; }

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

#ifndef SWIG_BUG_3465431_FIXED
    // SWIG uses this template to define the interface for the other specializations.
    // We can add other methods to full specializations using %extend, but we can't add
    // constructors that way.
    FieldBase() {}
    FieldBase(int) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    static FieldBase makeDefault() { return FieldBase(); }

    void stream(std::ostream & os) const {}

    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const { return Value(p[0], p[1]); }

    void setValue(Element * p, ndarray::Manager::Ptr const &, Value const & v) const {
        p[0] = v.getX();
        p[1] = v.getY();
    }
};

/**
 *  @brief Field base class specialization for shapes.
 */
template <typename U>
struct FieldBase< Moments<U> > {

    typedef afw::geom::ellipses::Quadrupole Value; ///< @brief the type returned by RecordBase::get
    typedef U Element; /// @brief the type of subfields

    /// @brief Return the number of subfield elements (always three for shapes).
    int getElementCount() const { return 3; }

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

#ifndef SWIG_BUG_3465431_FIXED
    // SWIG uses this template to define the interface for the other specializations.
    // We can add other methods to full specializations using %extend, but we can't add
    // constructors that way.
    FieldBase() {}
    FieldBase(int) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    static FieldBase makeDefault() { return FieldBase(); }

    void stream(std::ostream & os) const {}

    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        return Value(p[0], p[1], p[2]);
    }

    void setValue(Element * p, ndarray::Manager::Ptr const &, Value const & v) const {
        p[0] = v.getIXX();
        p[1] = v.getIYY();
        p[2] = v.getIXY();
    }
};

/**
 *  @brief Field base class specialization for arrays.
 */
template <typename U>
struct FieldBase< Array<U> > {

    typedef lsst::ndarray::Array<U const,1,1> Value; ///< @brief the type returned by RecordBase::get

    /// @brief the type returned by RecordBase::operator[]
    typedef lsst::ndarray::ArrayRef<U,1,1> Reference;

    /// @brief the type returned by RecordBase::operator[] (const)
    typedef lsst::ndarray::ArrayRef<U const,1,1> ConstReference;

    typedef U Element;  ///< @brief the type of subfields and array elements

    /**
     *  @brief Construct a FieldBase with the given size.
     *
     *  This constructor is implicit and has an invalid default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.  In other words,
     *  it allows one to construct a 25-element array field like this:
     *  @code
     *  Field< Array<float> >("name", "documentation", 25);
     *  @endcode
     *  ...even though the third argument to the Field constructor takes a FieldBase, not an int.
     */
    FieldBase(int size=-1) : _size(size) {
        if (size < 0) throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Size must be provided when constructing an array field."
        );
    }

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

    /// @brief Return the number of subfield elements (equal to the size of the array).
    int getElementCount() const { return _size; }

    /// @brief Return the size of the array (equal to the number of subfield elements).
    int getSize() const { return _size; }

protected:

    static FieldBase makeDefault() { return FieldBase(0); }

    void stream(std::ostream & os) const { os << ", size=" << _size; }

    Reference getReference(Element * p, ndarray::Manager::Ptr const & m) const {
        return ndarray::detail::ArrayAccess< Reference >::construct(p, makeCore(m));
    }

    ConstReference getConstReference(Element const * p, ndarray::Manager::Ptr const & m) const {
        return ndarray::detail::ArrayAccess< ConstReference >::construct(p, makeCore(m));
    }

    Value getValue(Element const * p, ndarray::Manager::Ptr const & m) const {
        return ndarray::detail::ArrayAccess< Value >::construct(p, makeCore(m));
    }

    template <typename T, typename Derived>
    void setValue(
        Element * p, ndarray::Manager::Ptr const &, ndarray::ExpressionBase<Derived> const & value
    ) const {
        if (value.template getShape<0>() != _size) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                "Incorrect size in array field assignment."
            );
        }
        for (int i = 0; i < _size; ++i) p[i] = value[i];
    }

private:

    ndarray::detail::Core<1> makeCore(ndarray::Manager::Ptr const & manager) {
        return ndarray::detail::Core<1>(ndarray::makeVector(_size), manager);
    }

    int _size;
};

/**
 *  @brief Field base class specialization for dynamically-sized covariance matrices.
 *
 *  Covariance matrices are always square and symmetric, and fields are stored packed 
 *  using the same scheme as LAPACK's UPLO=U:
 *  { (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ... }
 *
 *  These elements are packed and unpacked into a dense Eigen matrix when getting and setting
 *  the field.
 */
template <typename U>
struct FieldBase< Covariance<U> > {

    /// @brief the type returned by RecordBase::get
    typedef Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> Value; 

    typedef U Element;    ///< @brief the type of subfields and matrix elements

    /**
     *  @brief Construct a FieldBase with the given size.
     *
     *  This constructor is implicit and has an invalid default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.  In other words,
     *  it allows one to construct a 25x25 covariance field like this:
     *  @code
     *  Field< Covariance<float> >("name", "documentation", 25);
     *  @endcode
     *  ...even though the third argument to the Field constructor takes a FieldBase, not an int.
     */
    FieldBase(int size=-1) : _size(size) {
        if (size < 0) throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Size must be provided when constructing a covariance field."
        );
    }

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

    /// @brief Return the number of subfield elements (the packed size of the covariance matrix).
    int getElementCount() const { return getPackedSize(); }

    /// @brief Return the number of rows/columns of the covariance matrix.
    int getSize() const { return _size; }
    
    /// @brief Return the packed size of the covariance matrix.
    int getPackedSize() const { return detail::computeCovariancePackedSize(_size); }
    
protected:

    static FieldBase makeDefault() { return FieldBase(0); }

    void stream(std::ostream & os) const { os << ", size=" << _size; }

    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        Value m(_size, _size);
        for (int i = 0; i < _size; ++i) {
            for (int j = 0; j < _size; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    template <typename Derived>
    void setValue(
        Element * p, ndarray::Manager::Ptr const &, Eigen::MatrixBase<Derived> const & value
    ) const {
        if (value.rows() != _size || value.cols() != _size) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                "Incorrect size in covariance field assignment."
            );
        }
        for (int i = 0; i < _size; ++i) {
            for (int j = 0; j < _size; ++j) { 
                p[detail::indexCovariance(i, j)] = value(i, j);
            }
        }
    }

    int _size;
};

/**
 *  @brief Field base class specialization for covariance matrices for points.
 *
 *  Covariance fields are stored packed in the following order:
 *  { (x,x), (x,y), (y,y) }
 *
 *  These elements are packed and unpacked into a dense Eigen matrix when getting and setting
 *  the field.
 */
template <typename U>
struct FieldBase< Covariance< Point<U> > > {

    static int const SIZE = 2;
    static int const PACKED_SIZE = 3;

    typedef Eigen::Matrix<U,SIZE,SIZE> Value; ///< @brief the type returned by RecordBase::get
    typedef U Element; ///< @brief the type of subfields and matrix elements

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

    /// @brief Return the number of subfield elements (the packed size of the covariance matrix).
    int getElementCount() const { return getPackedSize(); }

    /// @brief Return the number of rows/columns of the covariance matrix.
    int getSize() const { return SIZE; }

    /// @brief Return the packed size of the covariance matrix.
    int getPackedSize() const { return PACKED_SIZE; }

#ifndef SWIG_BUG_3465431_FIXED
    // SWIG uses this template to define the interface for the other specializations.
    // We can add other methods to full specializations using %extend, but we can't add
    // constructors that way.
    FieldBase() {}
    FieldBase(int) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    static FieldBase makeDefault() { return FieldBase(); }

    void stream(std::ostream & os) const {}

    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        Value m;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    template <typename Derived>
    static void setValue(
        Element * p, ndarray::Manager::Ptr const &, Eigen::MatrixBase<Derived> const & value
    ) {
        BOOST_STATIC_ASSERT( Derived::RowsAtCompileTime == SIZE);
        BOOST_STATIC_ASSERT( Derived::ColsAtCompileTime == SIZE);
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) { 
                p[detail::indexCovariance(i, j)] = value(i, j);
            }
        }
    }

};

/**
 *  @brief Field base class specialization for covariance matrices for shapes.
 *
 *  Covariance fields are stored packed in the following order:
 *  { (xx,xx), (xx,yy), (yy,yy), (xx,xy), (yy,xy), (xy,xy) }
 *
 *  These elements are packed and unpacked into a dense Eigen matrix when getting and setting
 *  the field.
 */
template <typename U>
struct FieldBase< Covariance< Moments<U> > > {

    static int const SIZE = 3;
    static int const PACKED_SIZE = 6;

    typedef Eigen::Matrix<U,SIZE,SIZE> Value; ///< @brief the type returned by RecordBase::get
    typedef U Element; ///< @brief the type of subfields and matrix elements

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

    /// @brief Return the number of subfield elements (the packed size of the covariance matrix).
    int getElementCount() const { return getPackedSize(); }

    /// @brief Return the number of rows/columns of the covariance matrix.
    int getSize() const { return SIZE; }

    /// @brief Return the packed size of the covariance matrix.
    int getPackedSize() const { return PACKED_SIZE; }

#ifndef SWIG_BUG_3465431_FIXED
    // SWIG uses this template to define the interface for the other specializations.
    // We can add other methods to full specializations using %extend, but we can't add
    // constructors that way.
    FieldBase() {}
    FieldBase(int) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicErrorException,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    static FieldBase makeDefault() { return FieldBase(); }

    void stream(std::ostream & os) const {}

    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        Value m;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    template <typename Derived>
    static void setValue(
        Element * p, ndarray::Manager::Ptr const &, Eigen::MatrixBase<Derived> const & value
    ) {
        BOOST_STATIC_ASSERT( Derived::RowsAtCompileTime == SIZE);
        BOOST_STATIC_ASSERT( Derived::ColsAtCompileTime == SIZE);
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) { 
                p[detail::indexCovariance(i, j)] = value(i, j);
            }
        }
    }

};

namespace detail {

typedef boost::mpl::vector< AFW_TABLE_SCALAR_FIELD_TYPES > ScalarFieldTypes;
typedef boost::mpl::vector< AFW_TABLE_FIELD_TYPES > FieldTypes;

} // namespace detail

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_FieldBase_h_INCLUDED
