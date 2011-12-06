// -*- lsst-c++ -*-
#ifndef AFW_TABLE_FieldBase_h_INCLUDED
#define AFW_TABLE_FieldBase_h_INCLUDED

#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/preprocessor/punctuation/paren.hpp"
#include "Eigen/Core"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/table/Covariance.h"
#include "lsst/afw/table/KeyBase.h"

#define AFW_TABLE_SCALAR_FIELD_TYPE_N 4
#define AFW_TABLE_SCALAR_FIELD_TYPES              \
    int, boost::uint64_t, float, double
#define AFW_TABLE_SCALAR_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_SCALAR_FIELD_TYPES BOOST_PP_RPAREN()

#define AFW_TABLE_ARRAY_FIELD_TYPE_N 2
#define AFW_TABLE_ARRAY_FIELD_TYPES             \
    float, double
#define AFW_TABLE_ARRAY_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_ARRAY_FIELD_TYPES BOOST_PP_RPAREN()

#define AFW_TABLE_FIELD_TYPE_N 17
#define AFW_TABLE_FIELD_TYPES                                   \
    AFW_TABLE_SCALAR_FIELD_TYPES,                               \
    Array<float>, Array<double>,                                \
    Point<int>, Point<float>, Point<double>,                    \
    Shape<float>, Shape<double>,                                \
    Covariance<float>, Covariance<double>,                      \
    Covariance< Point<float> >, Covariance< Point<double> >,    \
    Covariance< Shape<float> >, Covariance< Shape<double> >
#define AFW_TABLE_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_FIELD_TYPES BOOST_PP_RPAREN()

namespace lsst { namespace afw { namespace table {

template <typename T>
struct FieldBase {

    typedef T Value;
    typedef T & Reference;
    typedef T Element;
    
    int getElementCount() const { return 1; }

    std::string getTypeString() const;

protected:

    Reference getReference(Element * p) const { return *p; }

    Value getValue(Element * p) const { return *p; }

    void setValue(Element * p, Value v) const { *p = v; }

};

template <typename U>
struct FieldBase< Point<U> > {

    typedef typename boost::mpl::if_<boost::is_same<U,int>,afw::geom::Point2I,afw::geom::Point2D>::type Value;
    typedef U Element;

    int getElementCount() const { return 2; }

    std::string getTypeString() const;

protected:

    Value getValue(Element * p) const { return Value(p[0], p[1]); }

    void setValue(Element * p, Value const & v) const {
        p[0] = v.getX();
        p[1] = v.getY();
    }
};

template <typename U>
struct FieldBase< Shape<U> > {

    typedef afw::geom::ellipses::Quadrupole Value;
    typedef U Element;

    int getElementCount() const { return 3; }

    std::string getTypeString() const;

protected:

    Value getValue(Element * p) const { return Value(p[0], p[1], p[2]); }

    void setValue(Element * p, Value const & v) const {
        p[0] = v.getIXX();
        p[1] = v.getIYY();
        p[2] = v.getIXY();
    }
};

template <typename U>
struct FieldBase< Array<U> > {

    typedef Eigen::Array<U,Eigen::Dynamic,1> Value;
    typedef Eigen::Map<Value> Reference;
    typedef U Element;

    /**
     *  Constructor is implicit and has an invalid default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.
     */
    FieldBase(int size=-1) : _size(size) {
        if (size < 0) throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Size must be provided when constructing an array field."
        );
    }

    std::string getTypeString() const;

    int getElementCount() const { return _size; }

    int getSize() const { return _size; }

protected:

    Reference getReference(Element * p) const { return Reference(p, _size); }

    Value getValue(Element * p) const { return Value(getReference(p)); }

    template <typename Derived>
    void setValue(Element * p, Eigen::ArrayBase<Derived> const & value) const {
        BOOST_STATIC_ASSERT( Derived::IsVectorAtCompileTime );
        if (value.size() != getSize()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                "Incorrect size in array field assignment."
            );
        }
        for (int i = 0; i < getSize(); ++i) {
            p[i] = value[i];
        }
    }

    int _size;
};

template <typename U>
struct FieldBase< Covariance<U> > {

    typedef Eigen::Matrix<U,Eigen::Dynamic,Eigen::Dynamic> Value;
    typedef U Element;

    /**
     *  Constructor is implicit and has an invalid default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.
     */
    FieldBase(int size=-1) : _size(size) {
        if (size < 0) throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthErrorException,
            "Size must be provided when constructing a covariance field."
        );
    }

    std::string getTypeString() const;

    int getElementCount() const { return getPackedSize(); }

    int getSize() const { return _size; }
    
    int getPackedSize() const { return detail::computeCovariancePackedSize(_size); }
    
protected:

    Value getValue(Element * p) const {
        Value m(_size, _size);
        for (int i = 0; i < _size; ++i) {
            for (int j = 0; j < _size; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    template <typename Derived>
    void setValue(Element * p, Eigen::MatrixBase<Derived> const & value) const {
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

template <typename U>
struct FieldBase< Covariance< Point<U> > > {

    static int const SIZE = 2;
    static int const PACKED_SIZE = 3;

    typedef Eigen::Matrix<U,SIZE,SIZE> Value;
    typedef U Element;

    std::string getTypeString() const;

    int getElementCount() const { return getPackedSize(); }

    int getSize() const { return SIZE; }

    int getPackedSize() const { return PACKED_SIZE; }

protected:

    Value getValue(Element * p) const {
        Value m;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    template <typename Derived>
    static void setValue(Element * p, Eigen::MatrixBase<Derived> const & value) {
        BOOST_STATIC_ASSERT( Derived::RowsAtCompileTime == SIZE);
        BOOST_STATIC_ASSERT( Derived::ColsAtCompileTime == SIZE);
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) { 
                p[detail::indexCovariance(i, j)] = value(i, j);
            }
        }
    }

};

template <typename U>
struct FieldBase< Covariance< Shape<U> > > {

    static int const SIZE = 3;
    static int const PACKED_SIZE = 6;

    typedef Eigen::Matrix<U,SIZE,SIZE> Value;
    typedef U Element;

    std::string getTypeString() const;

    int getElementCount() const { return getPackedSize(); }

    int getSize() const { return SIZE; }

    int getPackedSize() const { return PACKED_SIZE; }

protected:

    Value getValue(Element * p) const {
        Value m;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    template <typename Derived>
    static void setValue(Element * p, Eigen::MatrixBase<Derived> const & value) {
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
