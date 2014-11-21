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
#include "ndarray.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/coord.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/KeyBase.h"
#include "lsst/afw/table/types.h"

namespace lsst { namespace afw { namespace table {

namespace detail {

class TableImpl;

/**
 *  @brief Defines the ordering of packed covariance matrices.
 *
 *  This storage is equivalent to LAPACK 'UPLO=U'.
 */
inline int indexCovariance(int i, int j) {
    return (i < j) ? (i + j*(j+1)/2) : (j + i*(i+1)/2);
}

/// Defines the packed size of a covariance matrices.
inline int computeCovariancePackedSize(int size) {
    return size * (size + 1) / 2;
}

} // namespace detail

/**
 *  @brief Field base class default implementation (used for numeric scalars and Angle).
 *
 *  FieldBase is where all the implementation
 */
template <typename T>
struct FieldBase {

    typedef T Value;        ///< @brief the type returned by BaseRecord::get
    typedef T & Reference;  ///< @brief the type returned by BaseRecord::operator[] (non-const)
    typedef T const & ConstReference;  ///< @brief the type returned by BaseRecord::operator[] (const)
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
            lsst::pex::exceptions::LogicError,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const {}

    /// Used to implement RecordBase::operator[] (non-const).
    Reference getReference(Element * p, ndarray::Manager::Ptr const &) const { return *p; }

    /// Used to implement RecordBase::operator[] (const).
    ConstReference getConstReference(Element const * p, ndarray::Manager::Ptr const &) const { return *p; }

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const { return *p; }

    /// Used to implement RecordBase::set.
    void setValue(Element * p, ndarray::Manager::Ptr const &, Value v) const { *p = v; }

};

/**
 *  @brief Field base class specialization for Coord.
 *
 *  Coord fields are always stored in the ICRS system.  You can assign Coords in other
 *  systems to a record, but this will result in a conversion to ICRS, and returned
 *  values will always be IcrsCoords.
 *
 *  Coord fields do not support reference or const reference (operator[]) access.
 */
template <>
struct FieldBase< Coord > {

    /// @brief the type returned by BaseRecord::get (coord::IcrsCoord, in this case).
    typedef IcrsCoord Value;

    /// @brief the type of subfields
    typedef Angle Element;

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
            lsst::pex::exceptions::LogicError,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const {}

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        return Value(p[0], p[1]);
    }

    /// Used to implement RecordBase::set.
    void setValue(Element * p, ndarray::Manager::Ptr const &, Value const & v) const {
        p[0] = v.getRa();
        p[1] = v.getDec();
    }

    /// Used to implement RecordBase::set.
    void setValue(Element * p, ndarray::Manager::Ptr const & m, Coord const & v) const {
        setValue(p, m, v.toIcrs());
    }
};

/**
 *  @brief Field base class specialization for points.
 *
 *  Point fields do not support reference or const reference (operator[]) access.
 */
template <typename U>
struct FieldBase< Point<U> > {

    /**
     *  @brief the type returned by BaseRecord::get
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
            lsst::pex::exceptions::LogicError,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const {}

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const { return Value(p[0], p[1]); }

    /// Used to implement RecordBase::set.
    void setValue(Element * p, ndarray::Manager::Ptr const &, Value const & v) const {
        p[0] = v.getX();
        p[1] = v.getY();
    }
};

/**
 *  @brief Field base class specialization for second moments.
 *
 *  Moments are ordered (xx, yy, xy), matching the order in geom::ellipses::Quadrupole.
 *
 *  Moments fields do not support reference or const reference (operator[]) access.
 */
template <typename U>
struct FieldBase< Moments<U> > {

    typedef afw::geom::ellipses::Quadrupole Value; ///< @brief the type returned by BaseRecord::get
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
            lsst::pex::exceptions::LogicError,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const {}

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        return Value(p[0], p[1], p[2]);
    }

    /// Used to implement RecordBase::set.
    void setValue(Element * p, ndarray::Manager::Ptr const &, Value const & v) const {
        p[0] = v.getIxx();
        p[1] = v.getIyy();
        p[2] = v.getIxy();
    }
};

/**
 *  @brief Field base class specialization for arrays.
 *
 *  The Array tag is used for both fixed-length (same size in every record, accessible via ColumnView)
 *  and variable-length arrays; variable-length arrays are initialized with a negative size.  Ideally,
 *  we'd use complete different tag classes for those two very different types, but boost::variant and
 *  boost::mpl put a limit of 20 on the number of field types, and we're running out.  In a future
 *  reimplementation of afw::table, we should fix this.
 */
template <typename U>
struct FieldBase< Array<U> > {

    typedef ndarray::Array<U const,1,1> Value; ///< @brief the type returned by BaseRecord::get

    /// @brief the type returned by BaseRecord::operator[]
    typedef ndarray::ArrayRef<U,1,1> Reference;

    /// @brief the type returned by BaseRecord::operator[] (const)
    typedef ndarray::ArrayRef<U const,1,1> ConstReference;

    typedef U Element;  ///< @brief the type of subfields and array elements

    /**
     *  @brief Construct a FieldBase with the given size.
     *
     *  A size == 0 indicates a variable-length array.  Negative sizes are not permitted.
     *
     *  This constructor is implicit with a default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.  In other words,
     *  it allows one to construct a 25-element array field like this:
     *  @code
     *  Field< Array<float> >("name", "documentation", 25);
     *  @endcode
     *  ...even though the third argument to the Field constructor takes a FieldBase, not an int.
     */
    FieldBase(int size=0) : _size(size) {
        if (size < 0) throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthError,
            "A non-negative size must be provided when constructing an array field."
        );
    }

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

    /// @brief Return the number of subfield elements (equal to the size of the array).
    int getElementCount() const { return _size; }

    /// @brief Return the size of the array (equal to the number of subfield elements).
    int getSize() const { return _size; }

    /// @brief Return true if the field is variable-length (each record can have a different size array).
    bool isVariableLength() const { return _size == 0; }

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(0); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const { os << ", size=" << _size; }

    /// Used to implement RecordBase::operator[] (non-const).
    Reference getReference(Element * p, ndarray::Manager::Ptr const & m) const {
        if (isVariableLength()) {
            return reinterpret_cast< ndarray::Array<Element,1,1> * >(p)->deep();
        }
        return ndarray::external(p, ndarray::makeVector(_size), ndarray::ROW_MAJOR, m);
    }

    /// Used to implement RecordBase::operator[] (const).
    ConstReference getConstReference(Element const * p, ndarray::Manager::Ptr const & m) const {
        if (isVariableLength()) {
            return reinterpret_cast< ndarray::Array<Element,1,1> const * >(p)->deep();
        }
        return ndarray::external(p, ndarray::makeVector(_size), ndarray::ROW_MAJOR, m);
    }

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const & m) const {
        if (isVariableLength()) {
            return *reinterpret_cast< ndarray::Array<Element,1,1> const * >(p);
        }
        return ndarray::external(p, ndarray::makeVector(_size), ndarray::ROW_MAJOR, m);
    }

    /// Used to implement RecordBase::set; accepts only non-const arrays of the right type,
    /// and allows shallow assignment of variable-length arrays (which is the only kind of
    /// assignment allowed for variable-length arrays - if you want deep assignment, use
    /// operator[] to get a reference and assign to that.
    void setValue(
        Element * p, ndarray::Manager::Ptr const &, ndarray::Array<Element,1,1> const & value
    ) const {
        if (isVariableLength()) {
            *reinterpret_cast< ndarray::Array<Element,1,1>* >(p) = value;
        } else {
            setValueDeep(p, value);
        }
    }

    /// Used to implement RecordBase::set; accepts any ndarray expression.
    template <typename Derived>
    void setValue(
        Element * p, ndarray::Manager::Ptr const &, ndarray::ExpressionBase<Derived> const & value
    ) const {
        if (isVariableLength()) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicErrorException,
                "Assignment to a variable-length array must use a non-const array of the correct type."
            );
        }
        setValueDeep(p, value);
    }

private:

    template <typename Derived>
    void setValueDeep(Element * p, ndarray::ExpressionBase<Derived> const & value) const {
        if (value.template getSize<0>() != _size) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                "Incorrect size in array field assignment."
            );
        }
        for (int i = 0; i < _size; ++i) p[i] = value[i];
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
 *
 *  Covariance fields do not support reference (operator[]) access.
 */
template <typename U>
struct FieldBase< Covariance<U> > {

    /// @brief the type returned by BaseRecord::get
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
            lsst::pex::exceptions::LengthError,
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

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(0); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const { os << ", size=" << _size; }

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        Value m(_size, _size);
        for (int i = 0; i < _size; ++i) {
            for (int j = 0; j < _size; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    /// Used to implement RecordBase::set.
    template <typename Derived>
    void setValue(
        Element * p, ndarray::Manager::Ptr const &, Eigen::MatrixBase<Derived> const & value
    ) const {
        if (value.rows() != _size || value.cols() != _size) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
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

    typedef Eigen::Matrix<U,SIZE,SIZE> Value; ///< @brief the type returned by BaseRecord::get
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
            lsst::pex::exceptions::LogicError,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const {}

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        Value m;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    /// Used to implement RecordBase::set.
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

    typedef Eigen::Matrix<U,SIZE,SIZE> Value; ///< @brief the type returned by BaseRecord::get
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
            lsst::pex::exceptions::LogicError,
            "Constructor disabled (it only appears to exist as a workaround for a SWIG bug)."
        );
    }
#endif

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const {}

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const &) const {
        Value m;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                m(i, j) = p[detail::indexCovariance(i, j)];
            }
        }
        return m;
    }

    /// Used to implement RecordBase::set.
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
 *  @brief Field base class specialization for strings.
 */
template <>
struct FieldBase< std::string > {

    typedef std::string Value; ///< @brief the type returned by BaseRecord::get

    /// @brief the type returned by BaseRecord::operator[]
    typedef char * Reference;

    /// @brief the type returned by BaseRecord::operator[] (const)
    typedef char const * ConstReference;

    typedef char Element;  ///< @brief the type of subfields and array elements

    /**
     *  @brief Construct a FieldBase with the given size.
     *
     *  This constructor is implicit and has an invalid default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.  In other words,
     *  it allows one to construct a 25-character string field like this:
     *  @code
     *  Field< std::string >("name", "documentation", 25);
     *  @endcode
     *  ...even though the third argument to the Field constructor takes a FieldBase, not an int.
     */
    FieldBase(int size=-1);

    /// @brief Return a string description of the field type.
    static std::string getTypeString();

    /// @brief Return the number of subfield elements (equal to the size of the string,
    ///        including a null terminator).
    int getElementCount() const { return _size; }

    /// @brief Return the maximum length of the string, including a null terminator
    ///        (equal to the number of subfield elements).
    int getSize() const { return _size; }

protected:

    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() { return FieldBase(0); }

    /// Defines how Fields are printed.
    void stream(std::ostream & os) const { os << ", size=" << _size; }

    /// Used to implement RecordBase::operator[] (non-const).
    Reference getReference(Element * p, ndarray::Manager::Ptr const & m) const {
        return p;
    }

    /// Used to implement RecordBase::operator[] (const).
    ConstReference getConstReference(Element const * p, ndarray::Manager::Ptr const & m) const {
        return p;
    }

    /// Used to implement RecordBase::get.
    Value getValue(Element const * p, ndarray::Manager::Ptr const & m) const;

    /// Used to implement RecordBase::set
    void setValue(Element * p, ndarray::Manager::Ptr const &, std::string const & value) const;

private:
    int _size;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_FieldBase_h_INCLUDED
