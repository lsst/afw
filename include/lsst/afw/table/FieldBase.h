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
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/KeyBase.h"
#include "lsst/afw/table/types.h"

namespace lsst {
namespace afw {
namespace table {

namespace detail {

class TableImpl;

/**
 *  Defines the ordering of packed covariance matrices.
 *
 *  This storage is equivalent to LAPACK 'UPLO=U'.
 */
inline std::size_t indexCovariance(std::size_t i, std::size_t j) { return (i < j) ? (i + j * (j + 1) / 2) : (j + i * (i + 1) / 2); }

/// Defines the packed size of a covariance matrices.
inline std::size_t computeCovariancePackedSize(std::size_t size) { return size * (size + 1) / 2; }

}  // namespace detail

/**
 *  Field base class default implementation (used for numeric scalars and lsst::geom::Angle).
 */
template <typename T>
struct FieldBase {
    using Value = T;                  ///< the type returned by BaseRecord::get
    using Reference = T &;             ///< the type returned by BaseRecord::operator[] (non-const)
    using ConstReference = const T &;  ///< the type returned by BaseRecord::operator[] (const)
    using Element = T;                ///< the type of subfields (the same as the type itself for scalars)

    /// Return the number of subfield elements (always one for scalars).
    std::size_t getElementCount() const noexcept { return 1; }

    /// Return a string description of the field type.
    static std::string getTypeString();

    // Only the first of these constructors is valid for this specializations, but
    // it's convenient to be able to instantiate both, since the other is used
    // by other specializations.
    FieldBase() = default;
    FieldBase(std::size_t) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                          "Constructor disabled (this Field type is not sized).");
    }
    FieldBase(FieldBase const &) noexcept = default;
    FieldBase(FieldBase &&) noexcept = default;
    FieldBase &operator=(FieldBase const &) noexcept = default;
    FieldBase &operator=(FieldBase &&) noexcept = default;
    ~FieldBase() noexcept = default;

protected:
    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() noexcept { return FieldBase(); }

    /// Defines how Fields are printed.
    void stream(std::ostream &os) const {}

    /// Used to implement BaseRecord::operator[] (non-const).
    Reference getReference(Element *p, ndarray::Manager::Ptr const &) const { return *p; }

    /// Used to implement BaseRecord::operator[] (const).
    ConstReference getConstReference(Element const *p, ndarray::Manager::Ptr const &) const { return *p; }

    /// Used to implement BaseRecord::get.
    Value getValue(Element const *p, ndarray::Manager::Ptr const &) const { return *p; }

    /// Used to implement BaseRecord::set.
    void setValue(Element *p, ndarray::Manager::Ptr const &, Value v) const { *p = v; }
};

/**
 *  Field base class specialization for arrays.
 *
 *  The Array tag is used for both fixed-length (same size in every record, accessible via ColumnView)
 *  and variable-length arrays; variable-length arrays are initialized with a size of 0.  Ideally,
 *  we'd use complete different tag classes for those two very different types, but this was limited in
 *  the original (pre-C++11) implementation and now it would be an API change.
 */
template <typename U>
struct FieldBase<Array<U> > {
    using Value = ndarray::Array<const U, 1, 1>;  ///< the type returned by BaseRecord::get

    /// the type returned by BaseRecord::operator[]
    using Reference = ndarray::ArrayRef<U, 1, 1>;

    /// the type returned by BaseRecord::operator[] (const)
    using ConstReference = ndarray::ArrayRef<const U, 1, 1>;

    using Element = U;  ///< the type of subfields and array elements

    /**
     *  Construct a FieldBase with the given size.
     *
     *  A size == 0 indicates a variable-length array.  Negative sizes are not permitted.
     *
     *  This constructor is implicit with a default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.  In other words,
     *  it allows one to construct a 25-element array field like this:
     *
     *      Field< Array<float> >("name", "documentation", 25);
     *
     *  ...even though the third argument to the Field constructor takes a FieldBase, not an int.
     */
    FieldBase(size_t size = 0) : _size(size) {
    }

    FieldBase(FieldBase const &) noexcept = default;
    FieldBase(FieldBase &&) noexcept = default;
    FieldBase &operator=(FieldBase const &) noexcept = default;
    FieldBase &operator=(FieldBase &&) noexcept = default;
    ~FieldBase() noexcept = default;

    /// Return a string description of the field type.
    static std::string getTypeString();

    /// Return the number of subfield elements (equal to the size of the array),
    /// or 0 for a variable-length array.
    std::size_t getElementCount() const noexcept { return _size; }

    /// Return the size of the array (equal to the number of subfield elements),
    /// or 0 for a variable-length array.
    std::size_t getSize() const noexcept { return _size; }

    /// Return true if the field is variable-length (each record can have a different size array).
    bool isVariableLength() const noexcept { return _size == 0; }

protected:
    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() noexcept { return FieldBase(0); }

    /// Defines how Fields are printed.
    void stream(std::ostream &os) const { os << ", size=" << _size; }

    /// Used to implement BaseRecord::operator[] (non-const).
    Reference getReference(Element *p, ndarray::Manager::Ptr const &m) const {
        if (isVariableLength()) {
            return reinterpret_cast<ndarray::Array<Element, 1, 1> *>(p)->deep();
        }
        return ndarray::external(p, ndarray::makeVector(_size), ndarray::ROW_MAJOR, m);
    }

    /// Used to implement BaseRecord::operator[] (const).
    ConstReference getConstReference(Element const *p, ndarray::Manager::Ptr const &m) const {
        if (isVariableLength()) {
            return reinterpret_cast<ndarray::Array<Element, 1, 1> const *>(p)->deep();
        }
        return ndarray::external(p, ndarray::makeVector(_size), ndarray::ROW_MAJOR, m);
    }

    /// Used to implement BaseRecord::get.
    Value getValue(Element const *p, ndarray::Manager::Ptr const &m) const {
        if (isVariableLength()) {
            return *reinterpret_cast<ndarray::Array<Element, 1, 1> const *>(p);
        }
        return ndarray::external(p, ndarray::makeVector(_size), ndarray::ROW_MAJOR, m);
    }

    /// Used to implement BaseRecord::set; accepts only non-const arrays of the right type.
    /// Fixed-length arrays are handled by copying the data from `value` to `p` through `p + _size`.
    /// Variable-length arrays are handled by setting `p` to the address of `value`, an ndarray,
    /// hence a shallow copy (ndarray arrays are reference-counted so this will not leak memory).
    /// If you want deep assignment of variable-length data, use operator[] to get a reference
    /// and assign to that.
    void setValue(Element *p, ndarray::Manager::Ptr const &,
                  ndarray::Array<Element, 1, 1> const &value) const {
        if (isVariableLength()) {
            *reinterpret_cast<ndarray::Array<Element, 1, 1> *>(p) = value;
        } else {
            setValueDeep(p, value);
        }
    }

    /// Used to implement BaseRecord::set; accepts any ndarray expression.
    template <typename Derived>
    void setValue(Element *p, ndarray::Manager::Ptr const &,
                  ndarray::ExpressionBase<Derived> const &value) const {
        if (isVariableLength()) {
            throw LSST_EXCEPT(
                    lsst::pex::exceptions::LogicError,
                    "Assignment to a variable-length array must use a non-const array of the correct type.");
        }
        setValueDeep(p, value);
    }

private:
    template <typename Derived>
    void setValueDeep(Element *p, ndarray::ExpressionBase<Derived> const &value) const {
        if (value.template getSize<0>() != static_cast<std::size_t>(_size)) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                              "Incorrect size in array field assignment.");
        }
        for (std::size_t i = 0; i < _size; ++i) p[i] = value[i];
    }

    std::size_t _size;
};

/**
 *  Field base class specialization for strings.
 */
template <>
struct FieldBase<std::string> {
    using Value = std::string;  ///< the type returned by BaseRecord::get

    /// the type returned by BaseRecord::operator[]
    using Reference = char *;

    /// the type returned by BaseRecord::operator[] (const)
    using ConstReference = const char *;

    using Element = char;  ///< the type of subfields and array elements

    /**
     *  Construct a FieldBase with the given size.
     *
     *  A size == 0 indicates a variable-length string.  Negative sizes are not permitted.
     *
     *  This constructor is implicit and has an invalid default so it can be used in the Field
     *  constructor (as if it were an int argument) without specializing Field.  In other words,
     *  it allows one to construct a 25-character string field like this:
     *
     *      Field< std::string >("name", "documentation", 25);
     *
     *  ...even though the third argument to the Field constructor takes a FieldBase, not an int.
     */
    FieldBase(std::size_t size=std::numeric_limits<size_t>::max());

    FieldBase(FieldBase const &) noexcept = default;
    FieldBase(FieldBase &&) noexcept = default;
    FieldBase &operator=(FieldBase const &) noexcept = default;
    FieldBase &operator=(FieldBase &&) noexcept = default;
    ~FieldBase() noexcept = default;

    /// Return a string description of the field type.
    static std::string getTypeString();

    /// @brief Return the number of subfield elements (equal to the size of the string,
    ///        including a null terminator), or 0 for a variable-length string.
    std::size_t getElementCount() const noexcept { return _size; }

    /// @brief Return the maximum length of the string, including a null terminator
    ///        (equal to the number of subfield elements), or 0 for a variable-length string.
    std::size_t getSize() const noexcept { return _size; }

    /// Return true if the field is variable-length (each record can have a different size array).
    bool isVariableLength() const noexcept { return _size == 0; }

protected:
    /// Needed to allow Keys to be default-constructed.
    static FieldBase makeDefault() noexcept { return FieldBase(0); }

    /// Defines how Fields are printed.
    void stream(std::ostream &os) const { os << ", size=" << _size; }

    /// Used to implement BaseRecord::operator[] (non-const).
    Reference getReference(Element *p, ndarray::Manager::Ptr const &m) const {
        if (isVariableLength()) {
            // Can't be done until C++17, which allows read/write access to std::string's internal buffer
            throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                              "non-const operator[] not supported for variable-length strings");
        } else {
            return p;
        }
    }

    /// Used to implement BaseRecord::operator[] (const).
    ConstReference getConstReference(Element const *p, ndarray::Manager::Ptr const &m) const {
        if (isVariableLength()) {
            return reinterpret_cast<std::string const *>(p)->data();
        } else {
            return p;
        }
    }

    /// Used to implement BaseRecord::get.
    Value getValue(Element const *p, ndarray::Manager::Ptr const &m) const;

    /// Used to implement BaseRecord::set
    /// Fixed-lengths strings are handled by copying the data into `p` through `p + _size`,
    /// nulling extra characters, if any. The data is only null-terminated if value.size() < _size.
    /// Variable-length strings are handled by setting `p` to the address of a `std::string`
    /// that is a copy of `value`
    void setValue(Element *p, ndarray::Manager::Ptr const &, std::string const &value) const;

private:
    std::size_t _size;
};
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_FieldBase_h_INCLUDED
