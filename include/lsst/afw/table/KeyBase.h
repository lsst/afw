// -*- lsst-c++ -*-
#ifndef AFW_TABLE_KeyBase_h_INCLUDED
#define AFW_TABLE_KeyBase_h_INCLUDED

#include <vector>

#include "lsst/afw/table/misc.h"

namespace lsst { namespace afw { namespace table { 

class BaseRecord;

template <typename T> class Key;

/// @brief A base class for Key that allows subfield keys to be extracted for some field types.
template <typename T>
class KeyBase {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;

};

/// @brief KeyBase specialization for Coord.
template <>
class KeyBase< Coord > {
public:
    static bool const HAS_NAMED_SUBFIELDS = true;

    Key<Angle> getRa() const; ///< @brief Return a subfield key for the 'ra' coordinate.
    Key<Angle> getDec() const; ///< @brief Return a subfield key for the 'dec' coordinate.

#ifndef SWIG
    static char const * subfields[];
#endif
};

/// @brief KeyBase specialization for Point.
template <typename U>
class KeyBase< Point<U> > {
public:
    static bool const HAS_NAMED_SUBFIELDS = true;

    Key<U> getX() const; ///< @brief Return a subfield key for the 'x' coordinate.
    Key<U> getY() const; ///< @brief Return a subfield key for the 'y' coordinate.

    static char const * subfields[];
};

/// @brief KeyBase specialization for Moments.
template <typename U>
class KeyBase< Moments<U> > {
public:
    static bool const HAS_NAMED_SUBFIELDS = true;

    Key<U> getIxx() const; ///< @brief Return a subfield key for the 'xx' value.
    Key<U> getIyy() const; ///< @brief Return a subfield key for the 'yy' value.
    Key<U> getIxy() const; ///< @brief Return a subfield key for the 'xy' value.

    static char const * subfields[];
};

/// @brief KeyBase specialization for Arrays.
template <typename U>
class KeyBase< Array<U> > {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;

    std::vector<U> extractVector(BaseRecord const & record) const;

    void assignVector(BaseRecord & record, std::vector<U> const & values) const;

    Key<U> operator[](int i) const; ///< @brief Return a subfield key for the i-th element of the array.

    Key< Array<U> > slice(int begin, int end) const; ///< @brief Return a key for a range of elements
};

/// @brief KeyBase specialization for arbitrarily-size covariance matrices.
template <typename U>
class KeyBase< Covariance<U> > {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;

    ///< @brief Return a subfield key for element (i,j) of the covariance matrix.
    Key<U> operator()(int i, int j) const;
};

/// @brief KeyBase specialization for point covariance matrices.
template <typename U>
class KeyBase< Covariance< Point<U> > > {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;

    ///< @brief Return a subfield key for element (i,j) of the covariance matrix.
    Key<U> operator()(int i, int j) const;
};

/// @brief KeyBase specialization for moments covariance matrices.
template <typename U>
class KeyBase< Covariance< Moments<U> > > {
public:
    static bool const HAS_NAMED_SUBFIELDS = false;

    ///< @brief Return a subfield key for element (i,j) of the covariance matrix.
    Key<U> operator()(int i, int j) const;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_KeyBase_h_INCLUDED
