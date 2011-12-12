// -*- lsst-c++ -*-
#ifndef AFW_TABLE_misc_h_INCLUDED
#define AFW_TABLE_misc_h_INCLUDED

#include "boost/cstdint.hpp"

namespace lsst { namespace afw { namespace table {

/// @brief Type used for unique IDs for records.
typedef boost::uint64_t RecordId;

//@{
/**
 *  @brief Tag types used to declare specialized field types.
 *
 *  The documentation for specializations of FieldBase and KeyBase
 *  for more information.
 */
template <typename T> class Point;
template <typename T> class Shape;
template <typename T> class Array;
template <typename T> class Covariance;
//@}

/**
 *  @brief Enum used to specify how a tree iterator works.
 */
enum TreeMode {
    NO_NESTING, ///< Iterate over records in one level of tree without descending to children.
    DEPTH_FIRST ///< Iterate over all (recursive) children of a record before moving onto a sibling.
};

enum LinkMode {
    POINTERS,
    PARENT_ID
};

/**
 *  @brief Class used to attach arbitrary extra data members to table and record classes.
 *
 *  Final table and record classes that need to additional data members will generally
 *  create new subclasses of AuxBase that holds these additional members, and then static_cast
 *  the return value of TableBase::getAux and RecordBase::getAux to the subclass type.
 */
class AuxBase {
public:
    virtual ~AuxBase() {}
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_misc_h_INCLUDED
