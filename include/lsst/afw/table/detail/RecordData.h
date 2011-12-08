// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordData_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordData_h_INCLUDED

#include "boost/shared_ptr.hpp"
#include "boost/intrusive/set.hpp"
#include "boost/cstdint.hpp"

namespace lsst { namespace afw { namespace table {

/// @brief Type used for unique IDs for records.
typedef boost::uint64_t RecordId;

/**
 *  @brief Enum used to specify how a tree iterator works.
 */
enum TreeMode {
    NO_NESTING, ///< Iterate over records in one level of tree without descending to children.
    DEPTH_FIRST ///< Iterate over all (recursive) children of a record before moving onto a sibling.
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
    typedef boost::shared_ptr<AuxBase> Ptr;
    virtual ~AuxBase() {}
};

namespace detail {

struct RecordData : public boost::intrusive::set_base_hook<> {

    struct Links {
        RecordData * parent;
        RecordData * child;
        RecordData * previous;
        RecordData * next;

        void initialize() {
            parent = 0;
            child = 0;
            previous = 0;
            next = 0;
        }
    };
    
    RecordId id;
    AuxBase::Ptr aux;
    union {
        Links links;
        RecordId parentId;
    };

    bool operator<(RecordData const & other) const { return id < other.id; }

    RecordData() : id(0), aux() { links.initialize(); }
};

struct CompareRecordIdLess {
    bool operator()(RecordId id, RecordData const & data) const {
        return id < data.id;
    }
    bool operator()(RecordData const & data, RecordId id) const {
        return data.id < id;
    }
};

typedef boost::intrusive::set<RecordData> RecordSet;

void setupPointers(RecordSet & records, RecordData * & back);

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_RecordData_h_INCLUDED
