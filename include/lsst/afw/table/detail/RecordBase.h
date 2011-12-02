// -*- lsst-c++ -*-
#ifndef AFW_TABLE_DETAIL_RecordBase_h_INCLUDED
#define AFW_TABLE_DETAIL_RecordBase_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/Layout.h"
#include "lsst/afw/table/detail/Access.h"
#include "lsst/afw/table/detail/RecordData.h"
#include "lsst/afw/table/ModificationFlags.h"

namespace lsst { namespace afw { namespace table { namespace detail {

struct TableImpl;
class TableBase;

class TreeIteratorBase;
class IteratorBase;

class RecordBase : protected ModificationFlags {
public:

    Layout getLayout() const;

    bool hasParent() const { return _data->links.parent; }

    bool hasChildren() const { return _data->links.child; }

    RecordId getId() const { return _data->id; }

    void unlink() const;

    bool isLinked() const { return _data->is_linked(); }

    template <typename T> 
    typename Field<T>::Reference operator[](Key<T> const & key) const {
        assertBit(CAN_SET_FIELD);
        return Access::getReference(key, _data);
    }
    
    template <typename T>
    typename Field<T>::Value get(Key<T> const & key) const {
        return Access::getValue(key, _data);
    }

    template <typename T, typename U>
    void set(Key<T> const & key, U const & value) const {
        assertBit(CAN_SET_FIELD);
        Access::setValue(key, _data, value);
    }

    bool operator==(RecordBase const & other) const {
        return _data == other._data && _table == other._table;
    }

    bool operator!=(RecordBase const & other) const {
        return !this->operator==(other);
    }

    void disable(ModificationFlags::Bit n) { unsetBit(n); }
    void makeReadOnly() { unsetAll(); }

    RecordBase(RecordBase const & other)
        : ModificationFlags(other), _data(other._data), _table(other._table) {}

    ~RecordBase();

protected:

    AuxBase::Ptr getAux() const { return _data->aux; }

    RecordBase _getParent() const {
        if (!_data->links.parent) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundException,
                "Record has no parent."
            );
        }
        return RecordBase(_data->links.parent, _table, *this);
    }

    TreeIteratorBase _beginChildren(TreeMode mode) const;
    TreeIteratorBase _endChildren(TreeMode mode) const;

    RecordBase _addChild(AuxBase::Ptr const & aux = AuxBase::Ptr()) const;
    RecordBase _addChild(RecordId id, AuxBase::Ptr const & aux = AuxBase::Ptr()) const;

    void operator=(RecordBase const & other) {
        ModificationFlags::operator=(other);
        _data = other._data;
        _table = other._table;
    }

private:

    friend class TableBase;
    friend class IteratorBase;
    friend class TreeIteratorBase;


    RecordBase() : ModificationFlags(), _data(0), _table() {}

    RecordBase(
        RecordData * data,
        boost::shared_ptr<TableImpl> const & table,
        ModificationFlags const & flags
    ) : ModificationFlags(flags), _data(data), _table(table)
    {}

    RecordData * _data;
    boost::shared_ptr<TableImpl> _table;
};

}}}} // namespace lsst::afw::table::detail

#endif // !AFW_TABLE_DETAIL_RecordBase_h_INCLUDED
