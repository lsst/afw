// -*- c++ -*-
#ifndef AFW_TABLE_RecordInterface_h_INCLUDED
#define AFW_TABLE_RecordInterface_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "lsst/afw/table/detail/RecordBase.h"
#include "lsst/afw/table/Iterator.h"

namespace lsst { namespace afw { namespace table {

template <typename Derived, typename RecordAuxT=detail::AuxBase>
class RecordInterface : public detail::RecordBase {
public:

protected:

    template <typename RecordT, typename TableAuxT> friend class TableInterface;

    typedef RecordAuxT RecordAux;

    boost::shared_ptr<RecordAux> getAux() const {
        return boost::static_pointer_cast<RecordAux>(detail::RecordBase::getAux());
    }

    Record _addChild(boost::shared_ptr<RecordAux> const & aux = boost::shared_ptr<RecordAux>()) {
        return detail::Access::makeRecord<Record>(this->_addChild(aux));
    }

    explicit RecordInterface(detail::RecordBase const & other) : detail::RecordBase(other) {}

    RecordInterface(RecordInterface const & other) : detail::RecordBase(other) {}

    void operator=(RecordInterface const & other) { detail::RecordBase::operator=(other); }

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordInterface_h_INCLUDED
