// -*- c++ -*-
#ifndef AFW_TABLE_TableInterface_h_INCLUDED
#define AFW_TABLE_TableInterface_h_INCLUDED

#include "lsst/afw/table/config.h"

#include "boost/mpl/int.h"

#include "lsst/afw/table/RecordInterface.h"
#include "lsst/afw/table/detail/Iterator.h"
#include "lsst/afw/table/detail/TableBase.h"

namespace lsst { namespace afw { namespace table {

template <typename RecordT, typename TableAuxT=detail::TableAux>
class TableInterface : public detail::TableBase {
public:

    typedef RecordT Record;

protected:

    typedef TableAuxT TableAux;
    typedef typename Record::Aux RecordAux;

    boost::shared_ptr<TableAux> getAux() const {
        return boost::static_pointer_cast<TableAux>(detail::TableBase::getAux());
    }
    

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableInterface_h_INCLUDED
