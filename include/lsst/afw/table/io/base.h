// -*- lsst-c++ -*-
#ifndef LSST_AFW_TABLE_IO_base_h_INCLUDED
#define LSST_AFW_TABLE_IO_base_h_INCLUDED



#include "boost/noncopyable.hpp"

#include "lsst/afw/table/LayoutMapper.h"
#include "lsst/afw/table/TableBase.h"

namespace lsst { namespace afw { namespace table { namespace io {

class Loader : private boost::noncopyable {
public:

    virtual ~Loader() {}

    Layout const getLayout() const { return _layout;}

    int getRecordCount() const { return _recordCount; }

protected:

    /**
     *  @brief To be called by (most-)derived class constructor, or when reusing the Loader on a
     *         new file.
     *
     *  Will call loadLayout() and loadRecordCount().
     */
    void initialize();

    virtual Layout loadLayout() const = 0;

    virtual int loadRecordCount() const = 0;

    virtual void loadData(TableBase & output, LayoutMapper const & mapper) const = 0;

private:
    int _recordCount;
    Layout _layout;
};

}}}} // namespace lsst::afw::table::io

#endif // !LSST_AFW_TABLE_IO_base_h_INCLUDED
