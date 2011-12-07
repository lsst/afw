// -*- lsst-c++ -*-
#ifndef LSST_AFW_TABLE_IO_base_h_INCLUDED
#define LSST_AFW_TABLE_IO_base_h_INCLUDED



#include "boost/noncopyable.hpp"

#include "lsst/afw/table/LayoutMapper.h"
#include "lsst/afw/table/TableBase.h"

namespace lsst { namespace afw { namespace table { namespace io {

class Reader : private boost::noncopyable {
public:

    virtual ~Reader() {}

    Layout const getLayout() const { return _layout;}

    int getRecordCount() const { return _recordCount; }

protected:

    /**
     *  @brief To be called by (most-)derived class constructor, or when reusing the Reader on a
     *         new file.
     *
     *  Will call loadLayout() and loadRecordCount().
     */
    void initialize();

    virtual Layout readLayout() const = 0;

    virtual int readRecordCount() const = 0;

    virtual void readData(TableBase & output, LayoutMapper const & mapper) const = 0;

private:
    int _recordCount;
    Layout _layout;
};

class Writer : private boost::noncopyable {
public:

    virtual ~Writer() {}

protected:

    void writeData(TableBase const & output, LayoutMapper const & mapper) const = 0;

};

}}}} // namespace lsst::afw::table::io

#endif // !LSST_AFW_TABLE_IO_base_h_INCLUDED
