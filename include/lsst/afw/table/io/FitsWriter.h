// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_FitsWriter_h_INCLUDED
#define AFW_TABLE_IO_FitsWriter_h_INCLUDED

#include "boost/shared_ptr.hpp"

#include "lsst/afw/fits.h"
#include "lsst/afw/table/io/Writer.h"

namespace lsst { namespace afw { namespace table { namespace io {

class FitsWriter : public Writer {
public:

    typedef afw::fits::Fits Fits;

    template <typename ContainerT>
    static void apply(std::string const & filename, ContainerT const & container) {
        Fits fits = Fits::createFile(filename.c_str());
        fits.checkStatus();
        PTR(FitsWriter) writer 
            = boost::static_pointer_cast<TableBase const>(container.getTable())->makeFitsWriter(&fits);
        writer->write(container);
        fits.closeFile();
        fits.checkStatus();
    }

    explicit FitsWriter(Fits * fits) : _fits(fits) {}

protected:

    virtual void _writeTable(CONST_PTR(TableBase) const & table);
    virtual void _writeRecord(RecordBase const & source);

    Fits * _fits;
    std::size_t _row;

private:
    
    struct ProcessRecords;

    boost::shared_ptr<ProcessRecords> _processor;

};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_FitsWriter_h_INCLUDED
