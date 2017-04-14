// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_FitsWriter_h_INCLUDED
#define AFW_TABLE_IO_FitsWriter_h_INCLUDED

#include <set>

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace table { namespace io {

/**
 *  Writer object for FITS binary tables.
 *
 *  FitsWriter itself provides support for writing FITS binary tables from base containers.
 *  Derived record/base pairs should derive their own writer from FitsWriter and reimplement
 *  BaseTable::makeFitsWriter to return it.  Subclasses will usually delegate most of the
 *  work back to FitsWriter.
 */
class FitsWriter {
public:

    typedef afw::fits::Fits Fits;

    /**
     *  Driver for writing FITS files.
     *
     *  A container class will usually provide a member function that calls this driver,
     *  which opens the FITS file, calls makeFitsWriter on the container's table, and
     *  then calls Writer::write on it.
     */
    template <typename OutputT, typename ContainerT>
    static void apply(OutputT & output, std::string const & mode, ContainerT const & container, int flags) {
        Fits fits(output, mode, Fits::AUTO_CLOSE | Fits::AUTO_CHECK);
        apply(fits, container, flags);
    }

    /// Low-level driver for writing FITS files, operating on an open FITS file.
    template <typename ContainerT>
    static void apply(Fits & fits, ContainerT const & container, int flags) {
        std::shared_ptr<FitsWriter> writer
            = std::static_pointer_cast<BaseTable const>(container.getTable())->makeFitsWriter(&fits, flags);
        writer->write(container);
    }

    /**
     *  Write records in a container to disk.
     *
     *  The given container must have a getTable() member function that returns a shared_ptr
     *  to a table, and the iterators returned by begin() and end() must dereference to a type
     *  convertible to BaseRecord const &.
     */
    template <typename ContainerT>
    void write(ContainerT const & container) {
        std::set<std::shared_ptr<BaseTable const>> tables;
        for (typename ContainerT::const_iterator i = container.begin(); i != container.end(); ++i) {
            if (i->getTable() != container.getTable()) tables.insert(i->getTable());
        }
        for (std::set<std::shared_ptr<BaseTable const>>::iterator j = tables.begin(); j != tables.end(); ++j) {
            if (
                (**j).getSchema().compare(container.getTable()->getSchema(), Schema::IDENTICAL)
                != Schema::IDENTICAL
            ) {
                throw LSST_EXCEPT(
                    pex::exceptions::LogicError,
                    "Cannot save Catalog with heterogenous schemas"
                );
            }
        }
        _writeTable(container.getTable(), container.size());
        for (typename ContainerT::const_iterator i = container.begin(); i != container.end(); ++i) {
            _writeRecord(*i);
        }
        _finish();
    }

    /// Construct from a wrapped cfitsio pointer.
    explicit FitsWriter(Fits * fits, int flags) : _fits(fits), _flags(flags) {}

protected:

    /// Write a table and its schema.
    virtual void _writeTable(std::shared_ptr<BaseTable const> const & table, std::size_t nRows);

    /// Write an individual record.
    virtual void _writeRecord(BaseRecord const & source);

    /// Finish writing a catalog.
    virtual void _finish() {}

    Fits * _fits;      // wrapped cfitsio pointer
    int _flags;        // subclass-defined flags to control writing
    std::size_t _row;  // which row we're currently processing

private:

    struct ProcessRecords;


    std::shared_ptr<ProcessRecords> _processor; // a private Schema::forEach functor that write records

};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_FitsWriter_h_INCLUDED
