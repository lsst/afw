// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_FitsReader_h_INCLUDED
#define AFW_TABLE_IO_FitsReader_h_INCLUDED

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/io/Reader.h"

namespace lsst { namespace afw { namespace table { namespace io {

class FitsReader : public Reader {
public:

    typedef afw::fits::Fits Fits;
    
    /**
     *  @brief Factory class used to construct FitsReaders.
     *
     *  The constructor for Factory puts a raw pointer to itself in a global registry.
     *  This means Factory and its subclasses should only be constructed as namespace-scope
     *  objects (so they never go out of scope, and automatically get registered).
     */
    class Factory {
    public:

        virtual PTR(FitsReader) operator()(Fits * fits) const = 0;

        virtual ~Factory() {}

        explicit Factory(std::string const & name);

    };

    /**
     *  @brief Subclass for Factory that constructs a FitsReader whose constructor takes a single Fits*.
     */
    template <typename ReaderT>
    class FactoryT : public Factory {
    public:

        virtual PTR(FitsReader) operator()(Fits * fits) const { return boost::make_shared<ReaderT>(fits); }

        explicit FactoryT(std::string const & name) : Factory(name) {}

    };

    /**
     *  @brief Look for the header key that tells us the type of the FitsReader to use, the make
     *         it using the registered factory.
     */
    static PTR(FitsReader) make(Fits * fits);

    /**
     *  @brief Entry point for reading FITS files into arbitrary containers.
     */
    template <typename ContainerT>
    static ContainerT apply(std::string const & filename) {
        Fits fits = fits::Fits::openFile(filename.c_str(), true);
        fits.checkStatus();
        PTR(FitsReader) reader = make(&fits);
        return reader->template read<ContainerT>();
    }

    explicit FitsReader(Fits * fits) : _fits(fits) {}

protected:

    Schema _readSchema(int nCols=-1) const;

    virtual PTR(TableBase) _readTable();

    virtual void _readRecords(PTR(TableBase) const & table, RecordSink & sink);

    Fits * _fits;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_FitsReader_h_INCLUDED
