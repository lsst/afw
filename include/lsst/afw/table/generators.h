// -*- lsst-c++ -*-
#ifndef AFW_TABLE_generators_h_INCLUDED
#define AFW_TABLE_generators_h_INCLUDED

#include <vector>

#include "boost/make_shared.hpp"

#include "lsst/base.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw {

namespace fits {

class Fits;

} // namespace fits


namespace table {

/**
 *  @brief An interface base class for objects that fill a sequence of records.
 *
 *  A RecordOutputGenerator doesn't actually create new the new records; that's the responsibility of the
 *  object using the RecordOutputGenerator.  It just sets the values of those new records.
 */
class RecordOutputGenerator : private boost::noncopyable {
public:

    /// Return the schema of the records the RecordSource expects to fill.
    Schema const & getSchema() const { return _schema; }

    /// Return the total number of records the RecordSource will fill (less than 0 if unknown).
    int getRecordCount() const { return _recordCount; }

    /// Fill a single record.  Will be called exactly getRecordCount() times.
    virtual void fill(BaseRecord & record) = 0;

    virtual ~RecordOutputGenerator() {}

protected:

    RecordOutputGenerator(Schema const & schema, int recordCount=1) :
        _schema(schema), _recordCount(recordCount) {}

    Schema _schema;
    int _recordCount;
};

/**
 *  @brief A simple non-STL iterator class that yields a sequence of records.
 *
 *  This is essentially a type-erasure wrapper for a Catalog iterator pair; it provides
 *  a consistent, minimal interface for any kind of catalog iterator range that can be used
 *  without templates.
 */
class RecordInputGenerator {
public:

    /// @brief Return the schema shared by all records in the sequence.
    Schema getSchema() const { return _schema; }

    /// @brief Return the number of records in the sequence (less than 0 if unknown).
    int getRecordCount() const { return _recordCount; }

    /// @brief Return the next record in the sequence, or an empty pointer if at the end.
    virtual CONST_PTR(BaseRecord) next() = 0;

    /// @brief Create a RecordInputGenerator from a catalog.
    template <typename CatT>
    static PTR(RecordInputGenerator) make(CatT const & catalog);

    virtual ~RecordInputGenerator() {}

protected:

    RecordInputGenerator(Schema const & schema, int recordCount) :
        _schema(schema), _recordCount(recordCount) {}

private:

    template <typename CatT> class RangeRecordInputGenerator;

    Schema _schema;
    int _recordCount;
};

// Private implementation class for RecordInputGenerator::make.
// Can't go in a source file because we don't know all of the possible types for CatT.
template <typename CatT>
class RecordInputGenerator::RangeRecordInputGenerator : public RecordInputGenerator {
public:
    
    virtual CONST_PTR(BaseRecord) next() {
        if (_current == _catalog.end()) return CONST_PTR(BaseRecord)();
        return _current;
    }

    RangeRecordInputGenerator(CatT const & catalog) :
        RecordInputGenerator(catalog.getSchema(), catalog.size()),
        _catalog(catalog), _current(_catalog.begin()) {}
    
private:
    CatT const _catalog;
    typename CatT::const_iterator _current;
};

template <typename CatT>
PTR(RecordInputGenerator) RecordInputGenerator::make(CatT const & catalog) {
    return boost::make_shared< RangeRecordInputGenerator<CatT> >(catalog);
}

/**
 *  @brief A simple struct that combines a vector of output generators with a string identifier.
 *
 *  This is used as the output type for classes that serialize themselves into one or more catalogs.
 */
struct RecordOutputGeneratorSet {
    typedef std::vector<PTR(RecordOutputGenerator)> Vector;

    std::string name;
    Vector generators;

    explicit RecordOutputGeneratorSet(std::string const & name_, Vector const & generators_=Vector()) :
        name(name_), generators(generators_) {}

    /**
     *  @brief Use the record generators to write binary table HDUs to a FITS file.
     *
     *  @param[in]   fitsfile        A FITS file object to which additional HDUs will be appended.
     *  @param[in]   kind            A string used for a number of header entries; EXTTYPE="<kind>"
     *                               will be set for all HDUs, and <kind>_NAME and <kind>_NHDU will
     *                               be set in the first HDU, containing name and generators.size().
     *  @param[in]   metadata        Additional metadata to be saved to the first HDU's header.
     */
    void writeFits(
        fits::Fits & fitsfile, std::string const & kind,
        CONST_PTR(daf::base::PropertySet) metadata = CONST_PTR(daf::base::PropertySet)()
    ) const;
};

/**
 *  @brief A simple struct that combines a vector of input generators with a string identifier.
 *
 *  This is used as the input type for classes that serialize themselves into one or more catalogs.
 */
struct RecordInputGeneratorSet {
    typedef std::vector<PTR(RecordInputGenerator)> Vector;

    std::string name;
    Vector generators;

    explicit RecordInputGeneratorSet(std::string const & name_, Vector const & generators_=Vector()) :
        name(name_), generators(generators_) {}

    static RecordInputGeneratorSet readFits(
        fits::Fits & fitsfile,
        PTR(daf::base::PropertySet) metadata = PTR(daf::base::PropertySet)()
    );
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_generators_h_INCLUDED
