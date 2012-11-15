// -*- lsst-c++ -*-
#ifndef AFW_TABLE_generators_h_INCLUDED
#define AFW_TABLE_generators_h_INCLUDED

#include <vector>

#include "boost/make_shared.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table {

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
 *  without templates or copying the catalog.
 */
class RecordInputGenerator {
public:

    /// @brief Return the schema shared by all records in the sequence.
    Schema getSchema() const { return _schema; }

    /// @brief Return the number of records in the sequence (less than 0 if unknown).
    int getRecordCount() const { return _recordCount; }

    /// @brief Return the next record in the sequence, or an empty pointer if at the end.
    virtual CONST_PTR(BaseRecord) next() = 0;

    /**
     *  @brief Create a RecordInputGenerator from a schema and a random-access iterator range.
     *
     *  The user is responsible for ensuring that the given iterator pair is not invalidated
     *  while the returned RecordInputGenerator exists.
     */
    template <typename IteratorT>
    static PTR(RecordInputGenerator) make(Schema const & schema, IteratorT begin, IteratorT end);

    virtual ~RecordInputGenerator() {}

protected:

    RecordInputGenerator(Schema const & schema, int recordCount) :
        _schema(schema), _recordCount(recordCount) {}

private:

    template <typename IteratorT> class RangeRecordInputGenerator;

    Schema _schema;
    int _recordCount;
};

// Private implementation class for RecordInputGenerator::make.
// Can't go in a source file because we don't know all of the possible types
// for IteratorT.
template <typename IteratorT>
class RecordInputGenerator::RangeRecordInputGenerator : public RecordInputGenerator {
public:
    
    virtual CONST_PTR(BaseRecord) next() {
        if (_current == _end) return CONST_PTR(BaseRecord)();
        return _current;
    }
    
    RangeRecordInputGenerator(Schema const & schema, IteratorT begin, IteratorT end) :
        RecordInputGenerator(schema, end - begin), _current(begin), _end(end) {}
    
private:
    IteratorT _current;
    IteratorT _end;
};

template <typename IteratorT>
PTR(RecordInputGenerator) RecordInputGenerator::make(Schema const & schema, IteratorT begin, IteratorT end) {
    return boost::make_shared< RangeRecordInputGenerator<IteratorT> >(schema, begin, end);
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
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_generators_h_INCLUDED
