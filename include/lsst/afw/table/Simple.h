// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#ifndef AFW_TABLE_Simple_h_INCLUDED
#define AFW_TABLE_Simple_h_INCLUDED

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/IdFactory.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/io/FitsWriter.h"

namespace lsst { namespace afw { namespace table {

class SimpleRecord;
class SimpleTable;

template <typename RecordT> class SimpleCatalogT;

/**
 *  @brief Record class that must contain a unique ID field and a celestial coordinate field.
 */
class SimpleRecord : public BaseRecord {
public:

    typedef SimpleTable Table;
    typedef SimpleCatalogT<SimpleRecord> Catalog;
    typedef SimpleCatalogT<SimpleRecord const> ConstCatalog;

    CONST_PTR(SimpleTable) getTable() const {
        return boost::static_pointer_cast<SimpleTable const>(BaseRecord::getTable());
    }

    //@{
    /// @brief Convenience accessors for the keys in the minimal reference schema.
    RecordId getId() const;
    void setId(RecordId id);

    IcrsCoord getCoord() const;
    void setCoord(IcrsCoord const & coord);
    void setCoord(Coord const & coord);

    Angle getRa() const;
    void setRa(Angle ra);

    Angle getDec() const;
    void setDec(Angle dec);
    //@}

protected:

    SimpleRecord(PTR(SimpleTable) const & table);

};

/**
 *  @brief Table class that must contain a unique ID field and a celestial coordinate field.
 */
class SimpleTable : public BaseTable {
public:

    typedef SimpleRecord Record;
    typedef SimpleCatalogT<Record> Catalog;
    typedef SimpleCatalogT<Record const> ConstCatalog;

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If null, record IDs will default to zero.
     *
     *  Note that not passing an IdFactory at all will call the other override of make(), which will
     *  set the ID factory to IdFactory::makeSimple().
     */
    static PTR(SimpleTable) make(Schema const & schema, PTR(IdFactory) const & idFactory);

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *
     *  This overload sets the ID factory to IdFactory::makeSimple().
     */
    static PTR(SimpleTable) make(Schema const & schema) { return make(schema, IdFactory::makeSimple()); }

    /**
     *  @brief Return a minimal schema for Simple tables and records.
     *
     *  The returned schema can and generally should be modified further,
     *  but many operations on SimpleRecords will assume that at least the fields
     *  provided by this routine are present.
     */
    static Schema makeMinimalSchema() { return getMinimalSchema().schema; }

    /**
     *  @brief Return true if the given schema is a valid SimpleTable schema.
     *  
     *  This will always be true if the given schema was originally constructed
     *  using makeMinimalSchema(), and will rarely be true otherwise.
     */
    static bool checkSchema(Schema const & other) {
        return other.contains(getMinimalSchema().schema);
    }

    /// @brief Return the object that generates IDs for the table (may be null).
    PTR(IdFactory) getIdFactory() { return _idFactory; }

    /// @brief Return the object that generates IDs for the table (may be null).
    CONST_PTR(IdFactory) getIdFactory() const { return _idFactory; }

    //@{
    /**
     *  Get keys for standard fields shared by all references.
     *
     *  These keys are used to implement getters and setters on SimpleRecord.
     */
    /// @brief Key for the unique ID.
    static Key<RecordId> getIdKey() { return getMinimalSchema().id; }
    /// @brief Key for the celestial coordinates.
    static Key<Coord> getCoordKey() { return getMinimalSchema().coord; }
    //@}

    /// @copydoc BaseTable::clone
    PTR(SimpleTable) clone() const { return boost::static_pointer_cast<SimpleTable>(_clone()); }

    /// @copydoc BaseTable::makeRecord
    PTR(SimpleRecord) makeRecord() { return boost::static_pointer_cast<SimpleRecord>(_makeRecord()); }

    /// @copydoc BaseTable::copyRecord
    PTR(SimpleRecord) copyRecord(BaseRecord const & other) {
        return boost::static_pointer_cast<SimpleRecord>(BaseTable::copyRecord(other));
    }

    /// @copydoc BaseTable::copyRecord
    PTR(SimpleRecord) copyRecord(BaseRecord const & other, SchemaMapper const & mapper) {
        return boost::static_pointer_cast<SimpleRecord>(BaseTable::copyRecord(other, mapper));
    }

protected:

    SimpleTable(Schema const & schema, PTR(IdFactory) const & idFactory);

    SimpleTable(SimpleTable const & other);

private:

    // Struct that holds the minimal schema and the special keys we've added to it.
    struct MinimalSchema {
        Schema schema;
        Key<RecordId> id;
        Key<Coord> coord;

        MinimalSchema();
    };
    
    // Return the singleton minimal schema.
    static MinimalSchema & getMinimalSchema();

    friend class io::FitsWriter;

     // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    virtual PTR(io::FitsWriter) makeFitsWriter(io::FitsWriter::Fits * fits) const;

    PTR(IdFactory) _idFactory;        // generates IDs for new records
};

#ifndef SWIG

/**
 *  @brief Custom catalog class for SimpleRecord/Table.
 */
template <typename RecordT>
class SimpleCatalogT : public CatalogT<RecordT> {
    typedef CatalogT<RecordT> Base;
public:

    typedef RecordT Record;
    typedef typename Record::Table Table;

    typedef typename Base::iterator iterator;
    typedef typename Base::const_iterator const_iterator;

    using Base::isSorted;
    using Base::sort;
    using Base::find;

    /// @brief Return true if the vector is in ascending ID order.
    bool isSorted() const { return this->isSorted(SimpleTable::getIdKey()); }

    /// @brief Sort the vector in-place by ID.
    void sort() { this->sort(SimpleTable::getIdKey()); }

    //@{
    /**
     *  @brief Return an iterator to the record with the given ID.
     *
     *  @note The vector must be sorted in ascending ID order before calling find (i.e. 
     *        isSorted() must be true).
     *
     *  Returns end() if the Record cannot be found.
     */
    iterator find(RecordId id) { return this->find(id, SimpleTable::getIdKey()); }
    const_iterator find(RecordId id) const { return this->find(id, SimpleTable::getIdKey()); }
    //@}

    /**
     *  @brief Construct a vector from a table (or nothing).
     *
     *  A vector with no table is considered invalid; a valid table must be assigned to it
     *  before it can be used.
     */
    explicit SimpleCatalogT(PTR(Table) const & table = PTR(Table)()) : Base(table) {}

    /// @brief Construct a vector from a schema, creating a table with Table::make(schema).
    explicit SimpleCatalogT(Schema const & schema) : Base(schema) {}

    /**
     *  @brief Construct a vector from a table and an iterator range.
     *
     *  If deep is true, new records will be created using table->copyRecord before being inserted.
     *  If deep is false, records will be not be copied, but they must already be associated with
     *  the given table.  The table itself is never deep-copied.
     *
     *  The iterator must dereference to a record reference or const reference rather than a pointer,
     *  but should be implicitly convertible to a record pointer as well (see CatalogIterator).
     *
     *  If InputIterator models RandomAccessIterator (according to std::iterator_traits) and deep
     *  is true, table->preallocate will be used to ensure that the resulting records are
     *  contiguous in memory and can be used with ColumnView.  To ensure this is the case for
     *  other iterator types, the user must preallocate the table manually.
     */
    template <typename InputIterator>
    SimpleCatalogT(PTR(Table) const & table, InputIterator first, InputIterator last, bool deep=false) :
        Base(table, first, last, deep)
    {}

    /**
     *  @brief Shallow copy constructor from a container containing a related record type.
     *
     *  This conversion only succeeds if OtherRecordT is convertible to RecordT and OtherTable is
     *  convertible to Table.
     */
    template <typename OtherRecordT>
    SimpleCatalogT(SimpleCatalogT<OtherRecordT> const & other) : Base(other) {}

    /// Read a FITS binary table.
    static SimpleCatalogT readFits(std::string const & filename) {
        return io::FitsReader::apply<SimpleCatalogT>(filename);
    }

};

typedef SimpleCatalogT<SimpleRecord> SimpleCatalog;
typedef SimpleCatalogT<SimpleRecord const> ConstSimpleCatalog;

inline RecordId SimpleRecord::getId() const { return get(SimpleTable::getIdKey()); }
inline void SimpleRecord::setId(RecordId id) { set(SimpleTable::getIdKey(), id); }

inline IcrsCoord SimpleRecord::getCoord() const { return get(SimpleTable::getCoordKey()); }
inline void SimpleRecord::setCoord(IcrsCoord const & coord) { set(SimpleTable::getCoordKey(), coord); }
inline void SimpleRecord::setCoord(Coord const & coord) { set(SimpleTable::getCoordKey(), coord); }

inline Angle SimpleRecord::getRa() const { return get(SimpleTable::getCoordKey().getRa()); }
inline void SimpleRecord::setRa(Angle ra) { set(SimpleTable::getCoordKey().getRa(), ra); }

inline Angle SimpleRecord::getDec() const { return get(SimpleTable::getCoordKey().getDec()); }
inline void SimpleRecord::setDec(Angle dec) { set(SimpleTable::getCoordKey().getDec(), dec); }

#endif // !SWIG

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Simple_h_INCLUDED
